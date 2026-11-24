#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <algorithm>

inline float squash(float x) {
    if (x < -12.0f) return 0.000006f;
    if (x > 12.0f)  return 0.999994f;
    return 1.0f / (1.0f + std::exp(-x));
}

inline float stretch(float p) {
    p = std::max(0.00001f, std::min(0.99999f, p));
    return std::log(p / (1.0f - p));
}

// --- 1. Bitwise Arithmetic Coder ---
class ArithmeticCoder {
    uint32_t low = 0, high = 0xFFFFFFFF, code = 0;
    uint8_t out_buf = 0, out_bits = 0;
    uint8_t in_buf = 0, in_bits = 0;
    std::ofstream* out = nullptr;
    std::ifstream* in = nullptr;

    void put_bit(int bit) {
        out_buf = (out_buf << 1) | (bit & 1);
        if (++out_bits == 8) {
            out->put((char)out_buf);
            out_buf = 0;
            out_bits = 0;
        }
    }

    uint8_t get_bit() {
        if (in_bits == 0) {
            int c = in->get();
            in_buf = (c == EOF) ? 0 : (uint8_t)c;
            in_bits = 8;
        }
        return (in_buf >> (--in_bits)) & 1;
    }

public:
    void init_encode(std::ofstream& o) { out = &o; low = 0; high = 0xFFFFFFFF; out_buf = 0; out_bits = 0; }
    void init_decode(std::ifstream& i) {
        in = &i; low = 0; high = 0xFFFFFFFF; code = 0; in_bits = 0;
        for (int k = 0; k < 32; ++k) code = (code << 1) | get_bit();
    }

    void encode_bit(int bit, uint32_t p1) {
        uint32_t mid = low + ((uint64_t)(high - low) * p1 >> 12);
        if (bit) high = mid;
        else     low = mid + 1;

        while (((low ^ high) & 0x80000000) == 0) {
            put_bit(low >> 31);
            low <<= 1;
            high = (high << 1) | 1;
        }
    }

    int decode_bit(uint32_t p1) {
        uint32_t mid = low + ((uint64_t)(high - low) * p1 >> 12);
        int bit = (code <= mid) ? 1 : 0;
        if (bit) high = mid;
        else     low = mid + 1;

        while (((low ^ high) & 0x80000000) == 0) {
            low <<= 1;
            high = (high << 1) | 1;
            code = (code << 1) | get_bit();
        }
        return bit;
    }

    void flush() {
        for (int k = 0; k < 32; ++k) {
            put_bit(low >> 31);
            low <<= 1;
        }
        if (out_bits > 0) {
            out_buf <<= (8 - out_bits);
            out->put((char)out_buf);
        }
    }
};

// --- 2. Order-N / Sparse Context Model ---
struct ContextModel {
    std::vector<uint16_t> c0, c1;
    uint32_t mask;
    ContextModel(size_t bits) : c0(1ULL << bits, 1), c1(1ULL << bits, 1), mask((1ULL << bits) - 1) {}

    float predict(uint32_t h) const {
        uint32_t idx = h & mask;
        return (float)c1[idx] / (float)(c0[idx] + c1[idx]);
    }

    void update(uint32_t h, int bit) {
        uint32_t idx = h & mask;
        if (bit) c1[idx]++; else c0[idx]++;
        if (c0[idx] + c1[idx] > 400) {
            c0[idx] = (c0[idx] >> 1) + 1;
            c1[idx] = (c1[idx] >> 1) + 1;
        }
    }
};

// --- 3. Long-Range Match Model (Finds repeated strings anywhere in past history) ---
struct MatchModel {
    std::vector<uint8_t> buf;
    std::vector<uint32_t> table;
    uint32_t mask;
    uint32_t match_pos = 0;
    uint32_t match_len = 0;

    MatchModel(size_t hash_bits = 22) : table(1ULL << hash_bits, 0), mask((1ULL << hash_bits) - 1) {
        buf.reserve(1 << 24);
    }

    // Called once per byte AFTER byte is known
    void update_byte(uint8_t byte) {
        uint32_t pos = (uint32_t)buf.size();
        buf.push_back(byte);
        if (pos < 8) return;

        // Extend existing match or search for a new 6-byte match
        if (match_len > 0) {
            if (buf[match_pos] == byte) {
                match_pos++;
                match_len++;
            } else {
                match_len = 0;
            }
        }

        uint32_t h = 0;
        for (int i = 0; i < 6; ++i) h = h * 1315423911u + buf[pos - i];
        uint32_t idx = h & mask;
        uint32_t candidate = table[idx];
        table[idx] = pos + 1;

        if (match_len == 0 && candidate >= 6 && candidate <= pos) {
            uint32_t len = 0;
            while (len < 64 && candidate > len && buf[candidate - 1 - len] == buf[pos - len]) {
                len++;
            }
            if (len >= 4) {
                match_pos = candidate;
                match_len = len;
            }
        }
    }

    // Predicts current bit_idx (7 down to 0) based on matched byte
    float predict_bit(int bit_idx, uint32_t partial) const {
        if (match_len < 4 || match_pos >= buf.size()) return 0.5f;
        uint8_t predicted_byte = buf[match_pos];
        // Verify bits decoded so far in this byte match the candidate byte
        uint8_t prefix_mask = 0xFF << (bit_idx + 1);
        uint8_t decoded_prefix = (uint8_t)(partial << (bit_idx + 1));
        if ((predicted_byte & prefix_mask) != decoded_prefix) return 0.5f;

        int pred_bit = (predicted_byte >> bit_idx) & 1;
        // Confidence scales smoothly with match length
        float conf = std::min(0.995f, 0.72f + 0.04f * std::min(match_len, 7u));
        return pred_bit ? conf : (1.0f - conf);
    }
};

// --- 4. Context-Selected Neural Mixer (256 specialized mixers) ---
struct ContextNeuralMixer {
    static constexpr int N = 8;
    std::array<std::array<float, N>, 256> w{};
    float lr = 0.035f;

    ContextNeuralMixer() {
        for (int c = 0; c < 256; ++c) w[c].fill(1.0f / N);
    }

    float mix(uint8_t ctx, const std::array<float, N>& s) const {
        float sum = 0.0f;
        const auto& wc = w[ctx];
        for (int i = 0; i < N; ++i) sum += wc[i] * s[i];
        return squash(sum);
    }

    void train(uint8_t ctx, const std::array<float, N>& s, float p, int bit) {
        float err = (float)bit - p;
        auto& wc = w[ctx];
        for (int i = 0; i < N; ++i) wc[i] += lr * err * s[i];
    }
};

// --- 5. Secondary Symbol Estimation (SSE / Probability Calibrator) ---
struct SSE {
    static constexpr int BINS = 33;
    std::vector<std::array<float, BINS>> table;
    uint32_t mask;

    SSE(size_t ctx_bits = 14) : table(1ULL << ctx_bits), mask((1ULL << ctx_bits) - 1) {
        for (auto& row : table) {
            for (int b = 0; b < BINS; ++b) {
                float x = ((float)b / (BINS - 1) - 0.5f) * 16.0f;
                row[b] = squash(x);
            }
        }
    }

    float map(uint32_t ctx, float p, int& bin_out, float& frac_out) const {
        float x = (stretch(p) / 16.0f + 0.5f) * (BINS - 1);
        x = std::max(0.0f, std::min((float)(BINS - 1.0001f), x));
        bin_out = (int)x;
        frac_out = x - bin_out;
        const auto& row = table[ctx & mask];
        return row[bin_out] * (1.0f - frac_out) + row[bin_out + 1] * frac_out;
    }

    void update(uint32_t ctx, int bin, float frac, int bit) {
        auto& row = table[ctx & mask];
        float target = (float)bit;
        row[bin]     += 0.08f * (1.0f - frac) * (target - row[bin]);
        row[bin + 1] += 0.08f * frac * (target - row[bin + 1]);
    }
};

// --- 6. Main Predictor Pipeline ---
void process(bool encode, const std::string& in_path, const std::string& out_path) {
    std::ifstream in(in_path, std::ios::binary);
    std::ofstream out(out_path, std::ios::binary);

    ArithmeticCoder coder;
    uint64_t file_size = 0;

    if (encode) {
        in.seekg(0, std::ios::end);
        file_size = in.tellg();
        in.seekg(0, std::ios::beg);
        out.write(reinterpret_cast<char*>(&file_size), sizeof(file_size));
        coder.init_encode(out);
    } else {
        in.read(reinterpret_cast<char*>(&file_size), sizeof(file_size));
        coder.init_decode(in);
    }

    ContextModel m1(16), m2(20), m3(22), m4(23), mword(22), msparse1(20), msparse2(21);
    MatchModel matcher(22);
    ContextNeuralMixer mixer;
    SSE sse(14);

    uint32_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, word_hash = 0;

    for (uint64_t pos = 0; pos < file_size; ++pos) {
        uint8_t byte = encode ? (uint8_t)in.get() : 0;
        uint32_t partial = 1;

        for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
            uint32_t h1  = (b1 << 8) | partial;
            uint32_t h2  = ((b2 * 1315423911u) ^ (b1 << 8)) ^ partial;
            uint32_t h3  = ((b3 * 2654435761u) ^ (b2 << 8) ^ b1) ^ partial;
            uint32_t h4  = ((b4 * 97531u) ^ (b3 << 16) ^ (b2 << 8) ^ b1) ^ partial;
            uint32_t hw  = (word_hash * 31u) ^ partial;
            uint32_t hs1 = ((b2 * 65599u) ^ partial);
            uint32_t hs2 = ((b3 * 131313u) ^ (b1 << 8) ^ partial);

            std::array<float, 8> stretched = {
                stretch(m1.predict(h1)),
                stretch(m2.predict(h2)),
                stretch(m3.predict(h3)),
                stretch(m4.predict(h4)),
                stretch(mword.predict(hw)),
                stretch(msparse1.predict(hs1)),
                stretch(msparse2.predict(hs2)),
                stretch(matcher.predict_bit(bit_idx, partial))
            };

            float p_mix = mixer.mix((uint8_t)b1, stretched);

            int sse_bin = 0;
            float sse_frac = 0.0f;
            uint32_t sse_ctx = (b1 << 8) | partial;
            float p_sse = sse.map(sse_ctx, p_mix, sse_bin, sse_frac);
            float p = 0.75f * p_sse + 0.25f * p_mix;

            uint32_t p1 = std::max(1u, std::min(4095u, (uint32_t)(p * 4096.0f)));

            int bit;
            if (encode) {
                bit = (byte >> bit_idx) & 1;
                coder.encode_bit(bit, p1);
            } else {
                bit = coder.decode_bit(p1);
                byte = (byte << 1) | bit;
            }

            m1.update(h1, bit);
            m2.update(h2, bit);
            m3.update(h3, bit);
            m4.update(h4, bit);
            mword.update(hw, bit);
            msparse1.update(hs1, bit);
            msparse2.update(hs2, bit);
            mixer.train((uint8_t)b1, stretched, p_mix, bit);
            sse.update(sse_ctx, sse_bin, sse_frac, bit);

            partial = (partial << 1) | bit;
        }

        if (!encode) out.put((char)byte);

        matcher.update_byte(byte);
        b4 = b3; b3 = b2; b2 = b1; b1 = byte;
        if (std::isalpha(byte)) word_hash = word_hash * 131u + std::tolower(byte);
        else word_hash = 0;
    }

    if (encode) coder.flush();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./mini_cmix_v2 [c|d] <input> <output>\n";
        return 1;
    }
    process(argv[1][0] == "c"[0], argv[2], argv[3]);
    return 0;
}
