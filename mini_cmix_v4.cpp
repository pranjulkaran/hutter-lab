#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <random>
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
        if (c0[idx] + c1[idx] > 360) {
            c0[idx] = (c0[idx] >> 1) + 1;
            c1[idx] = (c1[idx] >> 1) + 1;
        }
    }
};

struct IndirectModel {
    std::vector<uint32_t> history;
    uint32_t hist_mask;
    ContextModel cm;

    IndirectModel(size_t hist_bits = 20, size_t cm_bits = 22)
        : history(1ULL << hist_bits, 0), hist_mask((1ULL << hist_bits) - 1), cm(cm_bits) {}

    uint32_t get_hash(uint32_t ctx, uint32_t partial) const {
        uint32_t seq = history[ctx & hist_mask];
        return (seq * 2654435761u) ^ partial;
    }

    float predict(uint32_t ctx, uint32_t partial) const {
        return cm.predict(get_hash(ctx, partial));
    }

    void update_bit(uint32_t ctx, uint32_t partial, int bit) {
        cm.update(get_hash(ctx, partial), bit);
    }

    void update_byte(uint32_t prev_ctx, uint8_t byte) {
        uint32_t idx = prev_ctx & hist_mask;
        history[idx] = (history[idx] << 8) | byte;
    }
};

struct MultiMatchModel {
    static constexpr int WAYS = 4;
    std::vector<uint8_t> buf;
    std::vector<std::array<uint32_t, WAYS>> table8;
    std::vector<uint32_t> table4;
    uint32_t mask8, mask4;
    uint32_t match_pos = 0;
    uint32_t match_len = 0;

    MultiMatchModel() : table8(1ULL << 21), table4(1ULL << 21, 0),
                        mask8((1ULL << 21) - 1), mask4((1ULL << 21) - 1) {
        buf.reserve(1 << 24);
    }

    void update_byte(uint8_t byte) {
        uint32_t pos = (uint32_t)buf.size();
        buf.push_back(byte);
        if (pos < 8) return;

        if (match_len > 0) {
            if (buf[match_pos] == byte) {
                match_pos++;
                match_len++;
            } else {
                match_len = 0;
            }
        }

        uint32_t h4 = 0, h8 = 0;
        for (int i = 0; i < 4; ++i) h4 = h4 * 1315423911u + buf[pos - i];
        for (int i = 0; i < 8; ++i) h8 = h8 * 2654435761u + buf[pos - i];

        uint32_t idx8 = h8 & mask8;
        auto& bucket = table8[idx8];
        uint32_t cand4 = table4[h4 & mask4];
        table4[h4 & mask4] = pos + 1;

        if (match_len < 12) {
            uint32_t best_pos = match_pos;
            uint32_t best_len = match_len;

            auto check_cand = [&](uint32_t cand) {
                if (cand >= 4 && cand <= pos) {
                    uint32_t len = 0;
                    while (len < 256 && cand > len && buf[cand - 1 - len] == buf[pos - len]) {
                        len++;
                    }
                    if (len > best_len) {
                        best_len = len;
                        best_pos = cand;
                    }
                }
            };

            for (int w = 0; w < WAYS; ++w) check_cand(bucket[w]);
            if (best_len < 4) check_cand(cand4);

            if (best_len >= 4) {
                match_pos = best_pos;
                match_len = best_len;
            }
        }

        for (int w = WAYS - 1; w > 0; --w) bucket[w] = bucket[w - 1];
        bucket[0] = pos + 1;
    }

    float predict_bit(int bit_idx, uint32_t partial) const {
        if (match_len < 4 || match_pos >= buf.size()) return 0.5f;
        uint8_t predicted_byte = buf[match_pos];
        uint8_t prefix_mask = 0xFF << (bit_idx + 1);
        uint8_t decoded_prefix = (uint8_t)(partial << (bit_idx + 1));
        if ((predicted_byte & prefix_mask) != decoded_prefix) return 0.5f;

        int pred_bit = (predicted_byte >> bit_idx) & 1;
        float conf = std::min(0.998f, 0.75f + 0.03f * std::min(match_len, 8u));
        return pred_bit ? conf : (1.0f - conf);
    }
};

struct BracketTracker {
    std::vector<uint8_t> stack;
    uint32_t dist = 0;

    void update(uint8_t b) {
        dist = std::min(dist + 1, 255u);
        if (b == '[' || b == '{' || b == '(' || b == '<') {
            stack.push_back(b);
            if (stack.size() > 32) stack.erase(stack.begin());
            dist = 0;
        } else if (!stack.empty()) {
            uint8_t top = stack.back();
            if ((top == '[' && b == ']') || (top == '{' && b == '}') ||
                (top == '(' && b == ')') || (top == '<' && b == '>')) {
                stack.pop_back();
                dist = 0;
            }
        }
    }

    uint32_t context() const {
        uint8_t top = stack.empty() ? 0 : stack.back();
        return ((uint32_t)top << 12) | ((uint32_t)(stack.size() & 0xF) << 8) | (dist >> 2);
    }
};

struct DeepOnlineByteRNN {
    static constexpr int H1 = 64;
    static constexpr int H2 = 64;
    static constexpr int IN1 = 256 + H1;
    static constexpr int IN2 = H1 + H2;

    std::array<float, H1> h1{};
    std::array<float, H2> h2{};
    std::array<float, 256> probs{};

    std::vector<float> Wz1, Wr1, Wn1, Wz2, Wr2, Wn2, Wy;
    std::array<float, H1> bz1{}, br1{}, bn1{};
    std::array<float, H2> bz2{}, br2{}, bn2{};
    std::array<float, 256> by{};

    std::array<float, IN1> x1_cache{};
    std::array<float, IN2> x2_cache{};
    std::array<float, H1> z1_c{}, r1_c{}, n1_c{}, h1_prev{};
    std::array<float, H2> z2_c{}, r2_c{}, n2_c{}, h2_prev{};

    DeepOnlineByteRNN()
        : Wz1(H1 * IN1), Wr1(H1 * IN1), Wn1(H1 * IN1),
          Wz2(H2 * IN2), Wr2(H2 * IN2), Wn2(H2 * IN2),
          Wy(256 * H2) {
        std::mt19937 rng(1337);
        std::normal_distribution<float> dist(0.0f, 0.08f);
        for (float& v : Wz1) v = dist(rng);
        for (float& v : Wr1) v = dist(rng);
        for (float& v : Wn1) v = dist(rng);
        for (float& v : Wz2) v = dist(rng);
        for (float& v : Wr2) v = dist(rng);
        for (float& v : Wn2) v = dist(rng);
        for (float& v : Wy)  v = dist(rng);
        probs.fill(1.0f / 256.0f);
    }

    float predict_bit(int bit_idx, uint32_t partial) const {
        uint8_t prefix = (uint8_t)(partial << (bit_idx + 1));
        uint8_t prefix_mask = 0xFF << (bit_idx + 1);
        uint8_t bit_mask = 1 << bit_idx;

        float p0 = 1e-7f, p1 = 1e-7f;
        for (int b = 0; b < 256; ++b) {
            if ((b & prefix_mask) == prefix) {
                if (b & bit_mask) p1 += probs[b];
                else              p0 += probs[b];
            }
        }
        return p1 / (p0 + p1);
    }

    void step(uint8_t target_byte) {
        const float lr = 0.011f;

        std::array<float, H2> dh2{};
        for (int c = 0; c < 256; ++c) {
            float grad_y = probs[c] - (c == target_byte ? 1.0f : 0.0f);
            by[c] -= lr * grad_y;
            float* wy_row = &Wy[c * H2];
            for (int i = 0; i < H2; ++i) {
                dh2[i] += grad_y * wy_row[i];
                wy_row[i] -= lr * grad_y * h2[i];
            }
        }

        std::array<float, H1> dh1{};
        for (int i = 0; i < H2; ++i) {
            float dz = dh2[i] * (n2_c[i] - h2_prev[i]) * z2_c[i] * (1.0f - z2_c[i]);
            float dn = dh2[i] * z2_c[i] * (1.0f - n2_c[i] * n2_c[i]);
            bz2[i] -= lr * dz;
            bn2[i] -= lr * dn;

            float* wz_row = &Wz2[i * IN2];
            float* wn_row = &Wn2[i * IN2];
            for (int j = 0; j < H1; ++j) {
                dh1[j] += dz * wz_row[j] + dn * wn_row[j];
            }
            for (int j = 0; j < IN2; ++j) {
                float xj = x2_cache[j];
                wz_row[j] -= lr * dz * xj;
                wn_row[j] -= lr * dn * xj;
            }
        }

        for (int i = 0; i < H1; ++i) {
            float dz = dh1[i] * (n1_c[i] - h1_prev[i]) * z1_c[i] * (1.0f - z1_c[i]);
            float dn = dh1[i] * z1_c[i] * (1.0f - n1_c[i] * n1_c[i]);
            bz1[i] -= lr * dz;
            bn1[i] -= lr * dn;

            float* wz_row = &Wz1[i * IN1];
            float* wn_row = &Wn1[i * IN1];
            for (int j = 0; j < IN1; ++j) {
                float xj = x1_cache[j];
                if (xj != 0.0f) {
                    wz_row[j] -= lr * dz * xj;
                    wn_row[j] -= lr * dn * xj;
                }
            }
        }

        h1_prev = h1;
        x1_cache.fill(0.0f);
        x1_cache[target_byte] = 1.0f;
        for (int i = 0; i < H1; ++i) x1_cache[256 + i] = h1_prev[i];

        for (int i = 0; i < H1; ++i) {
            const float* wz = &Wz1[i * IN1];
            const float* wr = &Wr1[i * IN1];
            float sz = bz1[i] + wz[target_byte];
            float sr = br1[i] + wr[target_byte];
            for (int j = 0; j < H1; ++j) {
                sz += wz[256 + j] * h1_prev[j];
                sr += wr[256 + j] * h1_prev[j];
            }
            z1_c[i] = squash(sz);
            r1_c[i] = squash(sr);
        }
        for (int i = 0; i < H1; ++i) {
            const float* wn = &Wn1[i * IN1];
            float sn = bn1[i] + wn[target_byte];
            for (int j = 0; j < H1; ++j) sn += wn[256 + j] * (r1_c[j] * h1_prev[j]);
            n1_c[i] = std::tanh(sn);
            h1[i] = (1.0f - z1_c[i]) * h1_prev[i] + z1_c[i] * n1_c[i];
        }

        h2_prev = h2;
        for (int i = 0; i < H1; ++i) x2_cache[i] = h1[i];
        for (int i = 0; i < H2; ++i) x2_cache[H1 + i] = h2_prev[i];

        for (int i = 0; i < H2; ++i) {
            const float* wz = &Wz2[i * IN2];
            const float* wr = &Wr2[i * IN2];
            float sz = bz2[i], sr = br2[i];
            for (int j = 0; j < IN2; ++j) {
                sz += wz[j] * x2_cache[j];
                sr += wr[j] * x2_cache[j];
            }
            z2_c[i] = squash(sz);
            r2_c[i] = squash(sr);
        }
        for (int i = 0; i < H2; ++i) {
            const float* wn = &Wn2[i * IN2];
            float sn = bn2[i];
            for (int j = 0; j < H1; ++j) sn += wn[j] * h1[j];
            for (int j = 0; j < H2; ++j) sn += wn[H1 + j] * (r2_c[j] * h2_prev[j]);
            n2_c[i] = std::tanh(sn);
            h2[i] = (1.0f - z2_c[i]) * h2_prev[i] + z2_c[i] * n2_c[i];
        }

        float max_logit = -1e9f;
        std::array<float, 256> logits{};
        for (int c = 0; c < 256; ++c) {
            float s = by[c];
            const float* wy_row = &Wy[c * H2];
            for (int i = 0; i < H2; ++i) s += wy_row[i] * h2[i];
            logits[c] = s;
            if (s > max_logit) max_logit = s;
        }
        float sum_exp = 0.0f;
        for (int c = 0; c < 256; ++c) {
            probs[c] = std::exp(logits[c] - max_logit);
            sum_exp += probs[c];
        }
        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < 256; ++c) probs[c] *= inv_sum;
    }
};

struct ContextNeuralMixer {
    static constexpr int N = 15;
    std::array<std::array<float, N>, 1024> w{};
    float lr = 0.028f;

    ContextNeuralMixer() {
        for (int c = 0; c < 1024; ++c) w[c].fill(1.0f / N);
    }

    float mix(uint32_t ctx, const std::array<float, N>& s) const {
        float sum = 0.0f;
        const auto& wc = w[ctx & 1023];
        for (int i = 0; i < N; ++i) sum += wc[i] * s[i];
        return squash(sum);
    }

    void train(uint32_t ctx, const std::array<float, N>& s, float p, int bit) {
        float err = (float)bit - p;
        auto& wc = w[ctx & 1023];
        for (int i = 0; i < N; ++i) wc[i] += lr * err * s[i];
    }
};

struct SSE {
    static constexpr int BINS = 33;
    std::vector<std::array<float, BINS>> table;
    uint32_t mask;

    SSE(size_t ctx_bits = 15) : table(1ULL << ctx_bits), mask((1ULL << ctx_bits) - 1) {
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
        row[bin]     += 0.075f * (1.0f - frac) * (target - row[bin]);
        row[bin + 1] += 0.075f * frac * (target - row[bin + 1]);
    }
};

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

    ContextModel m1(16), m2(20), m3(22), m4(23), m5(23), m6(23);
    ContextModel mword1(22), mword2(22), msparse1(20), msparse2(21), mbracket(20);
    IndirectModel mindirect1(20, 22), mindirect2(20, 22);
    MultiMatchModel matcher;
    BracketTracker brackets;
    DeepOnlineByteRNN rnn;
    ContextNeuralMixer mixer;
    SSE sse1(15), sse2(15), sse3(14);

    uint32_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    uint32_t word_hash = 0, prev_word_hash = 0;

    for (uint64_t pos = 0; pos < file_size; ++pos) {
        uint8_t byte = encode ? (uint8_t)in.get() : 0;
        uint32_t partial = 1;

        uint32_t ind_ctx1 = (b2 << 8) | b1;
        uint32_t ind_ctx2 = (b3 * 1315423911u) ^ (b2 << 8) ^ b1;
        uint32_t brk_ctx  = brackets.context();

        for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
            uint32_t h1   = (b1 << 8) | partial;
            uint32_t h2   = ((b2 * 1315423911u) ^ (b1 << 8)) ^ partial;
            uint32_t h3   = ((b3 * 2654435761u) ^ (b2 << 8) ^ b1) ^ partial;
            uint32_t h4   = ((b4 * 97531u) ^ (b3 << 16) ^ (b2 << 8) ^ b1) ^ partial;
            uint32_t h5   = ((b5 * 31337u) ^ (b4 * 97531u) ^ (b3 << 16) ^ (b2 << 8) ^ b1) ^ partial;
            uint32_t h6   = ((b6 * 7919u) ^ (b5 * 31337u) ^ (b4 * 97531u) ^ (b3 << 16) ^ (b2 << 8) ^ b1) ^ partial;
            uint32_t hw1  = (word_hash * 31u) ^ partial;
            uint32_t hw2  = ((prev_word_hash * 1315423911u) ^ (word_hash * 31u)) ^ partial;
            uint32_t hs1  = ((b2 * 65599u) ^ partial);
            uint32_t hs2  = ((b3 * 131313u) ^ (b1 << 8) ^ partial);
            uint32_t hb   = (brk_ctx * 31u) ^ partial;

            std::array<float, 15> stretched = {
                stretch(m1.predict(h1)),
                stretch(m2.predict(h2)),
                stretch(m3.predict(h3)),
                stretch(m4.predict(h4)),
                stretch(m5.predict(h5)),
                stretch(m6.predict(h6)),
                stretch(mword1.predict(hw1)),
                stretch(mword2.predict(hw2)),
                stretch(msparse1.predict(hs1)),
                stretch(msparse2.predict(hs2)),
                stretch(mbracket.predict(hb)),
                stretch(mindirect1.predict(ind_ctx1, partial)),
                stretch(mindirect2.predict(ind_ctx2, partial)),
                stretch(matcher.predict_bit(bit_idx, partial)),
                stretch(rnn.predict_bit(bit_idx, partial))
            };

            uint32_t mix_ctx = (b1 << 2) | (matcher.match_len >= 8 ? 2 : (matcher.match_len >= 4 ? 1 : 0));
            float p_mix = mixer.mix(mix_ctx, stretched);

            int b1_bin = 0, b2_bin = 0, b3_bin = 0;
            float f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
            uint32_t sse_ctx1 = (b1 << 8) | partial;
            uint32_t sse_ctx2 = ((b2 & 0x7F) << 8) | partial;
            uint32_t sse_ctx3 = ((matcher.match_len & 0x1F) << 8) | partial;

            float p_s1 = sse1.map(sse_ctx1, p_mix, b1_bin, f1);
            float p_s2 = sse2.map(sse_ctx2, p_s1, b2_bin, f2);
            float p_s3 = sse3.map(sse_ctx3, p_s2, b3_bin, f3);
            float p = 0.45f * p_s1 + 0.25f * p_s2 + 0.15f * p_s3 + 0.15f * p_mix;

            uint32_t p1 = std::max(1u, std::min(4095u, (uint32_t)(p * 4096.0f)));

            int bit;
            if (encode) {
                bit = (byte >> bit_idx) & 1;
                coder.encode_bit(bit, p1);
            } else {
                bit = coder.decode_bit(p1);
                byte = (byte << 1) | bit;
            }

            m1.update(h1, bit); m2.update(h2, bit); m3.update(h3, bit);
            m4.update(h4, bit); m5.update(h5, bit); m6.update(h6, bit);
            mword1.update(hw1, bit); mword2.update(hw2, bit);
            msparse1.update(hs1, bit); msparse2.update(hs2, bit);
            mbracket.update(hb, bit);
            mindirect1.update_bit(ind_ctx1, partial, bit);
            mindirect2.update_bit(ind_ctx2, partial, bit);
            mixer.train(mix_ctx, stretched, p_mix, bit);
            sse1.update(sse_ctx1, b1_bin, f1, bit);
            sse2.update(sse_ctx2, b2_bin, f2, bit);
            sse3.update(sse_ctx3, b3_bin, f3, bit);

            partial = (partial << 1) | bit;
        }

        if (!encode) out.put((char)byte);

        mindirect1.update_byte(ind_ctx1, byte);
        mindirect2.update_byte(ind_ctx2, byte);
        matcher.update_byte(byte);
        brackets.update(byte);
        rnn.step(byte);

        b6 = b5; b5 = b4; b4 = b3; b3 = b2; b2 = b1; b1 = byte;
        if (std::isalpha(byte)) {
            word_hash = word_hash * 131u + std::tolower(byte);
        } else if (word_hash != 0) {
            prev_word_hash = word_hash;
            word_hash = 0;
        }
    }

    if (encode) coder.flush();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./mini_cmix_v4 [c|d] <input> <output>\n";
        return 1;
    }
    process(argv[1][0] == 'c', argv[2], argv[3]);
    return 0;
}
