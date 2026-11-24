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
        if (c0[idx] + c1[idx] > 512) {
            c0[idx] = (c0[idx] >> 1) + 1;
            c1[idx] = (c1[idx] >> 1) + 1;
        }
    }
};

struct NeuralMixer {
    static constexpr int N = 6;
    std::array<float, N> w{};
    float lr = 0.04f;

    NeuralMixer() { w.fill(1.0f / N); }

    float mix(const std::array<float, N>& stretched) const {
        float sum = 0.0f;
        for (int i = 0; i < N; ++i) sum += w[i] * stretched[i];
        return squash(sum);
    }

    void train(const std::array<float, N>& stretched, float p, int bit) {
        float err = (float)bit - p;
        for (int i = 0; i < N; ++i) w[i] += lr * err * stretched[i];
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

    ContextModel m0(12), m1(16), m2(20), m3(22), m4(23), mword(22);
    NeuralMixer mixer;
    uint32_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, word_hash = 0;

    for (uint64_t pos = 0; pos < file_size; ++pos) {
        uint8_t byte = encode ? (uint8_t)in.get() : 0;
        uint32_t partial = 1;

        for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
            uint32_t h0 = partial;
            uint32_t h1 = (b1 << 8) | partial;
            uint32_t h2 = ((b2 * 1315423911u) ^ (b1 << 8)) ^ partial;
            uint32_t h3 = ((b3 * 2654435761u) ^ (b2 << 8) ^ b1) ^ partial;
            uint32_t h4 = ((b4 * 97531u) ^ (b3 << 16) ^ (b2 << 8) ^ b1) ^ partial;
            uint32_t hw = (word_hash * 31u) ^ partial;

            std::array<uint32_t, 6> hashes = {h0, h1, h2, h3, h4, hw};
            std::array<float, 6> stretched = {
                stretch(m0.predict(h0)), stretch(m1.predict(h1)),
                stretch(m2.predict(h2)), stretch(m3.predict(h3)),
                stretch(m4.predict(h4)), stretch(mword.predict(hw))
            };

            float p = mixer.mix(stretched);
            uint32_t p1 = std::max(1u, std::min(4095u, (uint32_t)(p * 4096.0f)));

            int bit;
            if (encode) {
                bit = (byte >> bit_idx) & 1;
                coder.encode_bit(bit, p1);
            } else {
                bit = coder.decode_bit(p1);
                byte = (byte << 1) | bit;
            }

            m0.update(hashes[0], bit); m1.update(hashes[1], bit);
            m2.update(hashes[2], bit); m3.update(hashes[3], bit);
            m4.update(hashes[4], bit); mword.update(hashes[5], bit);
            mixer.train(stretched, p, bit);

            partial = (partial << 1) | bit;
        }

        if (!encode) out.put((char)byte);

        b4 = b3; b3 = b2; b2 = b1; b1 = byte;
        if (std::isalpha(byte)) word_hash = word_hash * 131u + std::tolower(byte);
        else word_hash = 0;
    }

    if (encode) coder.flush();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./mini_cmix [c|d] <input> <output>\n";
        return 1;
    }
    process(argv[1][0] == "c"[0], argv[2], argv[3]);
    return 0;
}
