#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <cmath>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <random>
#include <algorithm>

// --- 0. Fast Lookup Tables for Stretch & Squash ---
struct MathLUT {
    static constexpr int SQUASH_SIZE = 4096;
    static constexpr float SQUASH_MIN = -12.0f;
    static constexpr float SQUASH_MAX = 12.0f;
    std::array<float, SQUASH_SIZE + 1> squash_tbl{};
    std::array<float, 4097> stretch_tbl{};

    MathLUT() {
        for (int i = 0; i <= SQUASH_SIZE; ++i) {
            float x = SQUASH_MIN + (SQUASH_MAX - SQUASH_MIN) * ((float)i / SQUASH_SIZE);
            squash_tbl[i] = 1.0f / (1.0f + std::exp(-x));
        }
        for (int i = 0; i <= 4096; ++i) {
            float p = std::max(0.00001f, std::min(0.99999f, (float)i / 4096.0f));
            stretch_tbl[i] = std::log(p / (1.0f - p));
        }
    }

    inline float squash(float x) const {
        if (x <= SQUASH_MIN) return 0.000006f;
        if (x >= SQUASH_MAX) return 0.999994f;
        float pos = (x - SQUASH_MIN) * (SQUASH_SIZE / (SQUASH_MAX - SQUASH_MIN));
        int idx = (int)pos;
        float frac = pos - idx;
        return squash_tbl[idx] + frac * (squash_tbl[idx + 1] - squash_tbl[idx]);
    }

    inline float stretch(float p) const {
        p = std::max(0.00001f, std::min(0.99999f, p));
        float pos = p * 4096.0f;
        int idx = (int)pos;
        float frac = pos - idx;
        return stretch_tbl[idx] + frac * (stretch_tbl[idx + 1] - stretch_tbl[idx]);
    }
};

static const MathLUT LUT;

// --- 1. Bitwise Arithmetic Coder ---
class ArithmeticCoder {
    uint32_t low = 0, high = 0xFFFFFFFF, code = 0;
    uint8_t out_buf = 0, out_bits = 0;
    uint8_t in_buf = 0, in_bits = 0;
    std::ofstream* out = nullptr;
    std::ifstream* in = nullptr;

    inline void put_bit(int bit) {
        out_buf = (out_buf << 1) | (bit & 1);
        if (++out_bits == 8) {
            out->put((char)out_buf);
            out_buf = 0;
            out_bits = 0;
        }
    }

    inline uint8_t get_bit() {
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

    inline void encode_bit(int bit, uint32_t p1) {
        uint32_t mid = low + ((uint64_t)(high - low) * p1 >> 12);
        if (bit) high = mid;
        else     low = mid + 1;

        while (((low ^ high) & 0x80000000) == 0) {
            put_bit(low >> 31);
            low <<= 1;
            high = (high << 1) | 1;
        }
    }

    inline int decode_bit(uint32_t p1) {
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

// --- 2. Compact Pretrained English & Wikipedia XML Prior ---
// Provides warm-started unigram, bigram, and high-frequency English/MediaWiki n-gram priors from byte 0.
struct PretrainedEnglishPrior {
    std::array<std::array<float, 256>, 256> bigram_probs{};
    std::array<float, 256> cur_probs{};

    PretrainedEnglishPrior() {
        // Base English + XML character weights
        std::array<float, 256> base{};
        base.fill(0.15f);
        const char* freq_chars =
            "         eeeeettttaaaaooooiiiinnnnsssshhhhrrrrddddllllcccuuummmwwwfffgggyyypppbvvkkjxqz"
            "ETAOINSHRDLCUMWFGYPBVKJXQZ0123456789.,;:'\"-()[]{}<>/=&\n";
        for (const char* p = freq_chars; *p; ++p) {
            base[(uint8_t)*p] += 2.2f;
        }

        for (int prev = 0; prev < 256; ++prev) {
            for (int nxt = 0; nxt < 256; ++nxt) {
                bigram_probs[prev][nxt] = base[nxt];
            }
        }

        // Boost canonical English & Wikipedia XML bigrams / trigrams
        struct WeightedNGram { const char* seq; float boost; };
        static const WeightedNGram PRIORS[] = {
            {"th", 28.0f}, {"he", 26.0f}, {"in", 22.0f}, {"er", 21.0f}, {"an", 20.0f},
            {"re", 18.0f}, {"on", 17.0f}, {"at", 16.0f}, {"en", 16.0f}, {"nd", 15.0f},
            {"ti", 15.0f}, {"es", 15.0f}, {"or", 14.0f}, {"te", 14.0f}, {"of", 14.0f},
            {"ed", 14.0f}, {"is", 13.0f}, {"it", 13.0f}, {"al", 13.0f}, {"ar", 13.0f},
            {"st", 13.0f}, {"to", 13.0f}, {"nt", 12.0f}, {"ng", 12.0f}, {"se", 12.0f},
            {"ha", 12.0f}, {"as", 12.0f}, {"ou", 11.0f}, {"io", 11.0f}, {"le", 11.0f},
            {"ve", 10.0f}, {"co", 10.0f}, {"me", 10.0f}, {"de", 10.0f}, {"hi", 10.0f},
            {"ri", 10.0f}, {"ro", 10.0f}, {"ic", 10.0f}, {"ne", 10.0f}, {"ea", 10.0f},
            {"ra", 10.0f}, {"ce", 9.0f},  {"li", 9.0f},  {"ch", 9.0f},  {"ll", 9.0f},
            {"be", 9.0f},  {"ma", 9.0f},  {"si", 9.0f},  {"om", 9.0f},  {"ur", 9.0f},
            {"e ", 32.0f}, {"s ", 24.0f}, {"d ", 20.0f}, {"t ", 20.0f}, {"n ", 18.0f},
            {"y ", 14.0f}, {" th", 30.0f}, {" of", 22.0f}, {" in", 20.0f}, {" to", 18.0f},
            {" an", 18.0f}, {" a ", 16.0f}, {" is", 15.0f}, {" wa", 14.0f}, {" fo", 13.0f},
            {"[[", 26.0f}, {"]]", 26.0f}, {"{{", 20.0f}, {"}}", 20.0f}, {"==", 18.0f},
            {"''", 18.0f}, {"</", 24.0f}, {"<p", 18.0f}, {"pa", 15.0f}, {"ag", 15.0f},
            {"ge", 16.0f}, {"<t", 18.0f}, {"tl", 14.0f}, {"<r", 14.0f}, {"ev", 12.0f},
            {"vi", 12.0f}, {"si", 12.0f}, {"&a", 18.0f}, {"am", 16.0f}, {"mp", 16.0f},
            {"&q", 18.0f}, {"qu", 18.0f}, {"uo", 16.0f}, {"ot", 16.0f}, {"&l", 16.0f},
            {"&g", 16.0f}, {"gt", 14.0f}, {"lt", 14.0f}, {"Ca", 14.0f}, {"eg", 12.0f},
            {"go", 12.0f}, {"ry", 12.0f}, {"http", 16.0f}, {"www.", 14.0f}, {".org", 14.0f}
        };

        for (const auto& item : PRIORS) {
            size_t len = std::strlen(item.seq);
            for (size_t i = 0; i + 1 < len; ++i) {
                uint8_t a = (uint8_t)item.seq[i];
                uint8_t b = (uint8_t)item.seq[i + 1];
                bigram_probs[a][b] += item.boost;
                // Also propagate lowercase/uppercase letter pair
                if (std::isalpha(a)) {
                    bigram_probs[(uint8_t)std::toupper(a)][b] += item.boost * 0.65f;
                }
            }
        }

        // Normalize rows into valid probability distributions
        for (int prev = 0; prev < 256; ++prev) {
            float sum = 0.0f;
            for (int nxt = 0; nxt < 256; ++nxt) sum += bigram_probs[prev][nxt];
            float inv = 1.0f / sum;
            for (int nxt = 0; nxt < 256; ++nxt) bigram_probs[prev][nxt] *= inv;
        }
        cur_probs = bigram_probs[' '];
    }

    inline void prepare_byte(uint8_t b1, uint8_t b2) {
        // Blend static pretrained English/XML bigram with subtle structural punctuation boost
        const auto& row1 = bigram_probs[b1];
        const auto& row2 = bigram_probs[b2];
        for (int c = 0; c < 256; ++c) {
            cur_probs[c] = 0.85f * row1[c] + 0.15f * row2[c];
        }
        if (b2 == '[' && b1 == '[') cur_probs['C'] *= 1.8f; // [[Category:
        if (b2 == '<' && b1 == '/') {
            cur_probs['p'] *= 2.2f; // </page>
            cur_probs['t'] *= 2.2f; // </text>, </title>
            cur_probs['r'] *= 2.0f; // </revision>
        }
    }

    inline float predict_bit(int bit_idx, uint32_t partial) const {
        uint8_t prefix = (uint8_t)(partial << (bit_idx + 1));
        uint8_t prefix_mask = 0xFF << (bit_idx + 1);
        uint8_t bit_mask = 1 << bit_idx;

        float p0 = 1e-7f, p1 = 1e-7f;
        for (int b = 0; b < 256; ++b) {
            if ((b & prefix_mask) == prefix) {
                if (b & bit_mask) p1 += cur_probs[b];
                else              p0 += cur_probs[b];
            }
        }
        return p1 / (p0 + p1);
    }
};

// --- 3. PAQ Non-Stationary 256-State Finite-State Machine ---
struct StateTable {
    uint8_t next_state[256][2]{};
    float init_prob[256]{};

    StateTable() {
        auto encode = [](int n0, int n1, int last) -> uint8_t {
            n0 = std::min(10, std::max(0, n0));
            n1 = std::min(10, std::max(0, n1));
            return (uint8_t)(1 + n0 * 22 + n1 * 2 + (last & 1));
        };

        init_prob[0] = 0.5f;
        next_state[0][0] = encode(1, 0, 0);
        next_state[0][1] = encode(0, 1, 1);

        for (int n0 = 0; n0 <= 10; ++n0) {
            for (int n1 = 0; n1 <= 10; ++n1) {
                for (int last = 0; last <= 1; ++last) {
                    uint8_t s = encode(n0, n1, last);
                    float bias = (last == 1) ? 0.15f : -0.15f;
                    init_prob[s] = std::max(0.002f, std::min(0.998f,
                        (n1 + 0.5f + bias) / (float)(n0 + n1 + 1.0f)));

                    int nn0 = n0 + 1;
                    int nn1 = n1;
                    if (last == 1 && nn1 > 1) nn1 = (nn1 >> 1) + 1;
                    if (nn0 > 10) { nn0 = 10; nn1 = std::max(0, nn1 - 1); }
                    next_state[s][0] = encode(nn0, nn1, 0);

                    nn0 = n0;
                    nn1 = n1 + 1;
                    if (last == 0 && nn0 > 1) nn0 = (nn0 >> 1) + 1;
                    if (nn1 > 10) { nn1 = 10; nn0 = std::max(0, nn0 - 1); }
                    next_state[s][1] = encode(nn0, nn1, 1);
                }
            }
        }
    }
};

static const StateTable STATE_FSM;

struct NonStationaryContextModel {
    std::vector<uint8_t> states;
    std::array<float, 256> statemap{};
    uint32_t mask;
    float lr;

    NonStationaryContextModel(size_t bits, float adapt_rate = 0.015f)
        : states(1ULL << bits, 0), mask((1ULL << bits) - 1), lr(adapt_rate) {
        for (int s = 0; s < 256; ++s) statemap[s] = STATE_FSM.init_prob[s];
    }

    inline float predict(uint32_t h) const {
        return statemap[states[h & mask]];
    }

    inline void update(uint32_t h, int bit) {
        uint32_t idx = h & mask;
        uint8_t s = states[idx];
        statemap[s] += lr * ((float)bit - statemap[s]);
        states[idx] = STATE_FSM.next_state[s][bit];
    }
};

// --- 4. Order-6 Byte-Level PPM (Orders 0..6, 32 Slots, Method C + Symbol Exclusion) ---
struct PPMModel {
    static constexpr int SLOTS = 32;
    struct Bucket {
        uint8_t sym[SLOTS]{};
        uint8_t cnt[SLOTS]{};
        uint8_t used = 0;
        uint16_t total = 0;
    };

    std::vector<Bucket> t1, t2, t3, t4, t5, t6;
    uint32_t m1, m2, m3, m4, m5, m6;
    std::array<uint16_t, 256> order0{};
    uint32_t order0_total = 256;
    std::array<float, 256> probs{};

    PPMModel()
        : t1(1ULL << 15), t2(1ULL << 18), t3(1ULL << 19),
          t4(1ULL << 20), t5(1ULL << 20), t6(1ULL << 20),
          m1((1ULL << 15) - 1), m2((1ULL << 18) - 1), m3((1ULL << 19) - 1),
          m4((1ULL << 20) - 1), m5((1ULL << 20) - 1), m6((1ULL << 20) - 1) {
        order0.fill(1);
        probs.fill(1.0f / 256.0f);
    }

    static void update_bucket(Bucket& b, uint8_t symbol) {
        for (int i = 0; i < b.used; ++i) {
            if (b.sym[i] == symbol) {
                b.cnt[i]++;
                b.total++;
                while (i > 0 && b.cnt[i] > b.cnt[i - 1]) {
                    std::swap(b.cnt[i], b.cnt[i - 1]);
                    std::swap(b.sym[i], b.sym[i - 1]);
                    --i;
                }
                if (b.cnt[0] > 240) {
                    uint16_t new_tot = 0;
                    for (int k = 0; k < b.used; ++k) {
                        b.cnt[k] = std::max<uint8_t>(1, b.cnt[k] >> 1);
                        new_tot += b.cnt[k];
                    }
                    b.total = new_tot;
                }
                return;
            }
        }
        if (b.used < SLOTS) {
            b.sym[b.used] = symbol;
            b.cnt[b.used] = 1;
            b.used++;
            b.total++;
        } else {
            b.total = b.total - b.cnt[SLOTS - 1] + 1;
            b.sym[SLOTS - 1] = symbol;
            b.cnt[SLOTS - 1] = 1;
        }
    }

    void prepare_byte(uint32_t b1, uint32_t b2, uint32_t b3, uint32_t b4, uint32_t b5, uint32_t b6) {
        probs.fill(0.0f);
        std::array<uint8_t, 256> excluded{};
        float remaining_mass = 1.0f;

        uint32_t h1 = b1;
        uint32_t h2 = (b2 * 1315423911u) ^ b1;
        uint32_t h3 = (b3 * 2654435761u) ^ (b2 << 8) ^ b1;
        uint32_t h4 = (b4 * 97531u) ^ (b3 << 16) ^ (b2 << 8) ^ b1;
        uint32_t h5 = (b5 * 31337u) ^ h4;
        uint32_t h6 = (b6 * 7919u) ^ h5;

        const Bucket* chain[6] = {
            &t6[h6 & m6], &t5[h5 & m5], &t4[h4 & m4],
            &t3[h3 & m3], &t2[h2 & m2], &t1[h1 & m1]
        };

        for (int ord = 0; ord < 6; ++ord) {
            const Bucket& b = *chain[ord];
            if (b.used == 0) continue;

            uint32_t unexcluded_total = 0, distinct_unexcluded = 0;
            for (int i = 0; i < b.used; ++i) {
                if (!excluded[b.sym[i]]) {
                    unexcluded_total += b.cnt[i];
                    distinct_unexcluded++;
                }
            }
            if (unexcluded_total == 0) continue;

            float denom = (float)(unexcluded_total + distinct_unexcluded);
            float scale = remaining_mass / denom;

            for (int i = 0; i < b.used; ++i) {
                uint8_t s = b.sym[i];
                if (!excluded[s]) {
                    probs[s] += b.cnt[i] * scale;
                    excluded[s] = 1;
                }
            }
            remaining_mass *= ((float)distinct_unexcluded / denom);
            if (remaining_mass < 1e-5f) break;
        }

        if (remaining_mass > 1e-6f) {
            uint32_t rem0 = 0;
            for (int s = 0; s < 256; ++s) if (!excluded[s]) rem0 += order0[s];
            if (rem0 > 0) {
                float scale0 = remaining_mass / (float)rem0;
                for (int s = 0; s < 256; ++s) if (!excluded[s]) probs[s] += order0[s] * scale0;
            }
        }
    }

    inline float predict_bit(int bit_idx, uint32_t partial) const {
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

    void update_byte(uint32_t b1, uint32_t b2, uint32_t b3, uint32_t b4, uint32_t b5, uint32_t b6, uint8_t target) {
        uint32_t h1 = b1;
        uint32_t h2 = (b2 * 1315423911u) ^ b1;
        uint32_t h3 = (b3 * 2654435761u) ^ (b2 << 8) ^ b1;
        uint32_t h4 = (b4 * 97531u) ^ (b3 << 16) ^ (b2 << 8) ^ b1;
        uint32_t h5 = (b5 * 31337u) ^ h4;
        uint32_t h6 = (b6 * 7919u) ^ h5;

        update_bucket(t1[h1 & m1], target);
        update_bucket(t2[h2 & m2], target);
        update_bucket(t3[h3 & m3], target);
        update_bucket(t4[h4 & m4], target);
        update_bucket(t5[h5 & m5], target);
        update_bucket(t6[h6 & m6], target);

        order0[target]++;
        order0_total++;
        if (order0_total > 65536) {
            order0_total = 0;
            for (int s = 0; s < 256; ++s) {
                order0[s] = (order0[s] >> 1) + 1;
                order0_total += order0[s];
            }
        }
    }
};

// --- 5. Indirect Byte-History Model (Non-Stationary) ---
struct IndirectModel {
    std::vector<uint32_t> history;
    uint32_t hist_mask;
    NonStationaryContextModel cm;

    IndirectModel(size_t hist_bits = 20, size_t cm_bits = 23)
        : history(1ULL << hist_bits, 0), hist_mask((1ULL << hist_bits) - 1), cm(cm_bits) {}

    inline uint32_t get_hash(uint32_t ctx, uint32_t partial) const {
        return (history[ctx & hist_mask] * 2654435761u) ^ partial;
    }

    inline float predict(uint32_t ctx, uint32_t partial) const {
        return cm.predict(get_hash(ctx, partial));
    }

    inline void update_bit(uint32_t ctx, uint32_t partial, int bit) {
        cm.update(get_hash(ctx, partial), bit);
    }

    inline void update_byte(uint32_t prev_ctx, uint8_t byte) {
        uint32_t idx = prev_ctx & hist_mask;
        history[idx] = (history[idx] << 8) | byte;
    }
};

// --- 6. 8-Way Associative 16-Byte Multi-Match Chain Model ---
struct MultiMatchModel {
    static constexpr int WAYS = 8;
    std::vector<uint8_t> buf;
    std::vector<std::array<uint32_t, WAYS>> table16;
    std::vector<std::array<uint32_t, 4>> table8;
    std::vector<uint32_t> table4;
    uint32_t mask16, mask8, mask4;
    uint32_t match_pos = 0;
    uint32_t match_len = 0;

    MultiMatchModel()
        : table16(1ULL << 20), table8(1ULL << 20), table4(1ULL << 21, 0),
          mask16((1ULL << 20) - 1), mask8((1ULL << 20) - 1), mask4((1ULL << 21) - 1) {
        buf.reserve(1 << 24);
    }

    void update_byte(uint8_t byte) {
        uint32_t pos = (uint32_t)buf.size();
        buf.push_back(byte);
        if (pos < 16) return;

        if (match_len > 0) {
            if (buf[match_pos] == byte) {
                match_pos++;
                match_len++;
            } else {
                match_len = 0;
            }
        }

        uint32_t h4 = 0, h8 = 0, h16 = 0;
        for (int i = 0; i < 4; ++i)  h4  = h4  * 1315423911u + buf[pos - i];
        for (int i = 0; i < 8; ++i)  h8  = h8  * 2654435761u + buf[pos - i];
        for (int i = 0; i < 16; ++i) h16 = h16 * 16777619u   + buf[pos - i];

        auto& b16 = table16[h16 & mask16];
        auto& b8  = table8[h8 & mask8];
        uint32_t cand4 = table4[h4 & mask4];
        table4[h4 & mask4] = pos + 1;

        if (match_len < 16) {
            uint32_t best_pos = match_pos;
            uint32_t best_len = match_len;

            auto check_cand = [&](uint32_t cand) {
                if (cand >= 4 && cand <= pos) {
                    uint32_t len = 0;
                    while (len < 512 && cand > len && buf[cand - 1 - len] == buf[pos - len]) {
                        len++;
                    }
                    if (len > best_len) {
                        best_len = len;
                        best_pos = cand;
                    }
                }
            };

            for (int w = 0; w < WAYS; ++w) check_cand(b16[w]);
            if (best_len < 8) {
                for (int w = 0; w < 4; ++w) check_cand(b8[w]);
            }
            if (best_len < 4) check_cand(cand4);

            if (best_len >= 4) {
                match_pos = best_pos;
                match_len = best_len;
            }
        }

        for (int w = WAYS - 1; w > 0; --w) b16[w] = b16[w - 1];
        b16[0] = pos + 1;

        for (int w = 3; w > 0; --w) b8[w] = b8[w - 1];
        b8[0] = pos + 1;
    }

    inline float predict_bit(int bit_idx, uint32_t partial) const {
        if (match_len < 4 || match_pos >= buf.size()) return 0.5f;
        uint8_t predicted_byte = buf[match_pos];
        uint8_t prefix_mask = 0xFF << (bit_idx + 1);
        uint8_t decoded_prefix = (uint8_t)(partial << (bit_idx + 1));
        if ((predicted_byte & prefix_mask) != decoded_prefix) return 0.5f;

        int pred_bit = (predicted_byte >> bit_idx) & 1;
        float conf = std::min(0.9990f, 0.78f + 0.028f * std::min(match_len, 8u));
        return pred_bit ? conf : (1.0f - conf);
    }
};

// --- 7. Structural Tracker (Brackets + Line Column + English Word Stemmer + Trigram) ---
struct StructuralTracker {
    std::vector<uint8_t> stack;
    uint32_t dist = 0;
    uint32_t col = 0;

    std::string cur_word;
    uint32_t word0 = 0, word1 = 0, word2 = 0;
    uint32_t stem0 = 0, stem1 = 0;

    static uint32_t hash_str(const std::string& s) {
        uint32_t h = 2166136261u;
        for (char c : s) h = (h ^ (uint8_t)c) * 16777619u;
        return h;
    }

    static std::string stem_english(std::string w) {
        size_t n = w.size();
        if (n <= 4) return w;
        if (n > 6 && w.substr(n - 4) == "tion") return w.substr(0, n - 3) + "e";
        if (n > 6 && w.substr(n - 4) == "ment") return w.substr(0, n - 4);
        if (n > 5 && w.substr(n - 3) == "ing")  return w.substr(0, n - 3);
        if (n > 4 && w.substr(n - 2) == "ed")   return w.substr(0, n - 2);
        if (n > 4 && w.substr(n - 2) == "ly")   return w.substr(0, n - 2);
        if (n > 4 && w.substr(n - 2) == "es")   return w.substr(0, n - 2);
        if (n > 3 && w.back() == 's' && w[n - 2] != 's') return w.substr(0, n - 1);
        return w;
    }

    inline void update(uint8_t b) {
        dist = std::min(dist + 1, 255u);
        if (b == '\n') col = 0;
        else           col = std::min(col + 1, 255u);

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

        if (std::isalpha(b)) {
            char lc = (char)std::tolower(b);
            cur_word.push_back(lc);
            word0 = word0 * 131u + (uint8_t)lc;
        } else if (!cur_word.empty()) {
            word2 = word1;
            word1 = word0;
            stem1 = stem0;
            stem0 = hash_str(stem_english(cur_word));
            cur_word.clear();
            word0 = 0;
        }
    }

    inline uint32_t bracket_context() const {
        uint8_t top = stack.empty() ? 0 : stack.back();
        return ((uint32_t)top << 12) | ((uint32_t)(stack.size() & 0xF) << 8) | (dist >> 2);
    }
};

// --- 8. Optimized 2-Layer Online Byte-Level GRU ---
struct DeepOnlineByteRNN {
    static constexpr int H1 = 64;
    static constexpr int H2 = 64;
    static constexpr int IN1 = 256 + H1;
    static constexpr int IN2 = H1 + H2;

    alignas(32) std::array<float, H1> h1{};
    alignas(32) std::array<float, H2> h2{};
    alignas(32) std::array<float, 256> probs{};

    std::vector<float> Wz1, Wr1, Wn1, Wz2, Wr2, Wn2, Wy;
    alignas(32) std::array<float, H1> bz1{}, br1{}, bn1{};
    alignas(32) std::array<float, H2> bz2{}, br2{}, bn2{};
    alignas(32) std::array<float, 256> by{};

    uint8_t prev_byte_cache = 0;
    alignas(32) std::array<float, IN2> x2_cache{};
    alignas(32) std::array<float, H1> z1_c{}, r1_c{}, n1_c{}, h1_prev{};
    alignas(32) std::array<float, H2> z2_c{}, r2_c{}, n2_c{}, h2_prev{};

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

    inline float predict_bit(int bit_idx, uint32_t partial) const {
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

        alignas(32) std::array<float, H2> dh2{};
        float* __restrict__ dh2_p = dh2.data();
        const float* __restrict__ h2_p = h2.data();

        for (int c = 0; c < 256; ++c) {
            float grad_y = probs[c] - (c == target_byte ? 1.0f : 0.0f);
            by[c] -= lr * grad_y;
            float lr_g = lr * grad_y;
            float* __restrict__ wy_row = &Wy[c * H2];
            for (int i = 0; i < H2; ++i) {
                dh2_p[i] += grad_y * wy_row[i];
                wy_row[i] -= lr_g * h2_p[i];
            }
        }

        alignas(32) std::array<float, H1> dh1{};
        float* __restrict__ dh1_p = dh1.data();
        const float* __restrict__ x2_p = x2_cache.data();

        for (int i = 0; i < H2; ++i) {
            float dz = dh2_p[i] * (n2_c[i] - h2_prev[i]) * z2_c[i] * (1.0f - z2_c[i]);
            float dn = dh2_p[i] * z2_c[i] * (1.0f - n2_c[i] * n2_c[i]);
            bz2[i] -= lr * dz;
            bn2[i] -= lr * dn;

            float lr_dz = lr * dz;
            float lr_dn = lr * dn;
            float* __restrict__ wz_row = &Wz2[i * IN2];
            float* __restrict__ wn_row = &Wn2[i * IN2];

            for (int j = 0; j < H1; ++j) dh1_p[j] += dz * wz_row[j] + dn * wn_row[j];
            for (int j = 0; j < IN2; ++j) {
                wz_row[j] -= lr_dz * x2_p[j];
                wn_row[j] -= lr_dn * x2_p[j];
            }
        }

        const float* __restrict__ h1_prev_p = h1_prev.data();
        for (int i = 0; i < H1; ++i) {
            float dz = dh1_p[i] * (n1_c[i] - h1_prev_p[i]) * z1_c[i] * (1.0f - z1_c[i]);
            float dn = dh1_p[i] * z1_c[i] * (1.0f - n1_c[i] * n1_c[i]);
            bz1[i] -= lr * dz;
            bn1[i] -= lr * dn;

            float lr_dz = lr * dz;
            float lr_dn = lr * dn;
            float* __restrict__ wz_row = &Wz1[i * IN1];
            float* __restrict__ wn_row = &Wn1[i * IN1];

            wz_row[prev_byte_cache] -= lr_dz;
            wn_row[prev_byte_cache] -= lr_dn;

            float* __restrict__ wz_h = wz_row + 256;
            float* __restrict__ wn_h = wn_row + 256;
            for (int j = 0; j < H1; ++j) {
                wz_h[j] -= lr_dz * h1_prev_p[j];
                wn_h[j] -= lr_dn * h1_prev_p[j];
            }
        }

        h1_prev = h1;
        prev_byte_cache = target_byte;

        for (int i = 0; i < H1; ++i) {
            const float* __restrict__ wz_h = &Wz1[i * IN1 + 256];
            const float* __restrict__ wr_h = &Wr1[i * IN1 + 256];
            float sz = bz1[i] + Wz1[i * IN1 + target_byte];
            float sr = br1[i] + Wr1[i * IN1 + target_byte];
            for (int j = 0; j < H1; ++j) {
                sz += wz_h[j] * h1_prev_p[j];
                sr += wr_h[j] * h1_prev_p[j];
            }
            z1_c[i] = LUT.squash(sz);
            r1_c[i] = LUT.squash(sr);
        }
        for (int i = 0; i < H1; ++i) {
            const float* __restrict__ wn_h = &Wn1[i * IN1 + 256];
            float sn = bn1[i] + Wn1[i * IN1 + target_byte];
            for (int j = 0; j < H1; ++j) sn += wn_h[j] * (r1_c[j] * h1_prev_p[j]);
            n1_c[i] = std::tanh(sn);
            h1[i] = (1.0f - z1_c[i]) * h1_prev_p[i] + z1_c[i] * n1_c[i];
        }

        h2_prev = h2;
        std::memcpy(x2_cache.data(), h1.data(), H1 * sizeof(float));
        std::memcpy(x2_cache.data() + H1, h2_prev.data(), H2 * sizeof(float));

        for (int i = 0; i < H2; ++i) {
            const float* __restrict__ wz = &Wz2[i * IN2];
            const float* __restrict__ wr = &Wr2[i * IN2];
            float sz = bz2[i], sr = br2[i];
            for (int j = 0; j < IN2; ++j) {
                sz += wz[j] * x2_p[j];
                sr += wr[j] * x2_p[j];
            }
            z2_c[i] = LUT.squash(sz);
            r2_c[i] = LUT.squash(sr);
        }
        for (int i = 0; i < H2; ++i) {
            const float* __restrict__ wn = &Wn2[i * IN2];
            float sn = bn2[i];
            for (int j = 0; j < H1; ++j) sn += wn[j] * h1[j];
            for (int j = 0; j < H2; ++j) sn += wn[H1 + j] * (r2_c[j] * h2_prev[j]);
            n2_c[i] = std::tanh(sn);
            h2[i] = (1.0f - z2_c[i]) * h2_prev[i] + z2_c[i] * n2_c[i];
        }

        float max_logit = -1e9f;
        alignas(32) std::array<float, 256> logits{};
        for (int c = 0; c < 256; ++c) {
            float s = by[c];
            const float* __restrict__ wy_row = &Wy[c * H2];
            for (int i = 0; i < H2; ++i) s += wy_row[i] * h2_p[i];
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

// --- 9. Context-Selected Neural Mixer (22 inputs) ---
template <size_t NUM_CTX>
struct ContextNeuralMixer {
    static constexpr int N = 22;
    alignas(32) std::vector<std::array<float, N>> w;
    float lr;

    ContextNeuralMixer(float learning_rate = 0.026f) : w(NUM_CTX), lr(learning_rate) {
        for (size_t c = 0; c < NUM_CTX; ++c) w[c].fill(1.0f / N);
    }

    inline float mix(uint32_t ctx, const std::array<float, N>& s) const {
        float sum = 0.0f;
        const float* __restrict__ wc = w[ctx & (NUM_CTX - 1)].data();
        const float* __restrict__ sp = s.data();
        for (int i = 0; i < N; ++i) sum += wc[i] * sp[i];
        return LUT.squash(sum);
    }

    inline void train(uint32_t ctx, const std::array<float, N>& s, float p, int bit) {
        float lr_err = lr * ((float)bit - p);
        float* __restrict__ wc = w[ctx & (NUM_CTX - 1)].data();
        const float* __restrict__ sp = s.data();
        for (int i = 0; i < N; ++i) wc[i] += lr_err * sp[i];
    }
};

// --- 10. PAQ Adaptive Probability Map (APM) with Visit-Count Learning Rate Decay ---
struct AdaptiveProbabilityMap {
    static constexpr int BINS = 65;
    struct Cell {
        float p;
        uint16_t visits;
    };
    std::vector<std::array<Cell, BINS>> table;
    uint32_t mask;

    AdaptiveProbabilityMap(size_t ctx_bits = 15)
        : table(1ULL << ctx_bits), mask((1ULL << ctx_bits) - 1) {
        for (auto& row : table) {
            for (int b = 0; b < BINS; ++b) {
                float x = ((float)b / (BINS - 1) - 0.5f) * 18.0f;
                row[b].p = LUT.squash(x);
                row[b].visits = 0;
            }
        }
    }

    inline float map(uint32_t ctx, float p, int& bin_out, float& frac_out) const {
        float x = (LUT.stretch(p) / 18.0f + 0.5f) * (BINS - 1);
        x = std::max(0.0f, std::min((float)(BINS - 1.0001f), x));
        bin_out = (int)x;
        frac_out = x - bin_out;
        const auto& row = table[ctx & mask];
        return row[bin_out].p + frac_out * (row[bin_out + 1].p - row[bin_out].p);
    }

    inline void update(uint32_t ctx, int bin, float frac, int bit) {
        auto& row = table[ctx & mask];
        float target = (float)bit;

        auto upd_cell = [&](Cell& c, float weight) {
            float lr = weight / (2.0f + 0.14f * (float)c.visits);
            c.p += lr * (target - c.p);
            if (c.visits < 2000) c.visits++;
        };

        upd_cell(row[bin], 1.0f - frac);
        upd_cell(row[bin + 1], frac);
    }
};

// --- 11. Main Predictor Pipeline ---
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

    PretrainedEnglishPrior en_prior;
    NonStationaryContextModel m1(16), m2(20), m3(22), m4(23), m5(24), m6(24), m7(24), m8(24);
    NonStationaryContextModel mword1(23), mword2(23), mword3(23), mword_gap(22), mstem(22);
    NonStationaryContextModel msparse1(21), msparse2(22), mstruct(22);
    IndirectModel mindirect1(20, 23), mindirect2(20, 23);
    MultiMatchModel matcher;
    StructuralTracker struct_trk;
    PPMModel ppm;
    DeepOnlineByteRNN rnn;

    ContextNeuralMixer<2048> mixer1(0.025f);
    ContextNeuralMixer<1024> mixer2(0.021f);
    AdaptiveProbabilityMap apm1(16), apm2(16), apm3(15);

    uint32_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, b7 = 0, b8 = 0;

    for (uint64_t pos = 0; pos < file_size; ++pos) {
        uint8_t byte = encode ? (uint8_t)in.get() : 0;
        uint32_t partial = 1;

        en_prior.prepare_byte((uint8_t)b1, (uint8_t)b2);
        ppm.prepare_byte(b1, b2, b3, b4, b5, b6);

        uint32_t ind_ctx1 = (b2 << 8) | b1;
        uint32_t ind_ctx2 = (b3 * 1315423911u) ^ (b2 << 8) ^ b1;
        uint32_t brk_ctx  = struct_trk.bracket_context();
        uint32_t col_ctx  = struct_trk.col;

        uint32_t base2 = (b2 * 1315423911u) ^ (b1 << 8);
        uint32_t base3 = (b3 * 2654435761u) ^ (b2 << 8) ^ b1;
        uint32_t base4 = (b4 * 97531u) ^ (b3 << 16) ^ (b2 << 8) ^ b1;
        uint32_t base5 = (b5 * 31337u) ^ base4;
        uint32_t base6 = (b6 * 7919u) ^ base5;
        uint32_t base7 = (b7 * 104729u) ^ base6;
        uint32_t base8 = (b8 * 224737u) ^ base7;

        for (int bit_idx = 7; bit_idx >= 0; --bit_idx) {
            uint32_t h1   = (b1 << 8) | partial;
            uint32_t h2   = base2 ^ partial;
            uint32_t h3   = base3 ^ partial;
            uint32_t h4   = base4 ^ partial;
            uint32_t h5   = base5 ^ partial;
            uint32_t h6   = base6 ^ partial;
            uint32_t h7   = base7 ^ partial;
            uint32_t h8   = base8 ^ partial;

            uint32_t hw1  = (struct_trk.word0 * 31u) ^ partial;
            uint32_t hw2  = ((struct_trk.word1 * 1315423911u) ^ (struct_trk.word0 * 31u)) ^ partial;
            uint32_t hw3  = ((struct_trk.word2 * 2654435761u) ^ (struct_trk.word1 * 1315423911u) ^ (struct_trk.word0 * 31u)) ^ partial;
            uint32_t hwg  = ((struct_trk.word2 * 2654435761u) ^ (struct_trk.word0 * 31u)) ^ partial;
            uint32_t hstm = ((struct_trk.stem1 * 97531u) ^ (struct_trk.word0 * 31u)) ^ partial;

            uint32_t hs1  = ((b2 * 65599u) ^ partial);
            uint32_t hs2  = ((b3 * 131313u) ^ (b1 << 8) ^ partial);
            uint32_t hst  = ((brk_ctx * 31u) ^ (col_ctx << 8) ^ partial);

            alignas(32) std::array<float, 22> stretched = {
                LUT.stretch(en_prior.predict_bit(bit_idx, partial)),
                LUT.stretch(m1.predict(h1)),
                LUT.stretch(m2.predict(h2)),
                LUT.stretch(m3.predict(h3)),
                LUT.stretch(m4.predict(h4)),
                LUT.stretch(m5.predict(h5)),
                LUT.stretch(m6.predict(h6)),
                LUT.stretch(m7.predict(h7)),
                LUT.stretch(m8.predict(h8)),
                LUT.stretch(mword1.predict(hw1)),
                LUT.stretch(mword2.predict(hw2)),
                LUT.stretch(mword3.predict(hw3)),
                LUT.stretch(mword_gap.predict(hwg)),
                LUT.stretch(mstem.predict(hstm)),
                LUT.stretch(msparse1.predict(hs1)),
                LUT.stretch(msparse2.predict(hs2)),
                LUT.stretch(mstruct.predict(hst)),
                LUT.stretch(mindirect1.predict(ind_ctx1, partial)),
                LUT.stretch(mindirect2.predict(ind_ctx2, partial)),
                LUT.stretch(matcher.predict_bit(bit_idx, partial)),
                LUT.stretch(ppm.predict_bit(bit_idx, partial)),
                LUT.stretch(rnn.predict_bit(bit_idx, partial))
            };

            uint32_t mix_ctx1 = (b1 << 3) | (matcher.match_len >= 16 ? 4 : (matcher.match_len >= 8 ? 2 : (matcher.match_len >= 4 ? 1 : 0)));
            uint32_t mix_ctx2 = ((partial & 0x7F) << 3) | ((brk_ctx >> 12) & 0x7);

            float p_m1 = mixer1.mix(mix_ctx1, stretched);
            float p_m2 = mixer2.mix(mix_ctx2, stretched);
            float p_mix = LUT.squash(0.58f * LUT.stretch(p_m1) + 0.42f * LUT.stretch(p_m2));

            int b1_bin = 0, b2_bin = 0, b3_bin = 0;
            float f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
            uint32_t apm_ctx1 = (b1 << 8) | partial;
            uint32_t apm_ctx2 = ((b2 & 0x3F) << 10) | (((brk_ctx >> 12) & 0x3) << 8) | partial;
            uint32_t apm_ctx3 = ((matcher.match_len & 0x1F) << 10) | ((col_ctx >> 6) << 8) | partial;

            float p_a1 = apm1.map(apm_ctx1, p_mix, b1_bin, f1);
            float p_a2 = apm2.map(apm_ctx2, p_a1,  b2_bin, f2);
            float p_a3 = apm3.map(apm_ctx3, p_a2,  b3_bin, f3);
            float p = 0.42f * p_a1 + 0.28f * p_a2 + 0.18f * p_a3 + 0.12f * p_mix;

            uint32_t p1 = std::max(1u, std::min(4095u, (uint32_t)(p * 4096.0f)));

            int bit;
            if (encode) {
                bit = (byte >> bit_idx) & 1;
                coder.encode_bit(bit, p1);
            } else {
                bit = coder.decode_bit(p1);
                byte = (byte << 1) | bit;
            }

            m1.update(h1, bit); m2.update(h2, bit); m3.update(h3, bit); m4.update(h4, bit);
            m5.update(h5, bit); m6.update(h6, bit); m7.update(h7, bit); m8.update(h8, bit);
            mword1.update(hw1, bit); mword2.update(hw2, bit); mword3.update(hw3, bit);
            mword_gap.update(hwg, bit); mstem.update(hstm, bit);
            msparse1.update(hs1, bit); msparse2.update(hs2, bit);
            mstruct.update(hst, bit);
            mindirect1.update_bit(ind_ctx1, partial, bit);
            mindirect2.update_bit(ind_ctx2, partial, bit);

            mixer1.train(mix_ctx1, stretched, p_m1, bit);
            mixer2.train(mix_ctx2, stretched, p_m2, bit);
            apm1.update(apm_ctx1, b1_bin, f1, bit);
            apm2.update(apm_ctx2, b2_bin, f2, bit);
            apm3.update(apm_ctx3, b3_bin, f3, bit);

            partial = (partial << 1) | bit;
        }

        if (!encode) out.put((char)byte);

        ppm.update_byte(b1, b2, b3, b4, b5, b6, byte);
        mindirect1.update_byte(ind_ctx1, byte);
        mindirect2.update_byte(ind_ctx2, byte);
        matcher.update_byte(byte);
        struct_trk.update(byte);
        rnn.step(byte);

        b8 = b7; b7 = b6; b6 = b5; b5 = b4; b4 = b3; b3 = b2; b2 = b1; b1 = byte;
    }

    if (encode) coder.flush();
}

int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: ./mini_cmix_v7 [c|d] <input> <output>\n";
        return 1;
    }
    process(argv[1][0] == 'c', argv[2], argv[3]);
    return 0;
}
