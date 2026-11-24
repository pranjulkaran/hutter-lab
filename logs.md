# `mini_cmix` Engineering Log Report (`v1` – `v5` on `enwik6`)

**Environment:** Linux (`ROGSTRIX`), single-core C++17 (`g++ -O3 -march=native -ffast-math`), zero external dependencies.

  

**Benchmark Dataset:** `enwik6` (`1,000,000` bytes — first 1 MB of English Wikipedia XML).

  

**Verification Standard:** 100% bit-exact lossless round-trip parity (`sha256sum` match on every build).

  

**Target Reference:** Official Hutter Prize `enwik9` record stands at **`0.8034 bpb`** (`100,424,672` total bytes by `fx2-cmix-T`, July 2026), requiring `< 0.7954 bpb` (`< 99,420,425` bytes) for the next 1% prize threshold.

  

## 1. Benchmark Results Summary (`enwik6` — 1,000,000 Bytes)

|**Version / Baseline**|**Compressed Size**|**Compression Ratio**|**Bits/Byte (bpb)**|**Runtime (Encode)**|**UPX Binary Size**|**Delta vs. Previous**|
|---|---|---|---|---|---|---|
|**Uncompressed (`enwik6`)**|`1,000,000 B`|`100.00%`|`8.000 bpb`|—|—|—|
|**`gzip -9`** _(DEFLATE baseline)_|`355,357 B`|`35.54%`|`2.843 bpb`|`~0.08 s`|—|—|
|**`mini_cmix v1`**|`299,381 B`|`29.94%`|`2.395 bpb`|`0.58 s`|—|`-55,976 B` vs. `gzip`|
|**`xz -9`** _(LZMA2 baseline)_|`290,700 B`|`29.07%`|`2.326 bpb`|`~0.45 s`|—|—|
|**`mini_cmix v2`**|`261,466 B`|`26.15%`|`2.092 bpb`|`0.99 s`|—|`-37,915 B` (`-12.7%`)|
|**`mini_cmix v3`**|`251,351 B`|`25.14%`|`2.011 bpb`|`6.78 s`|—|`-10,115 B` (`-3.9%`)|
|**`wiki_prep + v4`** _(escape-token test)_|`250,433 B`|`25.04%`|`2.003 bpb`|`9.48 s`|—|`-918 B` _(hurt by `0xFE`)_|
|**`mini_cmix v4`** _(raw stream)_|`245,520 B`|`24.55%`|`1.964 bpb`|`11.04 s`|`22,000 B`|`-5,831 B` (`-2.3%`)|
|**`mini_cmix v5`** _(+PPM + SIMD/LUT)_|**`243,764 B`**|**`24.38%`**|**`1.950 bpb`**|**`9.78 s`**|**`26,512 B`**|**`-1,756 B` (`-0.7%`)**|

**Net Gain (`v1` $\rightarrow$ `v5`):** Reduced archive size by **`55,617` bytes (`-18.6%`)** over `v1`, **`46,936` bytes (`-16.1%`)** better than `xz -9`, and **`111,593` bytes (`-31.4%`)** better than `gzip -9`.

## 2. Version-by-Version Architecture & Logic Changelog

### `mini_cmix v1` — Core Context-Mixing Engine (`299,381 B` | `2.395 bpb`)

- **Core Logic:** Established the foundational bitwise predictive coding pipeline:
    
    - **32-bit Bitwise Arithmetic Coder:** Renormalizes `low` and `high` intervals on matching MSB bits using 12-bit fixed-point probabilities (`1..4095`).
        
    - **Order-1 through Order-4 Byte Context Models:** Hash tables mapping preceding bytes + decoded partial byte bits (`1..255`) to adaptive `(c0, c1)` bit counters with halving decay when total counts exceed threshold.
        
    - **Logistic Neural Mixer (`stretch` / `squash`):** Transforms model probabilities into log-odds space ($x = \ln\frac{p}{1-p}$), computes a weighted linear combination, squashes back via sigmoid $\sigma(x)$, and updates weights online via gradient descent on bit prediction error `(bit - p)`.
        
- **Takeaway:** Immediately beat `gzip -9` by ~56 KB in `< 0.6s`, validating the bitwise context-mixing loop.
    

### `mini_cmix v2` — Deep Contexts, Word Hashing, Match Model & SSE (`261,466 B` | `2.092 bpb`)

- **New Models & Logic Added:**
    
    - **Extended Order-5 & Order-6 Context Models (`2^23` entries):** Captures longer multi-character substrings and XML tag suffixes.
        
    - **Case-Insensitive Word Hashers (`word_hash`, `prev_word_hash`):** Accumulates rolling hashes over `std::isalpha` runs to track unigram and bigram word boundaries across spaces/punctuation.
        
    - **Sparse / Gap Context Models:** Models non-adjacent byte relationships (e.g., skipping the immediately preceding byte to capture structural templates).
        
    - **Single-Chain LZ Match Model:** Scans backward in a rolling history buffer for exact 4-byte/8-byte hash matches, predicting the next byte's bits with confidence scaled by match length.
        
    - **Context-Selected Mixer + Secondary Symbol Estimation (SSE):** Replaced the single global mixer with a context-indexed mixer table (`1024` contexts) and added post-mixer SSE interpolation tables (`33` probability bins) to calibrate systematic probability bias.
        
- **Takeaway:** Largest single jump in the project (`-37,915 B`), decisively beating `xz -9` (`290,700 B`) by nearly 30 KB while staying under 1 second.
    

### `mini_cmix v3` — Online Byte-Level GRU Neural Network (`251,351 B` | `2.011 bpb`)

- **New Models & Logic Added:**
    
    - **2-Layer Online Byte-Level GRU (`64 -> 64 -> 256`):** Added a recurrent neural predictor trained online via truncated Backpropagation Through Time (BPTT) at every byte boundary, maintaining a 256-way softmax distribution over the next character and marginalizing over the decoded bit prefix for bitwise probability estimates.
        
- **Takeaway:** Cut another `10,115 B` by capturing smooth semantic and character-class transitions that discrete hash tables miss, at the cost of raising encode time from `0.99s` to `6.78s`.
    

### `mini_cmix v4` — 4-Way Multi-Match, Indirect Contexts & Bracket Tracker (`245,520 B` | `1.964 bpb`)

- **New Models & Logic Added:**
    
    - **4-Way Associative Multi-Match Model:** Replaced the single-slot match table with a 4-way set-associative 8-byte hash bucket (`2^21 x 4`) plus a 4-byte fallback table, testing multiple candidate history offsets and locking onto the longest match (up to 256 bytes).
        
    - **Order-1 & Order-2 Indirect History Models:** Two-stage lookup mapping a context hash to a 32-bit rolling history of the last 4 bytes seen in that exact context, then hashing that sequence into a `2^22` bit-counter table.
        
    - **Bracket & XML Delimiter Nesting Tracker:** Tracks an active stack of open delimiters (`[`, `{`, `(`, `<`) and distance since the last delimiter transition, giving the mixer explicit structural awareness of Wiki links (`[[...]]`), templates (`{{...}}`), and XML tags (`<...>`).
        
    - **3-Stage Cascaded SSE:** Upgraded post-mixer calibration to three cascaded SSE tables conditioned on `b1`, `b2`, and `match_len`.
        
- **Side Experiment (`wiki_prep + v4` $\rightarrow$ `250,433 B`):** Tested a naive `0xFE <token_id>` dictionary preprocessor. It performed **4,913 B worse** than raw `v4` because `0xFE` control bytes broke `std::isalpha` word-boundary tracking and disrupted long-range LZ matches—proving preprocessors must preserve word/structural semantics rather than injecting escape prefixes.
    

### `mini_cmix v5` — Byte-Level PPM (Method C + Exclusion) & Speed Optimizations (`243,764 B` | `1.950 bpb`)

- **New Models & Logic Added:**
    
    - **4-Order Byte-Level PPM Tree (`PPMModel`):** Maintains top-16 frequency-sorted symbol buckets across Orders 1–4 plus an Order-0 fallback. Computes a full 256-symbol distribution once per byte using **Method C escape probabilities** and **symbol exclusion** (preventing symbols predicted at higher orders from diluting lower-order probability mass), then marginalizes bit-by-bit into a 16-input neural mixer.
        
    - **Precomputed `MathLUT` (`4096`-entry `stretch`/`squash` tables):** Eliminated inner-loop `std::log` and `std::exp` calls during bit prediction and SSE mapping.
        
    - **Sparse Input Backprop & AVX2 Alignment:** Optimized `DeepOnlineByteRNN` to update only the active one-hot input column (`prev_byte_cache`) instead of scanning 256 inputs, with `32-byte` alignment and `__restrict__` pointers for SIMD auto-vectorization.
        
- **Takeaway:** Simultaneously improved compression (`-1,756 B` down to `1.950 bpb` on cold-start 1 MB text) **and** cut runtime by `1.26s` (`11.04s` $\rightarrow$ `9.78s`) despite adding a full 4-order PPM model, while keeping the UPX-packed executable at just **`26,512` bytes**.




====================================================================================
Compressor                       |   Total Bytes | Bits/Byte | Notes
------------------------------------------------------------------------------------
Original enwik8                  |   100,000,000 |    8.0000 | Uncompressed
gzip -9                          |    36,445,248 |    2.9156 | Standard DEFLATE
xz -9 (LZMA2)                    |    24,865,252 |    1.9892 | Best standard Linux archiver
YOUR mini_cmix_v4 (arc + bin)    |    22,207,727 |    1.7766 | arc=22,185,727 + bin=22,000
YOUR mini_cmix_v5 (arc + bin)    |    22,028,720 |    1.7623 | arc=22,002,208 + bin=26,512
paq8f (2006 Hutter Baseline)     |    18,324,887 |    1.4660 | Official pre-prize baseline
phda9 (2017 enwik8 Record)       |    15,284,944 |    1.2228 | Official enwik8 world record
cmix v21 (LTCB enwik8 SOTA)      |    14,623,723 |    1.1699 | All-time enwik8 SOTA