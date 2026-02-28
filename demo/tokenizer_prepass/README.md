# Tokenizer Prepass — Ea Demo

This demo implements a SIMD structural text scanner: classify every byte,
lowercase, and detect token boundaries. It is NOT a full tokenizer — it is a
pre-tokenization acceleration layer. Same strategy as simdjson: SIMD structural
scan first, then normal processing on the classified stream.

Three implementations compared:

- **NumPy** — idiomatic array operations (classify → lowercase → boundary detect)
- **Ea unfused** — 3 separate Ea kernel calls, same memory passes as NumPy
- **Ea fused** — 1 kernel call (classify + lowercase + boundary in one pass)

## Results

738,046 bytes (Pride and Prejudice, Project Gutenberg). 50 runs, median.

```
NumPy (3 stages)     :   17.74 ms
Ea unfused (3 calls) :    0.27 ms   (65.5x vs NumPy)
Ea fused (1 call)    :    0.32 ms   (56.1x vs NumPy)
```

Memory: NumPy 5.6 MB → Unfused 2.8 MB → Fused 2.1 MB.
Correctness: 100% exact match across all implementations.

## Why fusion is slower here

The fused kernel is **0.74x** — slower than unfused. This is the honest result.

The reason: the fused kernel classifies **both** the current byte and the
previous byte to detect boundaries inline. That doubles the classification
compute. Meanwhile, at 738 KB the intermediate flag array fits entirely in
L2/L3 cache, so eliminating it saves almost no memory traffic.

The fusion condition is:

> `compute_added <= memory_traffic_eliminated`

Here the condition is NOT satisfied. Classification is ~20 SIMD ops per 16
bytes. The fused kernel runs ~40 ops (2x classification) to avoid reading back
~738 KB of flags that were already in cache. The added compute exceeds the
saved traffic.

This is the same lesson as the skimage fusion demo's first attempt: naive
fusion that increases compute faster than it removes memory traffic makes
things slower. The difference is that the stencil fusion could be fixed by
algebraic reformulation. Here, there is no algebraic shortcut — boundary
detection inherently requires both current and previous classifications, and
the unfused path gets the previous classification for free from the already-
written flag array.

**Fusion does not make bad kernels fast. Fusion amplifies good kernel design.**

## Three-way comparison

| | NumPy | Ea unfused | Ea fused |
|---|---|---|---|
| Kernel calls | 3 Python | 3 ctypes | 1 ctypes |
| Memory passes | 3 | 3 | 1 |
| Intermediate arrays | ~8 temp | 1 (flags) | 0 |
| Peak memory | 5.6 MB | 2.8 MB | 2.1 MB |
| Time | 17.74 ms | 0.27 ms | 0.32 ms |
| vs NumPy | 1.0x | **65.5x** | **56.1x** |

The 65.5x speedup over NumPy comes from explicit SIMD on byte-width data.
NumPy's `np.isin()` and boolean masking create many temporary arrays and
operate through Python dispatch. Ea compiles to straight SSE2 `u8x16`
operations with zero allocation.

## The kernels

### Unfused (in `prepass.ea`)

Three separate streaming kernels:

- **`classify_u8x16`** — maps each byte to a flag: whitespace (1), letter (2),
  digit (4), punctuation (8), non-ASCII (16). Uses SIMD range checks via
  `select` + bitwise `.&` / `.|` on `u8x16` vectors.
- **`lowercase_u8x16`** — branchless `c .| (is_upper .& 0x20)`. The classic
  ASCII lowercase trick: OR the 0x20 bit if the byte is in A-Z range.
- **`boundary_detect`** — compares adjacent flag values. Where the flag class
  changes (letter→space, space→punctuation, etc.), a boundary is marked.

### Fused (in `prepass.ea`)

One kernel that produces all three outputs in a single pass over the input.
For each 16-byte chunk, it:

1. Loads `curr[i..i+16]` and `prev[i-1..i+15]`
2. Classifies both curr and prev (this is the 2x compute cost)
3. Lowercases curr via the OR-0x20 trick
4. Detects boundaries by comparing curr_flags vs prev_flags
5. Stores flags, lowered text, and boundary mask

Zero intermediate arrays. One memory pass. But 2x classification compute.

## What this demonstrates

**Ea extends beyond image processing.** This is byte-level text processing
using the same SIMD primitives — `u8x16` vectors, `select`, `.&`, `.|`, `.^`
— that drive the image kernels. The classification pattern (range-check via
compare + mask + OR flags) maps directly to the simdjson strategy of
structural character identification.

**Vector bitwise ops enable byte-level classification.** The `.&` / `.|`
operators on `u8x16` build flag bitmasks without branching. Each byte gets
classified in parallel — 16 bytes per SIMD cycle. This is the same pattern
used in production SIMD JSON parsers and UTF-8 validators.

**Honest fusion analysis.** Not every pipeline benefits from fusion. When the
intermediate data fits in cache and fusion doubles the compute, the unfused
path wins. The correct response is to report this honestly, not to hide it.
The unfused Ea kernels at 65.5x vs NumPy are the real result here.

**The unfused result is the headline.** Three simple, well-designed streaming
kernels — each doing one thing — deliver 65.5x over NumPy. Good kernel design
matters more than fusion. Fusion is a tool for when memory traffic is the
bottleneck, not a universal accelerator.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Build kernels
bash demo/tokenizer_prepass/build.sh

# Run demo
python3 demo/tokenizer_prepass/run.py
```

Requires: Python 3, NumPy.
Downloads Pride and Prejudice from Project Gutenberg on first run.
