# Tokenizer Prepass Demo — Design

**Date:** 2026-02-23
**Demo #9** in the Ea lineup.

## Goal

SIMD structural text scanning as a pre-tokenization acceleration layer.
NOT a full tokenizer — a prepass that classifies bytes, lowercases text,
and detects token boundaries in a single fused kernel. Feeds into existing
BPE engines (HuggingFace tokenizers). Same strategy as simdjson: SIMD
structural scan, then normal algorithmic processing.

Extends Ea from image/float compute into byte/text processing.

## Prerequisite: Vector Bitwise Ops

Ea currently lacks vector bitwise operations. Text scanning requires
range-based classification: `(c >= 'A') .& (c <= 'Z')`.

**Add to the compiler (separate commit):**
- `.&` (vector AND), `.|` (vector OR), `.^` (vector XOR)
- All integer vector types: u8x16, i8x16, i16x8, i32x4, etc.
- Maps to LLVM `build_and`, `build_or`, `build_xor`
- Same pattern as `.+` maps to `build_int_add`
- End-to-end tests with known values

**Skip:** `.~` (NOT) — express as `.^ splat(0xFF)`. YAGNI.
**Skip:** `movemask` — specialized, not needed for this demo.

## Pipeline

```
NumPy baseline (6+ array ops, 6+ memory passes):
  text bytes → classify → lowercase → boundary detect
  Each step creates temporary NumPy arrays.

Ea unfused (3 kernel calls, 3 memory passes):
  text bytes → classify_u8x16 → lowercase_u8x16 → boundary_detect
  classify writes flags array. boundary reads flags → intermediate.

Ea fused (1 kernel call, 1 memory pass):
  text bytes → [fused: classify + lowercase + boundary] → 3 output arrays
  Everything in registers. Zero intermediates.
```

## Three-way comparison

| Version | Kernel calls | Memory passes | Intermediates |
|---------|-------------|---------------|---------------|
| NumPy baseline | 6+ array ops | 6+ | 5+ temporary arrays |
| Ea unfused | 3 ctypes calls | 3 | 1 (flags read by boundary) |
| Ea fused | 1 ctypes call | 1 | 0 |

## File structure

```
demo/tokenizer_prepass/
├── prepass_fused.ea        # fused: classify + lowercase + boundary
├── prepass_unfused.ea      # 3 separate kernels
├── run.py                  # benchmark, correctness, visual output
├── build.sh                # compile .ea files
└── README.md               # results + explanation
```

## Kernel architecture

### Fused kernel: `text_prepass_fused`

```
Signature:
  text_prepass_fused(
      text: *restrict u8,
      flags: *mut u8,
      lower: *mut u8,
      boundaries: *mut u8,
      len: i32
  )

Per 16 input bytes:
  1. Load curr = load(text, i)       // current 16 bytes
  2. Load prev = load(text, i - 1)   // shifted by 1 for boundary detect
  3. Classify curr → curr_flags      // bitflags per byte
  4. Classify prev → prev_flags      // same classification on shifted load
  5. Lowercase curr → curr_lower     // upper → lower via select + OR
  6. Boundary = curr_flags != prev_flags  // class change = token boundary
  7. Store flags[i], lower[i], boundaries[i]
```

### Classification bitflags (u8 per byte)

- bit 0 (0x01): whitespace (space 0x20, tab 0x09, newline 0x0A, CR 0x0D)
- bit 1 (0x02): letter (A-Z or a-z)
- bit 2 (0x04): digit (0-9)
- bit 3 (0x08): punctuation (printable non-alphanum non-whitespace)
- bit 4 (0x10): non-ASCII (> 0x7F)

### Classification SIMD logic

```ea
// Whitespace: 4 equality checks OR'd together
let ws: bool = (c .== splat(0x20)) .| (c .== splat(0x09))
            .| (c .== splat(0x0A)) .| (c .== splat(0x0D))

// Letter: two range checks OR'd
let upper: bool = (c .>= splat(0x41)) .& (c .<= splat(0x5A))
let lower_range: bool = (c .>= splat(0x61)) .& (c .<= splat(0x7A))
let letter: bool = upper .| lower_range

// Digit: one range check
let digit: bool = (c .>= splat(0x30)) .& (c .<= splat(0x39))

// Non-ASCII: one comparison
let nonascii: bool = c .> splat(0x7F)

// Punctuation: printable AND NOT already classified
let printable: bool = (c .>= splat(0x21)) .& (c .<= splat(0x7E))
let known: bool = ws .| letter .| digit
let f_punct: u8x16 = select(printable, splat(0x08), splat(0x00))
let f_punct: u8x16 = select(known, splat(0x00), f_punct)  // zero where known

// Combine bitflags via OR
let flags: u8x16 = select(ws, splat(0x01), splat(0x00))
                .| select(letter, splat(0x02), splat(0x00))
                .| select(digit, splat(0x04), splat(0x00))
                .| f_punct
                .| select(nonascii, splat(0x10), splat(0x00))
```

### Lowercase logic

```ea
let lowered: u8x16 = select(upper, chunk .| splat(0x20), chunk)
```

One `select` — if uppercase, set bit 5 (0x20). ASCII 'A' (0x41) → 'a' (0x61).

### Boundary detection (overlapping load)

```ea
let prev: u8x16 = load(text, i - 1)
// Classify prev → prev_flags (same classification logic)
let boundary: u8x16 = select(curr_flags .== prev_flags, splat(0x00), splat(0x01))
```

`load(text, i - 1)` for all `i >= 1`. First byte (i=0) handled in scalar
preamble as always-boundary.

### Unfused kernels

In `prepass_unfused.ea`:
- `classify_u8x16(text, flags, len)` — write classification bitflags
- `lowercase_u8x16(text, lower, len)` — write lowercased text
- `boundary_detect(flags, boundaries, len)` — compare adjacent flags

## NumPy baseline

```python
def classify_numpy(text_bytes):
    b = np.frombuffer(text_bytes, dtype=np.uint8)
    flags = np.zeros_like(b)
    flags |= np.isin(b, [0x20, 0x09, 0x0A, 0x0D]).astype(np.uint8) * 0x01
    flags |= (((b >= 0x41) & (b <= 0x5A)) | ((b >= 0x61) & (b <= 0x7A))).astype(np.uint8) * 0x02
    flags |= ((b >= 0x30) & (b <= 0x39)).astype(np.uint8) * 0x04
    flags |= ((b > 0x7F)).astype(np.uint8) * 0x10
    return flags

def lowercase_numpy(text_bytes):
    b = np.frombuffer(text_bytes, dtype=np.uint8).copy()
    upper = (b >= 0x41) & (b <= 0x5A)
    b[upper] |= 0x20
    return b

def boundary_numpy(flags):
    boundaries = np.zeros_like(flags)
    boundaries[0] = 1
    boundaries[1:] = (flags[1:] != flags[:-1]).astype(np.uint8)
    return boundaries
```

Each call creates temporary arrays. 6+ memory passes total.

## Test input

Wikipedia text extract — real prose with mixed letters, digits, punctuation,
Unicode (accents, CJK). Download a ~1 MB sample from enwik8 or use
Project Gutenberg text. Fallback to synthetic text if download fails.

## Correctness

Byte-exact match. No floating-point issues — everything is integer/bitwise.
Fused and unfused must produce identical results. NumPy baseline must match
on the ASCII subset (non-ASCII classification may differ in edge cases).

## Visual output

Terminal-based, not PNG. Show a text excerpt with:
- Colored classification flags per byte
- Token boundaries marked with `|`
- Example: `|Hello|,| |world|!| |42|`

## Expected performance story

Byte ops are cheap per element (no FP multiplications). Compute intensity
is low → pipeline is memory-bound quickly. This is the opposite of
skimage: there, compute was expensive and fusion helped moderately.
Here, compute is cheap and fusion should help dramatically — same pattern
as video_anomaly (streaming ops, 13x fusion speedup).

Expected:
- Ea vs NumPy: significant speedup (NumPy array ops have per-call overhead)
- Fusion speedup: substantial (memory-bound pipeline, intermediates eliminated)
- Memory reduction: ~3x (eliminate intermediate flags array)

## What this demonstrates

1. **Ea expands beyond image compute.** Same SIMD principles, new domain (bytes).
2. **simdjson strategy for tokenizers.** SIMD structural scan → normal BPE.
3. **Fusion works for bytes like it works for floats.** Low compute intensity
   = memory-bound = fusion wins big.
4. **Vector bitwise ops** (`.&`, `.|`, `.^`) enable byte-level classification
   patterns that were not expressible before.

## Narrative

- Lead with: "Ea is not just an image processing language."
- Show: text scanning as a fusion candidate
- Honest about scope: "This is the pre-tokenizer, not the tokenizer."
- Connect to COMPUTE_PATTERNS.md: streaming fusion at byte width
