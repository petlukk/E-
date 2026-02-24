# Eä Compute Patterns

Five compute classes. Each has a memory model, a dependency structure,
and a measurable boundary where it wins or loses.

This is not marketing. This is measurement.

## The five classes

```
                    ┌─────────────────────────────────────────┐
                    │          Compute Patterns               │
                    ├──────────────┬──────────────────────────┤
                    │  No history  │  Has history             │
  ┌─────────────┐  ├──────────────┼──────────────────────────┤
  │ Single pass │  │ 1. Streaming │ 2. Reduction             │
  ├─────────────┤  ├──────────────┼──────────────────────────┤
  │ Neighborhood│  │ 3. Stencil   │ (stencil + reduction     │
  │             │  │              │  = multi-pass pipeline)   │
  ├─────────────┤  ├──────────────┼──────────────────────────┤
  │ Multi-frame │  │ 4. Streaming │ (accumulate + scale      │
  │             │  │    Dataset   │  = iterative streaming)   │
  ├─────────────┤  ├──────────────┼──────────────────────────┤
  │ Fused       │  │ 5. Fused     │ (diff + threshold + count│
  │ pipeline    │  │    Pipeline  │  = one pass, no RAM)     │
  └─────────────┘  └──────────────┴──────────────────────────┘
```

---

## 1. Streaming Kernel

**Pattern:** `out[i] = f(in[i])`

Each output depends only on the corresponding input element.
No neighborhood. No accumulator. No history.

```
  in[0]  in[1]  in[2]  in[3]  in[4]  in[5]  in[6]  in[7]
    │      │      │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
  f(·)   f(·)   f(·)   f(·)   f(·)   f(·)   f(·)   f(·)
    │      │      │      │      │      │      │      │
    ▼      ▼      ▼      ▼      ▼      ▼      ▼      ▼
 out[0] out[1] out[2] out[3] out[4] out[5] out[6] out[7]
```

### Memory model

- Reads: N elements
- Writes: N elements
- Extra memory: 0
- Bandwidth: 2 * N * sizeof(f32)

### Eä example

```
export func threshold_f32x8(data: *restrict f32, out: *mut f32, len: i32, thresh: f32) {
    let vthresh: f32x8 = splat(thresh)
    let vone: f32x8 = splat(1.0)
    let vzero: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 8 <= len {
        let v: f32x8 = load(data, i)
        store(out, i, select(v .> vthresh, vone, vzero))
        i = i + 8
    }
}
```

### When it wins

- Compute intensity is high relative to memory traffic (FMA: 2 flops per element)
- Multiple operations fused in one pass (diff + abs in a single loop)
- Called from Python where function call overhead dominates

### When it doesn't win

- **Simple element-wise operations against NumPy.** NumPy's `np.abs(a - b)` calls
  optimized BLAS/MKL routines that are already vectorized. For a single operation on
  contiguous data, NumPy is within 1-2x of optimal.
- Memory-bound at scale. Once the data exceeds L3 cache, all implementations hit
  the same DRAM bandwidth wall.

### Measured

Video anomaly demo, 1280x720 (921K elements):
```
NumPy  (abs + threshold + sum) :  3.2 ms
Eä     (3 kernel calls)        :  2.6 ms    1.2x
```

Eä's advantage is small because the operations are simple and memory-bound.
This is honest. The value of Eä for streaming is composition and control,
not raw throughput on trivial transforms.

### Real-world instances

- Image threshold, gamma correction, color space conversion
- Audio gain, normalization, clipping
- Sensor calibration (offset + scale per channel)

---

## 2. Reduction Kernel

**Pattern:** `scalar = reduce(in[0..N])`

All inputs contribute to a single output. The accumulator creates a
loop-carried dependency chain that limits IPC unless explicitly broken.

```
  in[0]  in[1]  in[2]  in[3]  in[4]  in[5]  in[6]  in[7]
    │      │      │      │      │      │      │      │
    └──┬───┘      └──┬───┘      └──┬───┘      └──┬───┘
     acc0           acc0          acc1           acc1
       │              │             │              │
       └──────┬───────┘             └──────┬───────┘
            acc0                         acc1         ← two independent chains
              │                            │
              └────────────┬───────────────┘
                        merge
                           │
                         scalar
```

### Memory model

- Reads: N elements
- Writes: 0 (returns scalar)
- Extra memory: K accumulators (K * vector_width * sizeof(f32))
- Bandwidth: N * sizeof(f32)

### Eä example

```
export func sum_f32x8(data: *restrict f32, len: i32) -> f32 {
    let mut acc0: f32x8 = splat(0.0)
    let mut acc1: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 16 <= len {
        acc0 = acc0 .+ load(data, i)
        acc1 = acc1 .+ load(data, i + 8)
        i = i + 16
    }
    let mut total: f32 = reduce_add(acc0 .+ acc1)
    while i < len {
        total = total + data[i]
        i = i + 1
    }
    return total
}
```

### When it wins

- **Always, if the programmer expresses ILP.** A single-accumulator reduction runs
  at ~0.25 IPC on Zen 1 (4-cycle latency, 1-cycle throughput for vaddps). Two
  accumulators reach ~0.5 IPC. Four reach ~1.0 IPC.
- LLVM does not auto-unroll reduction loops across LLVM versions. The programmer
  must express the parallelism. This is Eä's first design advantage.

### When it doesn't win

- If the reduction is trivially short (< 1000 elements), overhead dominates.
- If the reduction is memory-bound (data exceeds L3), bandwidth limits throughput
  regardless of ILP.

### Measured

Horizontal reduction benchmarks, 1M elements, AMD Ryzen 7 1700:
```
Sum (f32x8, multi-acc):
  C f32x8 (AVX2)  :  110 us
  Ea f32x8         :  105 us    0.96x (faster)

Max (f32x4, multi-acc):
  C f32x4 (SSE)   :  100 us
  Ea f32x4         :   78 us    0.78x (faster)
```

### Why the compiler cannot do this for you

```
// Single accumulator: serial dependency
cycle 0: vaddps ymm0, ymm0, [mem]    ; must wait for ymm0
cycle 3: vaddps ymm0, ymm0, [mem]    ; 3 cycles idle
cycle 6: vaddps ymm0, ymm0, [mem]

// Two accumulators: pipelined
cycle 0: vaddps ymm0, ymm0, [mem]    ; chain A
cycle 1: vaddps ymm1, ymm1, [mem]    ; chain B (independent)
cycle 3: vaddps ymm0, ymm0, [mem]    ; chain A ready
cycle 4: vaddps ymm1, ymm1, [mem]    ; chain B ready
```

LLVM optimizes instructions within a dependency graph.
It does not restructure the graph itself.
The programmer defines the graph. Eä preserves it.

### Real-world instances

- Statistical aggregation (sum, mean, variance, min, max)
- Histogram computation
- Anomaly counting (threshold + sum)
- Signal energy measurement
- Checksum / hash accumulation

---

## 3. Stencil Kernel

**Pattern:** `out[x,y] = f(neighborhood(in, x, y))`

Each output depends on a fixed neighborhood of inputs.
The neighborhood shape is known at compile time.

```
  ┌───┬───┬───┐
  │-1 │ 0 │+1 │  ← 3x3 Sobel Gx kernel
  ├───┼───┼───┤
  │-2 │ 0 │+2 │     9 loads per 4 output pixels
  ├───┼───┼───┤     8 multiply-adds
  │-1 │ 0 │+1 │     overlapping neighborhoods share loads
  └───┴───┴───┘
```

### Memory model

- Reads: N * K loads (K = stencil size, but overlapping reduces effective reads)
- Writes: N elements
- Extra memory: 0 (registers hold the neighborhood)
- Access pattern: sequential rows, stride = width

### Eä example

```
// Sobel: 8 loads per 4 output pixels (center row reused from Gy)
let r0a: f32x4 = load(input, row_above + x - 1)
let r0b: f32x4 = load(input, row_above + x)
let r0c: f32x4 = load(input, row_above + x + 1)
let r1a: f32x4 = load(input, row_curr + x - 1)
let r1c: f32x4 = load(input, row_curr + x + 1)
let r2a: f32x4 = load(input, row_below + x - 1)
let r2b: f32x4 = load(input, row_below + x)
let r2c: f32x4 = load(input, row_below + x + 1)

let gx: f32x4 = (r0c .- r0a) .+ (r1c .- r1a) .* vtwo .+ (r2c .- r2a)
```

### When it wins

- **Image processing pipelines.** Stencils have high arithmetic intensity (many
  operations per load) and predictable access patterns. CPU caches handle the
  row-sequential access well. SIMD processes multiple output pixels per iteration.
- When the stencil is small (3x3, 5x5). Register file holds the entire neighborhood.
- When the image fits in L2/L3 cache.

### When it doesn't win

- Large stencils (> 7x7) increase register pressure and may spill.
- Images much larger than L3 cache become bandwidth-bound.
- GPU wins when the image is very large and the stencil is compute-heavy,
  because GPU memory bandwidth is 5-10x higher than CPU.

### Measured

Sobel edge detection, 1920x1080, AMD Ryzen 7 1700:
```
NumPy   (array slicing) :  28.9 ms
OpenCV  (optimized C++) :   8.3 ms
Eä      (f32x4 stencil) :   3.1 ms    2.7x faster than OpenCV
```

### Why explicit SIMD matters here

An auto-vectorizer sees scalar code and must prove that vectorization is safe.
For stencils with overlapping loads, this is complex and fragile.

Eä's explicit SIMD means:
- The programmer controls which 4 pixels are processed together
- Overlapping loads are expressed directly (no aliasing ambiguity)
- Register allocation is visible (8 vector registers for the neighborhood)
- The generated assembly is predictable across LLVM versions

### Real-world instances

- Edge detection (Sobel, Canny, Laplacian)
- Image blur (Gaussian, box, median approximation)
- Convolution (arbitrary kernel weights)
- Morphological operations (erosion, dilation)
- Finite difference methods (PDE solvers, fluid simulation)
- Seismic data processing

---

## 4. Streaming Dataset Kernel

**Pattern:** Process N items one at a time, accumulating state.

Not a single-array operation. A loop over a sequence of inputs,
each processed by a streaming kernel, with shared persistent state.

```
  frame[0]  frame[1]  frame[2]  ...  frame[N-1]
     │         │         │              │
     ▼         ▼         ▼              ▼
  ┌──────────────────────────────────────────┐
  │  acc[i] += frame[i]   (called N times)   │
  └──────────────────────────────────────────┘
     │
     ▼
  ┌──────────────────────────────────────────┐
  │  out[i] = acc[i] * (1/N)  (called once)  │
  └──────────────────────────────────────────┘
     │
     ▼
   result
```

### Memory model

- Reads: N * pixels (one frame at a time)
- Writes: pixels (accumulator, updated in-place)
- Extra memory: **O(pixels)** — one accumulator array
- Peak memory: O(pixels + frame_size) — accumulator + one frame buffer

Compare with batch processing:
- NumPy `np.mean(stack, axis=0)`: allocates O(N * pixels) to hold all frames
- For N=16, 1024x1024: Eä uses 4 MB, NumPy uses 64 MB

### Eä example

```
// Called once per frame
export func accumulate_f32x8(acc: *mut f32, frame: *restrict f32, len: i32) {
    let mut i: i32 = 0
    while i + 8 <= len {
        store(acc, i, load(acc, i) .+ load(frame, i))
        i = i + 8
    }
}

// Called once after all frames
export func scale_f32x8(data: *restrict f32, out: *mut f32, len: i32, factor: f32) {
    let vfactor: f32x8 = splat(factor)
    let mut i: i32 = 0
    while i + 8 <= len {
        store(out, i, load(data, i) .* vfactor)
        i = i + 8
    }
}
```

The caller (Python, C, Rust) owns the loop over frames.
The kernel processes one frame. This is the separation.

### When it wins

- **Always, when N is large.** The memory advantage is O(N). For 100 frames of
  4K video (3840x2160), batch processing needs 3.3 GB. Streaming needs 33 MB.
- When frames arrive incrementally (camera, telescope, network stream).
  Batch processing must wait for all frames. Streaming processes as they arrive.
- When the per-frame operation is simple (accumulate, max, min).
  The kernel is a single streaming pass — cache-friendly, branch-free.

### When it doesn't win

- When N is small (< 4). The overhead of N kernel calls via ctypes may exceed
  the memory savings.
- When the operation requires random access across frames (e.g., temporal median
  needs all N values per pixel simultaneously). This requires batch or tiled approaches.

### Measured

Astronomy stacking, 16 frames, 1024x1024, AMD Ryzen 7 1700:
```
NumPy   (np.mean, batch)     :  39.0 ms    64 MB peak
Eä      (accumulate + scale) :   6.2 ms     4 MB peak    6.3x faster
```

The speedup comes from two sources:
1. **Cache efficiency.** Eä touches one frame at a time. The accumulator stays in L2.
   NumPy allocates a 64 MB array that thrashes L3.
2. **No allocation.** NumPy's `np.array(frames)` copies all frames into a contiguous
   block before computing. Eä accumulates in-place.

### Real-world instances

- Telescope image stacking (noise reduction)
- Particle physics event accumulation
- Radar signal integration
- Factory inspection (reference comparison over time)
- Satellite image compositing
- Video background subtraction (running average)
- Time-series sensor aggregation

---

## 5. Fused Pipeline Kernel

**Pattern:** Multiple operations in a single pass. No intermediate arrays.

The key insight: if data leaves registers between operations, you ended
a kernel too early.

```
  BEFORE: 3 kernels, 3 memory passes

  a[i], b[i]  →  diff[i]   →  mask[i]   →  count
               (write RAM)   (write RAM)   (read RAM)
               (read RAM)    (read RAM)

  ctypes ×3
  intermediate arrays ×2
  memory passes ×3


  AFTER: 1 fused kernel, 1 memory pass

  a[i], b[i]  →  |diff|  →  >thresh?  →  accumulate  →  count
                (register)  (register)    (register)

  ctypes ×1
  intermediate arrays ×0
  memory passes ×1
```

### Memory model

- Reads: N elements (input only)
- Writes: 0 (returns scalar or minimal output)
- Extra memory: **0** — no intermediate arrays
- Bandwidth: minimal — each element loaded once, never written back

Compare with multi-kernel pipeline:
- 3 separate kernels: 3 reads + 3 writes = 6N memory operations
- Fused kernel: 2 reads + 0 writes = 2N memory operations

### Eä example

```
// Fused: diff + threshold + count in one pass
export func anomaly_count_fused(a: *restrict f32, b: *restrict f32, len: i32, thresh: f32) -> f32 {
    let vzero: f32x8 = splat(0.0)
    let vone: f32x8 = splat(1.0)
    let vthresh: f32x8 = splat(thresh)
    let mut acc0: f32x8 = splat(0.0)
    let mut acc1: f32x8 = splat(0.0)
    let mut i: i32 = 0
    while i + 16 <= len {
        let va0: f32x8 = load(a, i)
        let vb0: f32x8 = load(b, i)
        let diff0: f32x8 = va0 .- vb0
        let abs0: f32x8 = select(diff0 .< vzero, vzero .- diff0, diff0)
        acc0 = acc0 .+ select(abs0 .> vthresh, vone, vzero)

        let va1: f32x8 = load(a, i + 8)
        let vb1: f32x8 = load(b, i + 8)
        let diff1: f32x8 = va1 .- vb1
        let abs1: f32x8 = select(diff1 .< vzero, vzero .- diff1, diff1)
        acc1 = acc1 .+ select(abs1 .> vthresh, vone, vzero)

        i = i + 16
    }
    return reduce_add(acc0 .+ acc1)
    // ... scalar tail omitted
}
```

### When it wins

- When the fused kernel's compute intensity does not exceed the memory traffic
  it eliminates. Fusion trades memory passes for register compute. If the
  added compute is less than the removed memory cost, fusion wins.
- When the unfused pipeline is memory-bound. Fusion converts a memory-bound
  pipeline into a compute-bound kernel.
- When FFI overhead is significant. One ctypes call instead of three removes
  ~0.3-0.5 ms of fixed cost on small data.
- When data exceeds L3 cache. Fusion's benefit grows with image size because
  eliminated intermediates become DRAM misses instead of cache hits.

### When it doesn't help

- When the pipeline has only one stage (nothing to fuse).
- When intermediate results are needed by the caller (diff image for visualization).
- When stages have fundamentally different access patterns (e.g., stencil followed
  by global reduction — the stencil must complete before the reduction can begin).

### When it hurts

- **When the fused kernel introduces more compute than it removes memory traffic.**
  A naive Gaussian+Sobel fusion that computes 8 separate Gaussian blurs per output
  pixel (~120 ops, 25 loads) is *slower* than unfused (blur: 9 loads + 1 store,
  sobel: 8 loads + 1 store, threshold: 1 load + 1 store). The unfused intermediates
  have spatial locality — adjacent pixels share most of their 3x3 neighborhoods,
  so L1/L2 cache serves the "intermediate" reads cheaply. The naive fusion paid
  compute cost without recovering equivalent memory savings.

  The fix: precompute the algebraic composition of Gaussian and Sobel as a single
  5x5 convolution kernel (~50 ops, 24 loads). Same fusion, different formulation.
  The rewritten kernel is faster than unfused — and the gap widens with image size.

  This is the same failure mode as XLA fusion failures, TVM schedule tuning,
  Halide algorithm-vs-schedule bugs, and CUDA kernel fusion disasters.

  **Fusion does not make bad kernels fast. Fusion amplifies good kernel design.**

### Measured

Video anomaly detection, 768x576, real video data (OpenCV vtest.avi):
```
NumPy              :  1.10 ms
OpenCV (C++)       :  0.97 ms
Ea (3 kernels)     :  1.12 ms    0.8x vs NumPy (slower — FFI + memory overhead)
Ea fused (1 kernel):  0.08 ms   13.4x vs NumPy, 11.9x vs OpenCV
```

Fusion speedup: **13.7x** (3 kernels → 1 kernel).

The unfused Ea was *slower* than NumPy. The fused Ea is *13x faster*.
Nothing changed in the compiler, the LLVM version, or the language.
Only the kernel boundary changed.

### Measured: stencil fusion (skimage pipeline)

Edge detection pipeline (blur → sobel → threshold), Kodak 768x512 image:
```
NumPy (4 stages)     :   9.75 ms
Ea unfused (4 calls) :   1.57 ms    6.2x vs NumPy
Ea fused (2 calls)   :   1.65 ms    5.9x vs NumPy
```

Fusion speedup at 768x512: **1.0x** — no measurable benefit.
Both Ea versions are ~6x faster than NumPy from native SIMD alone.

But fusion speedup grows with image size as intermediates leave cache:
```
  768x512    →  1.02x    (fits in L3)
 1920x1080   →  1.10x    (L3 boundary)
 3840x2160   →  1.33x    (DRAM-bound intermediates)
 4096x4096   →  1.30x    (DRAM-bound)
```

This is cache-theory confirmation: fusion's value is proportional to
the cost of eliminated memory traffic. When intermediates are in L1/L2
(stencils have spatial locality), fusion saves little. When intermediates
spill to DRAM (large images), fusion saves a lot.

Memory: NumPy 20.9 MB → Ea fused 3.0 MB (**7x reduction**).

**Critical insight:** The first fused kernel was *slower* than unfused (2.25 ms
vs 1.64 ms) because it computed 8 redundant Gaussian blurs per output pixel.
Algebraic reformulation — precomputing the combined Gaussian+Sobel as a single
5x5 separable convolution — reduced ops from ~120 to ~50 per 4 pixels.
The language didn't change. The LLVM didn't change. The compute formulation did.

### Fusion scaling: speedup grows linearly with pipeline depth

MNIST preprocessing, 60,000 images (47M pixels, 188 MB), real data:

```
Ops  NumPy      Ea fused   Speedup   NumPy passes   Ea passes
───  ─────      ────────   ───────   ────────────   ─────────
 1     77 ms      39 ms      2.0x          1            1
 2    154 ms      39 ms      4.0x          2            1
 4    470 ms      39 ms     12.0x          4            1
 6    756 ms      38 ms     19.8x          6            1
 8   1006 ms      40 ms     25.2x          8            1
```

```
  1 ops │███ 2.0x
  2 ops │██████ 4.0x
  4 ops │██████████████████ 12.0x
  6 ops │███████████████████████████████ 19.8x
  8 ops │████████████████████████████████████████ 25.2x
```

Ea fused time is **constant** (~39 ms) regardless of operation count.
NumPy scales linearly (~125 ms per additional memory pass).

Each additional operation in a fused Ea kernel costs nearly zero — it is
one more SIMD instruction operating on data already in registers. Each
additional NumPy operation costs a full RAM roundtrip (read + write 188 MB).

This is the fundamental scaling law of kernel fusion:
- **Unfused cost:** O(N × data_size) — N memory passes
- **Fused cost:** O(data_size) — 1 memory pass, N register operations
- **Speedup:** O(N) — linear in pipeline depth

### Why the compiler cannot fuse for you

Kernel fusion requires semantic knowledge: which operations compose, which
intermediate results are discarded, and what the final output is. This is
a design decision, not an optimization.

LLVM sees three separate function calls from Python. It cannot merge them.
Even a whole-program optimizer cannot fuse across FFI boundaries.

The programmer defines the compute boundary. The compiler optimizes within it.

### The principles

> **If data leaves registers, you probably ended a kernel too early.**

> **Fusion does not make bad kernels fast. Fusion amplifies good kernel design.**

The first principle tells you *when* to fuse. The second tells you that fusion
is necessary but not sufficient — the algebraic formulation of the fused kernel
must be efficient. A naively-composed fusion can increase compute faster than
it eliminates memory traffic. The programmer must understand both the compute
and the memory model.

This is the same problem that:
- XLA hits when fusion increases register pressure beyond spill threshold
- TVM hits when schedule search finds locally-optimal but globally-slow fusions
- Halide separates as "algorithm vs schedule" — the algorithm must be right first
- CUDA programmers learn after their first fused kernel is slower than the unfused version

Eä makes it explicit: the programmer writes the fused kernel.
No magic. No heuristics. The code *is* the optimization.
The kernel must also be *the right* optimization.

### Real-world instances

- Image processing pipelines (blur + threshold + count)
- Signal processing chains (filter + detect + measure)
- Feature extraction (gradient + magnitude + suppression)
- Anomaly detection (diff + classify + aggregate)
- Any multi-stage pipeline where intermediate data is not needed

---

## 6. Quantized Inference Kernel (u8×i8 → i16x8 / i32x4)

**Pattern:** Unsigned activations × signed weights → pairwise multiply-accumulate

Each output channel is a dot product of u8 activations and i8 weights.
The programmer explicitly chooses the output type — and with it, the overflow model:

- `maddubs_i16(u8x16, i8x16) -> i16x8` — pmaddubsw, 16 pairs/cycle, accumulator wraps at ±32,767
- `maddubs_i32(u8x16, i8x16) -> i32x4` — pmaddubsw + pmaddwd(ones), half the lanes, accumulator safe to ±2,147,483,647

Output type encodes the semantic choice. No silent widening. No magic.

```
  maddubs_i16:                           maddubs_i32:
  act[0..15]   wt[0..15]                act[0..15]   wt[0..15]
      │              │                      │              │
      └──────┬────────┘                     └──────┬────────┘
         pmaddubsw    ← 8 i16 lanes            pmaddubsw    ← 8 i16 lanes (intermediate)
             │                                     │
         acc (i16x8)  ← fast, wraps           pmaddwd(ones) ← adjacent i16 pairs → i32
                                                   │
                                               acc (i32x4)  ← safe, 4 lanes
```

### Memory model

- Reads: H × W × C_in activations (u8) + 9 × C_in weights per output channel (i8)
- Writes: (H-2) × (W-2) output elements (i16 or i32 depending on intrinsic)
- Extra memory: 2 accumulators × i16x8 (32 bytes) or 2 × i32x4 (32 bytes)
- Throughput: 16 multiply-adds per cycle for maddubs_i16; ~8 effective for maddubs_i32

### Eä example (inner loop)

```
// maddubs_i16 — fast, wraps at i16 boundary. Use when values are small.
let mut acc0: i16x8 = splat(0)
let mut acc1: i16x8 = splat(0)
let mut ci: i32 = 0
while ci < C_in {
    acc0 = acc0 .+ maddubs_i16(load(src_row, ci), load(wt_row, ci))
    acc1 = acc1 .+ maddubs_i16(load(src_row, ci + 16), load(wt_row, ci + 16))
    ci = ci + 32
}

// maddubs_i32 — safe accumulator. Use when large C_in or large values.
let mut acc0: i32x4 = splat(0)
let mut acc1: i32x4 = splat(0)
while ci < C_in {
    acc0 = acc0 .+ maddubs_i32(load(src_row, ci), load(wt_row, ci))
    acc1 = acc1 .+ maddubs_i32(load(src_row, ci + 16), load(wt_row, ci + 16))
    ci = ci + 32
}
```

5 lines express the innermost compute of a full 3×3 NHWC convolution.

### When it wins

- **Quantized inference.** Neural network activations fit in u8, weights in i8.
  `maddubs_i16` / `maddubs_i32` map directly to SSSE3+SSE2 instructions — exactly
  what TensorFlow Lite and ONNX Runtime use internally.
- **8-bit vision pipelines.** Camera input is u8. Keeping data in u8 avoids
  the 4× bandwidth expansion of float32.
- **C_in ≥ 32.** The dual-accumulator pattern requires at least 2 × 16 = 32 channels
  per iteration to fully utilize both accumulators.

### When it loses (or requires care)

- **Weights not pre-quantized.** If weights are float32, conversion adds overhead.
  Pre-quantize and store as i8 before calling the kernel.
- **Per-call overflow (both intrinsics).** pmaddubsw sums adjacent pairs into i16 before
  any widening. When `a[2i]*b[2i] + a[2i+1]*b[2i+1] > 32,767`, the result wraps —
  even for `maddubs_i32`. Safe range: both adjacent products combined ≤ 32,767.
  For symmetric values: act ≤ 127, wt ≤ 127 (127×127×2 = 32,258 — safe).
- **maddubs_i16 accumulator overflow.** i16 accumulates to ±32,767. `maddubs_i32`
  eliminates this — the i32 accumulator holds values to ±2 billion. Use `maddubs_i32`
  when accumulating over large C_in or many iterations.
- **C_in not a multiple of 32.** Requires padding. The kernel assumes aligned input.

### Measured

conv2d_3x3 on 56×56×64 input, 3×3 kernel (NHWC):
```
NumPy   (float32, unoptimized)  :  ~870 ms
Eä      (maddubs dual-acc)      :  ~18 ms    47.7x faster
Throughput: 38.5 GMACs/s
```

### Real-world instances

- Post-training quantization (PTQ) inference
- MobileNet, EfficientNet-lite, ResNet-int8 inference layers
- 8-bit convolutions in embedded vision (Coral, Hailo, ARM Ethos)
- Camera ISP pipelines (demosaic, denoise, sharpening in u8)
- Any pipeline where activations stay in u8 from input to output

---

## 7. Structural Scan Kernel

**Pattern:** `out[i] = classify(in[i])` — byte-domain integer SIMD.

Each input byte is classified using vector comparisons and bitwise ops.
No float. No neighborhood. No accumulator. Pure integer lane-parallel work.

This is the byte-domain analog of Pattern 1 (Streaming), but with different
fusion characteristics. Float-domain streaming operates on 4-8 lanes of 32-bit
data. Structural scanning operates on 16 lanes of 8-bit data — twice the
elements per instruction.

```
  in[0]  in[1]  in[2]  ...  in[14] in[15]
    │      │      │              │      │
    ▼      ▼      ▼              ▼      ▼
  ┌─────────────────────────────────────────┐
  │         classify (u8x16)                │
  │   cmp_ge + cmp_le + select + or         │
  └─────────────────────────────────────────┘
    │      │      │              │      │
    ▼      ▼      ▼              ▼      ▼
 out[0] out[1] out[2]  ...  out[14] out[15]
```

### Memory model

- Reads: N bytes
- Writes: N bytes
- Extra memory: 0
- Bandwidth: 2 * N bytes

### Eä example

```
export func classify_alpha(input: *restrict u8, out: *mut u8, len: i32) {
    let lower_a: u8x16 = splat(0x61)
    let lower_z: u8x16 = splat(0x7A)
    let upper_a: u8x16 = splat(0x41)
    let upper_z: u8x16 = splat(0x5A)
    let one: u8x16 = splat(1)
    let zero: u8x16 = splat(0)
    let mut i: i32 = 0
    while i + 16 <= len {
        let v: u8x16 = load(input, i)
        let is_lower: u8x16 = select(v .>= lower_a, select(v .<= lower_z, one, zero), zero)
        let is_upper: u8x16 = select(v .>= upper_a, select(v .<= upper_z, one, zero), zero)
        store(out, i, is_lower .| is_upper)
        i = i + 16
    }
}
```

### When it wins

- **Byte scanning at memory bandwidth.** 16 bytes per SIMD instruction means
  the classification loop runs at memory bandwidth on most CPUs. A single
  pcmpgtb + pand sequence classifies 16 characters per cycle.
- **Range-based classification.** Alphabetic, numeric, whitespace, and delimiter
  detection all reduce to a small number of compares and bitwise ops. No
  lookup tables. No branches.
- **Multiple classifications sharing one load.** A tokenizer prepass needs
  is_alpha, is_digit, and is_space masks. All three derive from the same
  loaded u8x16 — one load, three classification results.

### When it doesn't win

- **Lookup-table classification.** When the classification is irregular (e.g.,
  Unicode category mapping, arbitrary character sets), a 256-byte lookup table
  with pshufb is faster than a chain of range comparisons. Structural scan
  assumes range-structured input.
- **Small input (< 256 bytes).** FFI overhead from ctypes dominates. For short
  strings, a Python `str.isalpha()` or a C `isalpha()` loop is fast enough.

### When fusion hurts

This is the critical finding from the tokenizer prepass demo.

Tokenizer prepass (738 KB real text — Pride and Prejudice):
```
NumPy (6+ array ops)  :  18.51 ms
Eä unfused (3 calls)  :  0.24 ms    78.7x faster
Eä fused (1 call)     :  0.32 ms    58.1x faster
Fusion: 0.74x — fused is SLOWER than unfused.
```

This is the opposite of Pattern 5 (Fused Pipeline), where fusion always helps
for streaming float work. Why does fusion hurt here?

**The unfused pipeline:**
1. Classify bytes into masks (is_alpha, is_space, etc.) — one pass, write masks
2. Detect boundaries (mask[i] != mask[i-1]) — one pass, read masks, write boundaries
3. Count tokens — one pass, read boundaries

Each stage reads and writes ~738 KB. Total memory traffic: ~4.4 MB.
But 738 KB fits comfortably in L2/L3 cache. Cache reads are nearly free.

**The fused pipeline:**
1. One pass: classify + detect boundaries + count — no intermediate arrays

But boundary detection needs classification of adjacent bytes. In the fused kernel,
there is no mask array to index into. The kernel must re-classify the previous byte
to compare with the current byte. This means every byte is classified twice.

The unfused version writes 738 KB of masks to cache and reads them back cheaply.
The fused version eliminates that cache traffic but pays with redundant compute.
At 738 KB, the eliminated traffic costs ~0.05 ms (L2 latency). The redundant
compute costs ~0.08 ms (extra comparison chain per byte).

**Redundant compute > eliminated cache traffic. Fusion loses.**

This is distinct from Pattern 5's fusion failure (naive stencil fusion). There,
the failure was compute explosion from unshared neighborhood loads. Here, the
failure is that byte-domain intermediates are so small they fit in cache, making
the eliminated traffic nearly free.

The rule: **fusion helps when eliminated traffic hits DRAM. Fusion hurts when
eliminated traffic hits L2 and the fused kernel adds redundant compute.**

### Real-world instances

- Tokenizer preprocessing (byte classification, boundary detection)
- JSON structural scanning (find quotes, brackets, colons, commas)
- CSV delimiter detection (field boundaries, quote tracking)
- Log parsing (timestamp boundaries, severity markers)
- Compression preprocessing (byte frequency, run-length detection)
- UTF-8 validation (byte range classification, continuation byte detection)

---

## Summary

| Class | Bottleneck | Eä advantage | When NumPy is enough |
|-------|-----------|-------------|---------------------|
| Streaming | Memory bandwidth | Composition, no allocation | Single simple operation |
| Reduction | Dependency chain | Explicit ILP (multi-acc) | Small arrays (< 1K) |
| Stencil | Compute intensity | Explicit SIMD, register control | Never (NumPy is 10x slower) |
| Streaming Dataset | Peak memory | O(1) extra memory | Small N, small frames |
| Fused Pipeline | Memory passes | Zero intermediate arrays | Single-stage pipeline |
| Quantized Inference | Integer throughput | maddubs_i16 (16 pairs/cycle) / maddubs_i32 (safe i32 acc) | Weights not pre-quantized |
| Structural Scan | Memory bandwidth | 16 bytes/cycle, integer-only | Input < 256 bytes (FFI overhead) |

### The honest answer

Eä does not make everything faster.

For a single `np.abs(a - b)` on contiguous data, NumPy is fine.
For a single `cv2.threshold()`, OpenCV is fine.

Eä wins when:
- The operation has dependency structure that matters (reductions)
- The access pattern benefits from explicit SIMD (stencils)
- Memory residency matters (streaming datasets)
- Multiple operations compose without intermediate allocation (fused pipelines)

Eä loses when:
- The operation is trivially memory-bound and NumPy already calls optimized BLAS
- The data is small enough that function call overhead dominates
- The algorithm requires GPU-class memory bandwidth (large stencils on large images)

This is not a limitation. This is the design space.
A kernel language that claims to win everywhere is lying.

### The lessons from kernel fusion

**Lesson 1: Fusion eliminates memory passes (video anomaly demo).**

```
3 separate kernels :  1.12 ms   (slower than NumPy)
1 fused kernel     :  0.08 ms   (13x faster than NumPy)
```

Streaming operations (per-pixel, no neighborhood) fuse trivially — each pixel
is independent, so fusion replaces N memory passes with N register operations.
Speedup scales linearly with pipeline depth:

```
1 op  →   2.0x     (memory-bound baseline)
4 ops →  12.0x     (3 passes eliminated)
8 ops →  25.2x     (7 passes eliminated)
```

**Lesson 2: Fusion requires algebraic optimization (skimage pipeline demo).**

Stencil fusion is harder. A naive Gaussian+Sobel fusion that computes 8 separate
Gaussian blurs per output pixel was *slower* than unfused:

```
Ea unfused (4 calls) :  1.64 ms
Ea fused (naive)     :  2.25 ms   ← SLOWER (0.7x)
```

The fix was not a language change or a compiler change. It was a mathematical
reformulation: precompute the algebraic composition of Gaussian and Sobel as
a single 5x5 separable convolution. This reduced ops from ~120 to ~50 per
4 output pixels:

```
Ea unfused (4 calls) :  1.57 ms
Ea fused (optimized) :  1.65 ms   ← on par (small image, fits in cache)
```

And fusion's benefit grows as data leaves cache:

```
  768x512   →  1.02x    (L3-resident)
 3840x2160  →  1.33x    (DRAM-bound intermediates)
```

Same language. Same compiler. Same LLVM. Same data.

The only differences: **where the kernel boundary was drawn** and
**how the compute was formulated**.

Performance comes from expressing the computation correctly —
not from a better compiler, more features, or a smarter optimizer.

> **If data leaves registers, you probably ended a kernel too early.**

> **Fusion does not make bad kernels fast. Fusion amplifies good kernel design.**

This is the same challenge that drives CUDA kernel design, XLA fusion,
Halide algorithm/schedule separation, and every high-performance computing
framework: minimize data movement, maximize register residency, and get
the algebra right before fusing.
