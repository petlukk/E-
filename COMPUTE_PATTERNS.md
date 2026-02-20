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

- **Always, compared to the unfused version of the same pipeline.** This is not
  an optimization hint — it is an architectural transformation. Eliminating
  intermediate memory traffic is a guaranteed win.
- When the unfused pipeline is memory-bound. Fusion converts a memory-bound
  pipeline into a compute-bound kernel.
- When FFI overhead is significant. One ctypes call instead of three removes
  ~0.3-0.5 ms of fixed cost on small data.

### When it doesn't help

- When the pipeline has only one stage (nothing to fuse).
- When intermediate results are needed by the caller (diff image for visualization).
- When stages have fundamentally different access patterns (e.g., stencil followed
  by global reduction — the stencil must complete before the reduction can begin).

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

### The principle

> **If data leaves registers, you probably ended a kernel too early.**

This is the same principle behind:
- CUDA kernel fusion (reduce kernel launch overhead)
- XLA operator fusion (eliminate temporary tensors)
- Polyhedral compilation (fuse loop nests)

Eä makes it explicit: the programmer writes the fused kernel.
No magic. No heuristics. The code *is* the optimization.

### Real-world instances

- Image processing pipelines (blur + threshold + count)
- Signal processing chains (filter + detect + measure)
- Feature extraction (gradient + magnitude + suppression)
- Anomaly detection (diff + classify + aggregate)
- Any multi-stage pipeline where intermediate data is not needed

---

## Summary

| Class | Bottleneck | Eä advantage | When NumPy is enough |
|-------|-----------|-------------|---------------------|
| Streaming | Memory bandwidth | Composition, no allocation | Single simple operation |
| Reduction | Dependency chain | Explicit ILP (multi-acc) | Small arrays (< 1K) |
| Stencil | Compute intensity | Explicit SIMD, register control | Never (NumPy is 10x slower) |
| Streaming Dataset | Peak memory | O(1) extra memory | Small N, small frames |
| Fused Pipeline | Memory passes | Zero intermediate arrays | Single-stage pipeline |

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

### The lesson from kernel fusion

The video anomaly demo tells the full story:

```
3 separate kernels :  1.12 ms   (slower than NumPy)
1 fused kernel     :  0.08 ms   (13x faster than NumPy)
```

The MNIST scaling experiment confirms this is not an anomaly — it is a law:

```
1 op  →   2.0x     (memory-bound baseline)
2 ops →   4.0x     (1 pass eliminated)
4 ops →  12.0x     (3 passes eliminated)
8 ops →  25.2x     (7 passes eliminated)
```

Ea fused time is constant. NumPy time scales linearly with operations.
Speedup is proportional to pipeline depth.

Same language. Same compiler. Same LLVM. Same data. Same result.

The only difference: **where the kernel boundary was drawn.**

Performance comes from expressing the computation boundary correctly —
not from a better compiler, more features, or a smarter optimizer.

This is the same principle that drives CUDA kernel design, XLA fusion,
and every high-performance computing framework: minimize data movement,
maximize register residency, let the programmer define the compute boundary.

> **If data leaves registers, you probably ended a kernel too early.**
