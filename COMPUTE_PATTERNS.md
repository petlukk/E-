# Eä Compute Patterns

Four compute classes. Each has a memory model, a dependency structure,
and a measurable boundary where it wins or loses.

This is not marketing. This is measurement.

## The four classes

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

## Summary

| Class | Bottleneck | Eä advantage | When NumPy is enough |
|-------|-----------|-------------|---------------------|
| Streaming | Memory bandwidth | Composition, no allocation | Single simple operation |
| Reduction | Dependency chain | Explicit ILP (multi-acc) | Small arrays (< 1K) |
| Stencil | Compute intensity | Explicit SIMD, register control | Never (NumPy is 10x slower) |
| Streaming Dataset | Peak memory | O(1) extra memory | Small N, small frames |

### The honest answer

Eä does not make everything faster.

For a single `np.abs(a - b)` on contiguous data, NumPy is fine.
For a single `cv2.threshold()`, OpenCV is fine.

Eä wins when:
- The operation has dependency structure that matters (reductions)
- The access pattern benefits from explicit SIMD (stencils)
- Memory residency matters (streaming datasets)
- Multiple operations compose without intermediate allocation (pipelines)

Eä loses when:
- The operation is trivially memory-bound and NumPy already calls optimized BLAS
- The data is small enough that function call overhead dominates
- The algorithm requires GPU-class memory bandwidth (large stencils on large images)

This is not a limitation. This is the design space.
A kernel language that claims to win everywhere is lying.
