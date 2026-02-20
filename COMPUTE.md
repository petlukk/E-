# Eä Compute Model

Eä is a compute kernel language. The programmer controls data layout, vector width,
and dependency structure. The compiler translates — it does not transform.

This document describes the mental model for writing kernels in Eä.

## What is a compute kernel?

A function that processes arrays of numbers. No allocation, no I/O, no exceptions.
Input pointers in, output pointers out. The caller owns memory.

```
export func kernel(input: *restrict f32, output: *mut f32, len: i32) { ... }
```

Kernels are compiled to object files or shared libraries and called from C, Python,
or any language with a C FFI.

## The six kernel patterns

Every compute workload is a combination of these patterns.

### 1. Streaming (element-wise)

Each output depends only on the corresponding input. No neighborhood, no history.

```
// out[i] = a[i] * b[i] + c[i]
let va: f32x8 = load(a, i)
let vb: f32x8 = load(b, i)
let vc: f32x8 = load(c, i)
store(out, i, fma(va, vb, vc))
```

Memory-bound at scale. Performance depends on bandwidth, not compute.
See `examples/fma.ea`.

### 2. Reduction

All inputs contribute to a single scalar output. The accumulator creates a
loop-carried dependency chain.

**Single accumulator** — serial, latency-bound:
```
acc = acc .+ load(data, i)    // each iteration waits for the previous
```

**Multi-accumulator** — parallel, throughput-bound:
```
acc0 = acc0 .+ load(data, i)      // independent
acc1 = acc1 .+ load(data, i + 8)  // independent
```

The CPU executes independent chains in parallel on its execution units.
Two accumulators double throughput. This is not a compiler optimization — the
programmer must express the parallelism.

See `examples/reduction.ea`, `examples/reduction_single.ea`, `examples/reduction_multi_acc.ea`.

### 3. Branchless conditional

SIMD has no branches. Use `select` to choose between values per lane.

```
// out[i] = (data[i] > threshold) ? 1.0 : 0.0
let mask: f32x8 = select(v .> vthresh, vone, vzero)
```

This compiles to a single compare + blend instruction. No branch misprediction.
See `examples/threshold.ea`.

### 4. Multi-pass

Some operations require multiple passes over the data. Normalization needs the
mean (reduction) before it can scale (streaming).

```
// Pass 1: sum reduction
total = reduce_add(acc0 .+ acc1)

// Pass 2: scale and shift
store(out, i, (load(data, i) .- voffset) .* vscale)
```

Each pass is a simple kernel. The caller orchestrates the sequence.
See `examples/normalize.ea`.

### 5. Stencil

Each output depends on a neighborhood of inputs. Classic example: 2D convolution.

```
// 3x3 convolution: 9 loads, 8 fma operations per 4 output pixels
let r0a: f32x4 = load(input, row_above + x - 1)
let r0b: f32x4 = load(input, row_above + x)
...
acc = fma(r0a, k00, acc)
acc = fma(r0b, k01, acc)
```

The 2D image is a flat array. Row offsets are computed explicitly: `y * width + x`.
Kernel weights are splatted once outside the loop.
See `examples/conv2d.ea`.

### 6. Kernel pipeline

Complex operations combine multiple patterns. Sobel edge detection is a
stencil (gradient) followed by an element-wise operation (magnitude).

```
// Stencil: compute Gx, Gy gradients
let gx: f32x4 = (r0c .- r0a) .+ (r1c .- r1a) .* vtwo .+ (r2c .- r2a)
let gy: f32x4 = (r2a .- r0a) .+ (r2b .- r0b) .* vtwo .+ (r2c .- r0c)

// Element-wise: |Gx| + |Gy|
let abs_gx: f32x4 = select(gx .< vzero, vzero .- gx, gx)
```

No `abs` intrinsic needed. `select` expresses it directly.
See `examples/sobel.ea`.

## Memory patterns

### Flat arrays

All data is `*f32` or `*mut f32`. Multi-dimensional data uses explicit indexing:

```
data[y * width + x]          // 2D
data[z * height * width + y * width + x]  // 3D
```

This is the same layout as C, NumPy (row-major), and most image libraries.

### Restrict pointers

`*restrict f32` tells the compiler that this pointer does not alias other pointers.
This enables LLVM to reorder loads and stores more aggressively.

```
export func kernel(input: *restrict f32, output: *mut f32, len: i32)
```

Use `restrict` on read-only input pointers. The output pointer is `*mut` (mutable).

### Alignment

`load` and `store` use element alignment (4 bytes for f32), not vector alignment.
This generates `vmovups` (unaligned) instead of `vmovaps` (aligned). The performance
difference is negligible on modern CPUs, and unaligned access is always safe.

## Dependency structure

This is the most important concept in Eä.

### Why the compiler cannot save you

LLVM optimizes instructions. It does not restructure dependency graphs.

A single-accumulator reduction has a loop-carried dependency chain:

```
cycle 0: vmaxps xmm0, xmm0, [mem]    ; xmm0 depends on previous xmm0
cycle 4: vmaxps xmm0, xmm0, [mem]    ; must wait 4 cycles (Zen 1 latency)
cycle 8: vmaxps xmm0, xmm0, [mem]
```

IPC = 0.25. Three execution units sit idle.

Two accumulators break the chain:

```
cycle 0: vmaxps xmm0, xmm0, [mem]    ; chain A
cycle 1: vmaxps xmm1, xmm1, [mem]    ; chain B (independent, can execute immediately)
cycle 4: vmaxps xmm0, xmm0, [mem]    ; chain A (xmm0 ready)
cycle 5: vmaxps xmm1, xmm1, [mem]    ; chain B (xmm1 ready)
```

IPC = 1.0. The CPU pipelines both chains.

This is instruction-level parallelism (ILP). The programmer expresses it.
The compiler preserves it. Neither creates it from the other's representation.

### When to use multiple accumulators

Use multiple accumulators when **all** of these are true:

1. The loop has a **recurrence** (output feeds back as input)
2. The operation has **latency > 1 cycle** (most SIMD ops on modern CPUs)
3. The loop is **compute-bound**, not memory-bound

Streaming kernels (fma, threshold) do not benefit — they have no recurrence.
Reductions (sum, max, min) always benefit.

### Choosing accumulator count

Match the operation latency divided by throughput:

- `vaddps` on Zen 1: latency 3, throughput 1 → 3 accumulators saturate
- `vmaxps` on Zen 1: latency 4, throughput 1 → 4 accumulators saturate
- In practice, 2 accumulators recover most of the performance

More accumulators increase register pressure. Two is the pragmatic default.

## Vector width

Eä supports `f32x4` (SSE, 128-bit) and `f32x8` (AVX2, 256-bit).

### Choosing width

| Width | Registers | Throughput | Compatibility |
|-------|-----------|------------|---------------|
| f32x4 | xmm (16)  | 4 floats/op | All x86-64 |
| f32x8 | ymm (16)  | 8 floats/op | AVX2 (2013+) |

Use f32x8 for throughput. Use f32x4 when register pressure matters (stencils)
or when targeting older hardware.

### Tail handling

SIMD processes N elements per iteration. The remainder is handled with a scalar loop:

```
while i + 8 <= len {
    // vector body
    i = i + 8
}
while i < len {
    // scalar tail
    i = i + 1
}
```

This is explicit by design. No masked operations, no predication.
The programmer sees exactly what runs.

## Design principles

**Explicit over implicit.** The programmer chooses vector width, loop structure,
accumulator count, and memory access pattern. The compiler does not second-guess.

**Predictable over optimal.** A kernel that is 5% slower but behaves identically
across LLVM versions is better than one that depends on optimizer heuristics.

**Structure over flags.** Performance comes from expressing the right dependency
structure, not from compiler flags or pragma hints.

**Kernels over programs.** Eä compiles functions, not applications. The host
language (C, Python, Rust) handles everything else.
