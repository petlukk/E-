# Eä Benchmarks

Measured on two machines. Full methodology and scripts in `benchmarks/`.

**Eä uses strict IEEE floating point — no fast-math flags.** The C reference was
compiled with `gcc -O3 -march=native -ffast-math`. Eä matching this baseline
without fast-math is the stronger claim.

Competitors are optional — benchmarks run with whatever toolchains are installed.
GCC is required; Clang, ISPC, and Rust nightly are detected and included automatically.

---

## AMD Ryzen 7 1700 (Zen 1, AVX2/FMA)

GCC 11.4, LLVM 18, Linux (WSL2). 1M elements, 100–200 runs, averaged.

### FMA Kernel (result[i] = a[i] \* b[i] + c[i])

| Implementation   | Avg (us) | vs Fastest C |
| ---------------- | -------- | ------------ |
| GCC f32x8 (AVX2) | 887      | 1.03x        |
| **Ea f32x4**     | **885**  | **1.03x**    |
| Clang-14 f32x8   | 859      | 1.00x        |
| ISPC             | 828      | 0.96x        |
| Rust std::simd   | 1001     | 1.17x        |

### Sum Reduction

| Implementation   | Avg (us) | vs Fastest C |
| ---------------- | -------- | ------------ |
| C f32x8 (AVX2)   | 110      | 1.00x        |
| **Ea f32x8**     | **105**  | **0.96x**    |
| Clang-14 f32x8   | 130      | 1.18x        |
| ISPC             | 105      | 0.95x        |
| Rust f32x8       | 111      | 1.01x        |

### Max Reduction (multi-accumulator)

| Implementation  | Avg (us) | vs Fastest C |
| --------------- | -------- | ------------ |
| C f32x4 (SSE)   | 100      | 1.00x        |
| **Ea f32x4**    | **78**   | **0.83x**    |
| Clang-14 f32x4  | 89       | 0.95x        |
| ISPC            | 71       | 0.76x        |
| Rust f32x4      | 180      | 1.93x        |

### Min Reduction (multi-accumulator)

| Implementation  | Avg (us) | vs Fastest C |
| --------------- | -------- | ------------ |
| C f32x4 (SSE)   | 80       | 1.00x        |
| **Ea f32x4**    | **78**   | **0.97x**    |
| Clang-14 f32x4  | 88       | 1.10x        |
| ISPC            | 73       | 0.91x        |
| Rust f32x4      | 220      | 2.77x        |

Eä's reduction kernels use explicit multi-accumulator patterns to break dependency
chains — see `examples/reduction_multi_acc.ea`. This is faster than relying on
compiler auto-unrolling and stable across LLVM versions.

---

## Intel i7-1260P (Alder Lake, AVX2/FMA)

LLVM 18, Linux (WSL2). 1M elements, 100–200 runs, **minimum** time reported.

### FMA Kernel (1M f32, 100 runs, min time)

| Implementation       | Min (us) |
| -------------------- | -------- |
| GCC f32x8 (AVX2)     | ~395     |
| Clang-14 f32x8       | ~396     |
| **Ea f32x4**         | **~326** |
| Clang-14 f32x4       | ~376     |
| Rust std::simd f32x4 | ~365     |
| ISPC                 | ~656     |

### Sum Reduction (1M f32, 200 runs, min time)

| Implementation | Min (us) |
| -------------- | -------- |
| ISPC           | ~62      |
| **Ea f32x8**   | **~65**  |
| Rust f32x8     | ~66      |
| C f32x8 (AVX2) | ~68      |

---

## foreach vs Explicit SIMD

v0.6.0 added `foreach` (auto-vectorized scalar loops) and `unroll(N)` hints.
Measured on Intel i7-1260P (Alder Lake), 1M f32 elements, 100 runs.

### FMA (streaming — element-independent)

| Implementation       | Avg (us) | vs GCC f32x8 |
| -------------------- | -------- | ------------ |
| GCC f32x8 (AVX2)     | 2629     | 1.00x        |
| GCC f32x4 (SSE)      | 2425     | 0.92x        |
| GCC scalar           | 3740     | 1.42x        |
| Ea f32x8             | 3176     | 1.21x        |
| Ea f32x4             | 5775     | 2.20x        |
| **Ea foreach**       | **2667** | **1.01x**    |
| Ea foreach+unroll    | 3148     | 1.20x        |

`foreach` matches GCC f32x8 on streaming FMA — LLVM auto-vectorizes the trivial
element-independent pattern. The `unroll` hint does not help here; LLVM already
unrolls optimally.

### Sum Reduction (cross-element dependency)

| Implementation       | Avg (us) | vs C scalar |
| -------------------- | -------- | ----------- |
| C scalar             | 578      | 1.00x       |
| C f32x8 (AVX2)       | 634      | 1.10x       |
| Ea f32x8 (multi-acc) | 929      | 1.61x       |
| Ea f32x4             | 1376     | 2.38x       |
| Ea unroll(4)         | 2961     | 5.12x       |
| **Ea foreach**       | **4175** | **7.22x**   |

`foreach` is 7x slower than C scalar for sum reduction. LLVM cannot automatically
break the accumulator dependency chain into multiple independent accumulators.
Explicit SIMD with multi-accumulator ILP remains essential for reductions.

### Interpretation

`foreach` is a good default for element-wise streaming operations where each
output depends only on its own index. For anything with cross-element dependencies
(reductions, prefix sums, recurrences), explicit SIMD with the multi-accumulator
pattern is necessary.

This matches the language design: `foreach` is convenience for simple cases,
explicit vector types are control for performance-critical patterns.

See `examples/foreach_fma.ea` and `examples/foreach_reduction.ea` for
side-by-side comparison code.

---

## `restrict` / noalias analysis

The `noalias` attribute is correctly emitted in LLVM IR, but it has no measurable impact on these benchmarks because:

- The generated assembly is byte-identical with and without `restrict` (confirmed via `objdump -d` + MD5 comparison)
- Eä's explicit SIMD intrinsics (`load` / `store` / `fma`) already use distinct base pointers, so LLVM's alias analysis does not require `noalias` hints
- Eä's explicit SIMD means the loop vectorizer and SLP passes have little to contribute beyond what is already expressed
- Reduction kernels have a single pointer parameter, making `noalias` vacuous

The implementation is correct and complete — it is simply not performance-relevant for these specific kernels. The feature is positioned for value when the optimizer pipeline grows (auto-tiling, software pipelining, prefetching) or when users write more complex aliasing patterns.

---

## Performance Principle

LLVM optimizes instructions. Eä lets you optimize dependency structure.

A single-accumulator reduction creates a serial chain — each iteration waits for
the previous one. On a superscalar CPU, this wastes execution units:

```
// Single accumulator: serial dependency, ~0.25 IPC on Zen 1
acc = max(acc, load(data, i))   // must wait for previous acc
```

Express the parallelism explicitly with multiple accumulators:

```
// Two accumulators: independent chains, ~1.0 IPC on Zen 1
acc0 = max(acc0, load(data, i))      // independent
acc1 = max(acc1, load(data, i + 4))  // independent
```

Result: 2x throughput from a source-level change, stable across LLVM versions,
no compiler flags or optimizer tuning required.

See `examples/reduction_single.ea` and `examples/reduction_multi_acc.ea`.
