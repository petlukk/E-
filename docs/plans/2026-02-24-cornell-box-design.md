# Cornell Box Ray Tracer — Design Document

## Goal

Prove Eä can express a non-trivial algorithm beyond streaming SIMD kernels by rendering a classic Cornell Box with direct lighting and single-bounce mirror reflections. No new compiler features required — uses only v0.5.1 capabilities (sqrt, rsqrt, to_f32, unary negation).

## Scope

- **Rendering**: scalar per-pixel ray tracing, no SIMD
- **Scene**: 5 axis-aligned walls (floor, ceiling, back, left=red, right=green), 2 spheres (one diffuse white, one mirror), 1 point light on ceiling
- **Lighting**: Lambert diffuse (`max(0, dot(N, L))`), hard shadows via shadow rays
- **Reflections**: single bounce on mirror sphere only (iterative, not recursive)
- **Output**: flat f32 RGB buffer ([0,1] range), Python converts to PNG
- **Resolution**: 256x256 default, 512x512 for showcase

## Architecture

```
demo/cornell_box/
├── cornell.ea      # Ray tracer kernel (single export + internal helpers)
├── run.py          # Compile, render, save PNG, benchmark
```

### Kernel signature

```ea
export func render(out: *mut f32, width: i32, height: i32)
```

Writes `width * height * 3` floats (RGB, row-major).

### Internal helper functions

- `dot(ax,ay,az, bx,by,bz) -> f32` — 3 muls + 2 adds
- `intersect_sphere(ox,oy,oz, dx,dy,dz, cx,cy,cz, r) -> f32` — quadratic formula, returns t or -1
- `intersect_walls(ox,oy,oz, dx,dy,dz, ...)` — axis-aligned slab tests, writes nearest hit via out-params
- `shade(...)` — Lambert diffuse + shadow ray to light

Multi-return values use `*mut` out-params (caller passes pointer to stack-allocated struct or scalar). Avoids struct-return ABI complications.

### Per-pixel logic

1. Compute ray origin (camera) and direction (perspective projection)
2. Find closest intersection with 5 walls + 2 spheres (7 tests)
3. Diffuse surface: Lambert shading + shadow ray
4. Mirror sphere: compute reflection `R = D - 2*dot(D,N)*N`, trace one bounce, shade secondary hit
5. Write RGB to output buffer

### Python harness

1. Compile `cornell.ea` → `cornell.so`
2. Allocate `numpy.zeros((H, W, 3), float32)`
3. Call `render(ptr, W, H)` via ctypes
4. Clip, scale to 255, save as PNG
5. Benchmark vs pure-Python ray tracer reference

## What we do NOT need

- No new compiler features
- No trigonometry (diffuse uses dot product, not cosine sampling)
- No random numbers (deterministic, no Monte Carlo)
- No BVH (7 objects = brute force)
- No SIMD (scalar per-pixel; this demo proves algorithmic expressiveness, not throughput)

## Success criteria

1. Renders a recognizable Cornell Box image (red/green walls, white floor/ceiling, two spheres, shadows)
2. Mirror sphere shows reflected scene
3. All existing 171 tests still pass
4. Demo runs via `python run.py` like all other demos
