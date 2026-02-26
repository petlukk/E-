# Particle Life — Eä Demo

N colored particles interact via asymmetric attraction/repulsion forces.
A random interaction matrix defines how each type affects every other type.
Emergent structures form from simple math running live from a single Eä kernel.

## Results

AMD Ryzen 7 1700 (Zen 1, AVX2). 50 runs, median time.

```
            N=1000    N=2000    N=4000    N=8000
Eä         1.23 ms   4.32 ms  16.75 ms  140.75 ms
C (-O2)    1.21 ms   4.54 ms  18.87 ms  158.01 ms
```

Eä matches hand-written C compiled with `clang-18 -O2`. Both go through the
same LLVM-18 backend — Eä adds zero overhead.

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Run with pygame UI
python demo/particle_life/run.py --particles 2000

# Headless benchmark (no pygame required)
python demo/particle_life/run.py  # falls back if pygame missing
```

## Controls

Keyboard or click the bottom panel buttons:

| Key | Action |
|-----|--------|
| F | Toggle fused / unfused kernel |
| R | Rerandomize interaction matrix |
| +/- | Increase / decrease particle count by 500 |
| ESC | Quit |

## The kernel

This is the entire Eä implementation. Nothing is hidden.

```
export func particle_life_step(
    px: *mut f32, py: *mut f32,
    vx: *mut f32, vy: *mut f32,
    types: *i32,
    matrix: *f32,
    n: i32, num_types: i32,
    r_max: f32, dt: f32, friction: f32, size: f32
) {
    let r_max2: f32 = r_max * r_max
    let mut i: i32 = 0
    while i < n {
        let xi: f32 = px[i]
        let yi: f32 = py[i]
        let ti: i32 = types[i]
        let mut fx: f32 = 0.0
        let mut fy: f32 = 0.0

        let mut j: i32 = 0
        while j < n {
            let dx: f32 = px[j] - xi
            let dy: f32 = py[j] - yi
            let dist2: f32 = dx * dx + dy * dy
            if dist2 > 0.0 && dist2 < r_max2 {
                let dist: f32 = sqrt(dist2)
                let strength: f32 = matrix[ti * num_types + types[j]]
                let force: f32 = strength * (1.0 - dist / r_max)
                fx = fx + force * dx / dist
                fy = fy + force * dy / dist
            }
            j = j + 1
        }

        vx[i] = (vx[i] + fx * dt) * friction
        vy[i] = (vy[i] + fy * dt) * friction
        px[i] = px[i] + vx[i]
        py[i] = py[i] + vy[i]

        let cur_px: f32 = px[i]
        let cur_py: f32 = py[i]
        if cur_px < 0.0 { px[i] = cur_px + size }
        if cur_px >= size { px[i] = cur_px - size }
        if cur_py < 0.0 { py[i] = cur_py + size }
        if cur_py >= size { py[i] = cur_py - size }

        i = i + 1
    }
}
```

## How it works

Python handles rendering (pygame) and state management. Eä does the compute.
The kernel compiles to a `.so` and is called via `ctypes` — no runtime,
no framework, no bindings library.

```bash
ea particle_life.ea --lib   # → particle_life.so
```

```python
lib = ctypes.CDLL("./particle_life.so")
lib.particle_life_step(px, py, vx, vy, types, matrix, n, num_types,
                       r_max, dt, friction, size)
```

The O(N²) force loop dominates — every particle checks every other particle.
At 4K particles the kernel stays under 17ms per frame (60fps). At 8K it drops
to ~7fps, limited by the N² algorithm, not by Eä overhead.
