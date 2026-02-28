# Particle Struct Demo

Proves that Ea's `struct` layout matches C exactly over FFI.
This is a **correctness demo**, not a performance demo.

## What it tests

- **Struct field layout**: `ctypes.Structure` and Ea's `struct Particle` must have identical field offsets, sizes, and alignment
- **Field read/write over FFI**: `update_particles` reads position and velocity fields, writes back updated positions
- **Field read over FFI**: `kinetic_energy` reads velocity fields and returns a computed scalar
- **Pointer-to-struct passing**: both functions receive `*Particle` / `*mut Particle` from Python

## The kernel

```
struct Particle {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
}

export func update_particles(particles: *mut Particle, count: i32, dt: f32)
export func kinetic_energy(particles: *Particle, count: i32) -> f32
```

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Build shared library
bash demo/particles/build.sh

# Run demo
python demo/particles/run.py
```

## How it works

Python allocates an array of `Particle` structs via `ctypes.Structure`, passes
a pointer to the Ea shared library, then verifies the results match a Python
reference computation. If any field offset or alignment were wrong, the
positions and energies would be garbage.

The performance section compares Ea (AoS structs) against NumPy (SoA arrays).
100,000 particles, median of 50 runs:

```
Ea (AoS struct)  :    0.233 ms
NumPy (SoA)      :    0.061 ms
```

NumPy is 3.8x faster (SoA vs AoS layout difference, expected).
The point is correctness, not speed.
