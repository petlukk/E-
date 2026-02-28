# Cornell Box Ray Tracer — Eä Demo

A complete ray tracer written in Eä. Renders the classic Cornell Box scene:
5 colored walls, 2 spheres (one diffuse, one mirror), point light with
hard shadows, and single-bounce mirror reflection.

First non-SIMD demo — proves Eä handles scalar math, struct returns,
and recursion at native speed.

## Results

512x512 resolution (262,144 pixels). 10 runs, median time.

```
Eä render     : 16.5 ms
Rays/sec      : 15,853,039
```

## How to run

```bash
# Build compiler (once)
cargo build --features=llvm --release

# Render (default 512x512)
python demo/cornell_box/run.py

# Custom resolution
python demo/cornell_box/run.py 256 256
```

Saves `output.png` and prints timing + rays/sec.

## The kernel

245 lines of Eä. Nothing is hidden.

The kernel implements:
- **Vec3 math** — add, sub, scale, dot, normalize, reflect
- **Ray-sphere intersection** — quadratic discriminant with epsilon
- **Ray-plane intersection** — 5 bounded walls
- **Closest hit search** — finds nearest surface by distance
- **Shadow rays** — hard shadows from point light
- **Mirror reflection** — depth-limited recursive trace

```
export func render(out: *mut f32, width: i32, height: i32) {
    let cam: Vec3 = v3(0.5, 0.5, -1.0)
    let fh: f32 = to_f32(height)

    let mut py: i32 = 0
    while py < height {
        let mut px: i32 = 0
        while px < width {
            let u: f32 = (to_f32(px) + 0.5 - to_f32(width) * 0.5) / fh
            let v: f32 = -(to_f32(py) + 0.5 - fh * 0.5) / fh
            let rd: Vec3 = v3_normalize(v3(u, v, 1.0))

            let color: Vec3 = trace(cam, rd, 0)

            let idx: i32 = (py * width + px) * 3
            out[idx] = color.x
            out[idx + 1] = color.y
            out[idx + 2] = color.z

            px = px + 1
        }
        py = py + 1
    }
}
```

## How it works

Python handles image I/O and benchmarking. Eä does the compute.
The kernel compiles to a `.so` and is called via `ctypes`.

```bash
ea cornell.ea --lib   # → cornell.so
```

```python
lib = ctypes.CDLL("./cornell.so")
lib.render(buffer_ptr, width, height)
```
