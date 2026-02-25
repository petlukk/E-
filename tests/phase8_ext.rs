#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Type conversions ===

    #[test]
    fn test_to_f32_from_i32() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 42
                let y: f32 = to_f32(x)
                println(y)
            }
            "#,
            "42",
        );
    }

    #[test]
    fn test_to_f64_from_i32() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 100
                let y: f64 = to_f64(x)
                println(y)
            }
            "#,
            "100",
        );
    }

    #[test]
    fn test_to_i32_from_f32() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 3.7
                let y: i32 = to_i32(x)
                println(y)
            }
            "#,
            "3",
        );
    }

    #[test]
    fn test_to_i32_from_f64() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 9.9
                let y: i32 = to_i32(x)
                println(y)
            }
            "#,
            "9",
        );
    }

    #[test]
    fn test_to_f32_from_i64() {
        assert_output(
            r#"
            func main() {
                let x: i64 = 1000
                let y: f32 = to_f32(x)
                println(y)
            }
            "#,
            "1000",
        );
    }

    #[test]
    fn test_to_i64_from_f64() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 123.456
                let y: i64 = to_i64(x)
                println(y)
            }
            "#,
            "123",
        );
    }

    #[test]
    fn test_to_f32_in_expression() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 50
                let width: i32 = 100
                let u: f32 = to_f32(x) / to_f32(width)
                println(u)
            }
            "#,
            "0.5",
        );
    }

    #[test]
    fn test_to_f64_from_f32() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 2.5
                let y: f64 = to_f64(x)
                println(y)
            }
            "#,
            "2.5",
        );
    }

    #[test]
    fn test_to_f32_from_f64() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 1.5
                let y: f32 = to_f32(x)
                println(y)
            }
            "#,
            "1.5",
        );
    }

    // === Combined: the spec example ===

    #[test]
    fn test_spec_magnitude_example() {
        // From EA_V2_SPECIFICATION.md Part 6:
        //   export func magnitude(v: *Vec2) -> f32 {
        //       return sqrt(v.x * v.x + v.y * v.y);
        //   }
        assert_c_interop(
            r#"
            struct Vec2 { x: f32, y: f32 }

            export func magnitude(v: *Vec2) -> f32 {
                return sqrt(v.x * v.x + v.y * v.y)
            }
            "#,
            r#"
                #include <stdio.h>
                typedef struct { float x; float y; } Vec2;
                extern float magnitude(const Vec2*);
                int main() {
                    Vec2 v = {3.0f, 4.0f};
                    printf("%g\n", magnitude(&v));
                    return 0;
                }
            "#,
            "5",
        );
    }

    #[test]
    fn test_combined_sqrt_cast_negate() {
        // Use all three features together
        assert_output(
            r#"
            func main() {
                let ix: i32 = 3
                let iy: i32 = -4
                let fx: f32 = to_f32(ix)
                let fy: f32 = to_f32(iy)
                let dist: f32 = sqrt(fx * fx + fy * fy)
                println(dist)
            }
            "#,
            "5",
        );
    }

    // === Cornell Box ===

    #[test]
    fn test_cornell_box_compiles_and_renders() {
        assert_c_interop(
            include_str!("../demo/cornell_box/cornell.ea"),
            r#"
            #include <stdio.h>
            extern void render(float*, int, int);
            int main() {
                float buf[16 * 16 * 3];
                render(buf, 16, 16);
                int center = (8 * 16 + 8) * 3;
                float r = buf[center];
                float g = buf[center + 1];
                float b = buf[center + 2];
                printf("%d\n", (r > 0.01f || g > 0.01f || b > 0.01f) ? 1 : 0);
                return 0;
            }
            "#,
            "1",
        );
    }

    // === Particle Life fused kernel ===

    #[test]
    fn test_particle_life_fused_step() {
        let result = compile_and_link_with_c(
            r#"
export func particle_life_step(
    px: *mut f32, py: *mut f32,
    vx: *mut f32, vy: *mut f32,
    types: *i32,
    matrix: *f32,
    n: i32, num_types: i32,
    r_max: f32, dt: f32, friction: f32, size: f32
) {
    let mut i: i32 = 0
    while i < n {
        let xi: f32 = px[i]
        let yi: f32 = py[i]
        let ti: i32 = types[i]
        let mut fx: f32 = 0.0
        let mut fy: f32 = 0.0

        let r_max2: f32 = r_max * r_max
        let mut j: i32 = 0
        while j < n {
            let dx: f32 = px[j] - xi
            let dy: f32 = py[j] - yi
            let dist2: f32 = dx * dx + dy * dy
            if dist2 > 0.0 {
                if dist2 < r_max2 {
                    let dist: f32 = sqrt(dist2)
                    let strength: f32 = matrix[ti * num_types + types[j]]
                    let force: f32 = strength * (1.0 - dist / r_max)
                    fx = fx + force * dx / dist
                    fy = fy + force * dy / dist
                }
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
"#,
            r#"
#include <stdio.h>
#include <math.h>

extern void particle_life_step(
    float* px, float* py,
    float* vx, float* vy,
    int* types,
    float* matrix,
    int n, int num_types,
    float r_max, float dt, float friction, float size
);

int main() {
    float px[2] = {0.0f, 50.0f};
    float py[2] = {0.0f, 0.0f};
    float vx[2] = {0.0f, 0.0f};
    float vy[2] = {0.0f, 0.0f};
    int types[2] = {0, 0};
    float matrix[1] = {1.0f};

    particle_life_step(px, py, vx, vy, types, matrix,
                       2, 1, 100.0f, 1.0f, 0.5f, 1000.0f);

    /* Particle 0 expected:
       dx=50, dy=0, dist2=2500, r_max2=10000 -> in range
       dist=50, strength=1.0, force=1.0*(1-50/100)=0.5
       fx=0.5*50/50=0.5, fy=0
       vx=(0+0.5*1.0)*0.5=0.25, vy=(0+0)*0.5=0
       px=0+0.25*1.0=0.25, py=0+0*1.0=0 */
    int ok = 1;
    if (fabsf(px[0] - 0.25f) > 0.001f) ok = 0;
    if (fabsf(py[0] - 0.0f) > 0.001f) ok = 0;
    if (fabsf(vx[0] - 0.25f) > 0.001f) ok = 0;
    if (fabsf(vy[0] - 0.0f) > 0.001f) ok = 0;
    printf("%d\n", ok);
    return 0;
}
"#,
        );
        assert_eq!(result.stdout.trim(), "1");
    }

    // === Particle Life unfused kernels ===

    #[test]
    fn test_particle_life_unfused_step() {
        let result = compile_and_link_with_c(
            r#"
export func compute_forces(
    px: *f32, py: *f32,
    types: *i32,
    matrix: *f32,
    fx: *mut f32, fy: *mut f32,
    n: i32, num_types: i32,
    r_max: f32
) {
    let r_max2: f32 = r_max * r_max
    let mut i: i32 = 0
    while i < n {
        let xi: f32 = px[i]
        let yi: f32 = py[i]
        let ti: i32 = types[i]
        let mut sum_fx: f32 = 0.0
        let mut sum_fy: f32 = 0.0

        let mut j: i32 = 0
        while j < n {
            let dx: f32 = px[j] - xi
            let dy: f32 = py[j] - yi
            let dist2: f32 = dx * dx + dy * dy
            if dist2 > 0.0 {
                if dist2 < r_max2 {
                    let dist: f32 = sqrt(dist2)
                    let strength: f32 = matrix[ti * num_types + types[j]]
                    let force: f32 = strength * (1.0 - dist / r_max)
                    sum_fx = sum_fx + force * dx / dist
                    sum_fy = sum_fy + force * dy / dist
                }
            }
            j = j + 1
        }

        fx[i] = sum_fx
        fy[i] = sum_fy
        i = i + 1
    }
}

export func update_velocities(
    vx: *mut f32, vy: *mut f32,
    fx: *f32, fy: *f32,
    n: i32, dt: f32, friction: f32
) {
    let mut i: i32 = 0
    while i < n {
        vx[i] = (vx[i] + fx[i] * dt) * friction
        vy[i] = (vy[i] + fy[i] * dt) * friction
        i = i + 1
    }
}

export func update_positions(
    px: *mut f32, py: *mut f32,
    vx: *f32, vy: *f32,
    n: i32, dt: f32, size: f32
) {
    let mut i: i32 = 0
    while i < n {
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
"#,
            r#"
#include <stdio.h>
#include <math.h>

extern void compute_forces(
    float* px, float* py,
    int* types,
    float* matrix,
    float* fx, float* fy,
    int n, int num_types,
    float r_max
);

extern void update_velocities(
    float* vx, float* vy,
    float* fx, float* fy,
    int n, float dt, float friction
);

extern void update_positions(
    float* px, float* py,
    float* vx, float* vy,
    int n, float dt, float size
);

int main() {
    float px[2] = {0.0f, 50.0f};
    float py[2] = {0.0f, 0.0f};
    float vx[2] = {0.0f, 0.0f};
    float vy[2] = {0.0f, 0.0f};
    int types[2] = {0, 0};
    float matrix[1] = {1.0f};
    float fx[2] = {0.0f, 0.0f};
    float fy[2] = {0.0f, 0.0f};

    compute_forces(px, py, types, matrix, fx, fy,
                   2, 1, 100.0f);
    update_velocities(vx, vy, fx, fy,
                      2, 1.0f, 0.5f);
    update_positions(px, py, vx, vy,
                     2, 1.0f, 1000.0f);

    /* Both particles see original positions (unfused):
       Particle 0: dx=50, dist=50, strength=1.0, force=0.5, fx=0.5
       Particle 1: dx=-50, dist=50, strength=1.0, force=0.5, fx=-0.5
       After update_velocities: vx[0]=0.25, vx[1]=-0.25
       After update_positions: px[0]=0.25, px[1]=49.75 */
    int ok = 1;
    if (fabsf(px[0] - 0.25f) > 0.001f) ok = 0;
    if (fabsf(px[1] - 49.75f) > 0.001f) ok = 0;
    if (fabsf(vx[0] - 0.25f) > 0.001f) ok = 0;
    if (fabsf(vx[1] - (-0.25f)) > 0.001f) ok = 0;
    if (fabsf(vy[0]) > 0.001f) ok = 0;
    if (fabsf(vy[1]) > 0.001f) ok = 0;
    if (fabsf(py[0]) > 0.001f) ok = 0;
    if (fabsf(py[1]) > 0.001f) ok = 0;

    if (ok) {
        printf("PASS\n");
    } else {
        printf("FAIL px[0]=%f px[1]=%f vx[0]=%f vx[1]=%f\n",
               px[0], px[1], vx[0], vx[1]);
    }
    return 0;
}
"#,
        );
        assert_eq!(result.stdout.trim(), "PASS");
    }
}
