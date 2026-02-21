#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase 8: Integer Types ===

    #[test]
    fn test_i8_literal() {
        assert_output(
            r#"
            func main() {
                let x: i8 = -1
                println(x)
            }
            "#,
            "-1",
        );
    }

    #[test]
    fn test_u8_literal() {
        assert_output(
            r#"
            func main() {
                let x: u8 = 255
                println(x)
            }
            "#,
            "255",
        );
    }

    #[test]
    fn test_i16_literal() {
        assert_output(
            r#"
            func main() {
                let x: i16 = -1000
                println(x)
            }
            "#,
            "-1000",
        );
    }

    #[test]
    fn test_u16_literal() {
        assert_output(
            r#"
            func main() {
                let x: u16 = 60000
                println(x)
            }
            "#,
            "60000",
        );
    }

    #[test]
    fn test_u8_wrapping_add() {
        assert_output(
            r#"
            func main() {
                let x: u8 = 255
                let y: u8 = x + 1
                println(y)
            }
            "#,
            "0",
        );
    }

    #[test]
    fn test_u8_comparison_unsigned() {
        assert_output(
            r#"
            func main() {
                let a: u8 = 200
                let b: u8 = 100
                if a > b {
                    println(1)
                } else {
                    println(0)
                }
            }
            "#,
            "1",
        );
    }

    // Verify u8 comparison is unsigned: 200 > 100 must be true,
    // but if signed (i8), 200 wraps to -56 which is < 100.
    #[test]
    fn test_u8_large_values_unsigned_ordering() {
        assert_output(
            r#"
            func main() {
                let a: u8 = 200
                let b: u8 = 50
                if b < a {
                    println(1)
                } else {
                    println(0)
                }
            }
            "#,
            "1",
        );
    }

    #[test]
    fn test_i8_pointer_c_interop() {
        assert_c_interop(
            r#"
            export func sum_i8(ptr: *i8, n: i32) -> i8 {
                let mut i: i32 = 0
                let mut acc: i8 = 0
                while i < n {
                    acc = acc + ptr[i]
                    i = i + 1
                }
                return acc
            }
            "#,
            r#"
            #include <stdio.h>
            extern signed char sum_i8(signed char* ptr, int n);
            int main() {
                signed char data[4] = {1, 2, 3, 4};
                printf("%d\n", (int)sum_i8(data, 4));
                return 0;
            }
            "#,
            "10",
        );
    }

    #[test]
    fn test_u8_pointer_c_interop() {
        assert_c_interop(
            r#"
            export func scale_bytes(ptr: *mut u8, n: i32, factor: u8) {
                let mut i: i32 = 0
                while i < n {
                    let v: u8 = ptr[i]
                    ptr[i] = v * factor
                    i = i + 1
                }
            }
            "#,
            r#"
            #include <stdio.h>
            extern void scale_bytes(unsigned char* ptr, int n, unsigned char factor);
            int main() {
                unsigned char data[4] = {10, 20, 30, 40};
                scale_bytes(data, 4, 2);
                for (int i = 0; i < 4; i++) printf("%d\n", (int)data[i]);
                return 0;
            }
            "#,
            "20\n40\n60\n80",
        );
    }

    // === Phase 8: SIMD byte vector types ===

    #[test]
    fn test_i8x16_literal_element() {
        assert_output(
            r#"
            func main() {
                let v: i8x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]i8x16
                let x: i8 = v[0]
                println(x)
                let y: i8 = v[15]
                println(y)
            }
            "#,
            "1\n16",
        );
    }

    #[test]
    fn test_i8x16_add_dot() {
        assert_output(
            r#"
            func main() {
                let a: i8x16 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]i8x16
                let b: i8x16 = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]i8x16
                let c: i8x16 = a .+ b
                let x: i8 = c[0]
                println(x)
                let y: i8 = c[15]
                println(y)
            }
            "#,
            "11\n26",
        );
    }

    #[test]
    fn test_i8x32_splat() {
        assert_output(
            r#"
            func main() {
                let s: i8 = 42
                let v: i8x32 = splat(s)
                let x: i8 = v[0]
                println(x)
                let y: i8 = v[31]
                println(y)
            }
            "#,
            "42\n42",
        );
    }

    #[test]
    fn test_u8x16_reduce_max() {
        assert_output(
            r#"
            func main() {
                let v: u8x16 = [10, 200, 30, 150, 5, 180, 20, 100, 15, 90, 25, 60, 50, 40, 70, 80]u8x16
                let m: u8 = reduce_max(v)
                println(m)
            }
            "#,
            "200",
        );
    }

    #[test]
    fn test_u8x16_load_store_c_interop() {
        assert_c_interop(
            r#"
            export func add_constant_u8x16(src: *u8, dst: *mut u8, n: i32, val: u8) {
                let mut i: i32 = 0
                let splat_val: u8x16 = splat(val)
                while i < n {
                    let chunk: u8x16 = load(src, i)
                    let result: u8x16 = chunk .+ splat_val
                    store(dst, i, result)
                    i = i + 16
                }
            }
            "#,
            r#"
            #include <stdio.h>
            #include <string.h>
            extern void add_constant_u8x16(unsigned char* src, unsigned char* dst, int n, unsigned char val);
            int main() {
                unsigned char src[16] = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
                unsigned char dst[16] = {0};
                add_constant_u8x16(src, dst, 16, 10);
                for (int i = 0; i < 4; i++) printf("%d\n", (int)dst[i]);
                return 0;
            }
            "#,
            "11\n12\n13\n14",
        );
    }

    #[test]
    fn test_i8x32_c_interop() {
        assert_c_interop(
            r#"
            export func negate_i8x32(src: *i8, dst: *mut i8, n: i32) {
                let mut i: i32 = 0
                let zero: i8x32 = splat(0)
                while i < n {
                    let chunk: i8x32 = load(src, i)
                    let result: i8x32 = zero .- chunk
                    store(dst, i, result)
                    i = i + 32
                }
            }
            "#,
            r#"
            #include <stdio.h>
            extern void negate_i8x32(signed char* src, signed char* dst, int n);
            int main() {
                signed char src[32];
                signed char dst[32];
                for (int i = 0; i < 32; i++) src[i] = (signed char)(i + 1);
                negate_i8x32(src, dst, 32);
                printf("%d\n", (int)dst[0]);
                printf("%d\n", (int)dst[1]);
                printf("%d\n", (int)dst[31]);
                return 0;
            }
            "#,
            "-1\n-2\n-32",
        );
    }

    // === Phase 8: Widening / Narrowing ===

    #[test]
    fn test_widen_i8_f32x4() {
        assert_output(
            r#"
            func main() {
                let v: i8x16 = [10, 20, -30, 40, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]i8x16
                let w: f32x4 = widen_i8_f32x4(v)
                println(w[0])
                println(w[1])
                println(w[2])
                println(w[3])
            }
            "#,
            "10\n20\n-30\n40",
        );
    }

    #[test]
    fn test_widen_u8_f32x4() {
        assert_output(
            r#"
            func main() {
                let v: u8x16 = [200, 100, 50, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]u8x16
                let w: f32x4 = widen_u8_f32x4(v)
                println(w[0])
                println(w[1])
                println(w[2])
                println(w[3])
            }
            "#,
            "200\n100\n50\n25",
        );
    }

    #[test]
    fn test_narrow_f32x4_i8() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = [10.0, 20.0, -5.0, 127.0]f32x4
                let n: i8x16 = narrow_f32x4_i8(v)
                let a: i8 = n[0]
                let b: i8 = n[1]
                let c: i8 = n[2]
                let d: i8 = n[3]
                println(a)
                println(b)
                println(c)
                println(d)
            }
            "#,
            "10\n20\n-5\n127",
        );
    }

    #[test]
    fn test_roundtrip_u8_f32_u8() {
        assert_c_interop(
            r#"
            export func normalize_first4(src: *u8, dst: *mut f32) {
                let chunk: u8x16 = load(src, 0)
                let floats: f32x4 = widen_u8_f32x4(chunk)
                let scale: f32x4 = splat(0.00392156862)
                let normalized: f32x4 = floats .* scale
                store(dst, 0, normalized)
            }
            "#,
            r#"
            #include <stdio.h>
            #include <math.h>
            extern void normalize_first4(unsigned char* src, float* dst);
            int main() {
                unsigned char src[16] = {0, 128, 255, 64, 0};
                float dst[4] = {0.0f};
                normalize_first4(src, dst);
                printf("%d\n", (int)roundf(dst[0] * 255.0f));
                printf("%d\n", (int)roundf(dst[1] * 255.0f));
                printf("%d\n", (int)roundf(dst[2] * 255.0f));
                printf("%d\n", (int)roundf(dst[3] * 255.0f));
                return 0;
            }
            "#,
            "0\n128\n255\n64",
        );
    }
}
