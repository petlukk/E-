#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Vector Bitwise Operations (.&, .|, .^) ===

    #[test]
    fn test_u8x16_and_dot() {
        assert_c_interop(
            r#"
export func test_and(a: *u8, b: *u8, out: *mut u8, len: i32) {
    let mut i: i32 = 0
    while i + 16 <= len {
        let va: u8x16 = load(a, i)
        let vb: u8x16 = load(b, i)
        store(out, i, va .& vb)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_and(const uint8_t*, const uint8_t*, uint8_t*, int);

int main() {
    uint8_t a[16] = {0xFF, 0x0F, 0xAA, 0x55, 1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t b[16] = {0x0F, 0xFF, 0x55, 0xAA, 255,254,253,252,251,250,249,248,247,246,245,244};
    uint8_t out[16] = {0};
    test_and(a, b, out, 16);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "15 15 0 0",
        );
    }

    #[test]
    fn test_u8x16_or_dot() {
        assert_c_interop(
            r#"
export func test_or(a: *u8, b: *u8, out: *mut u8, len: i32) {
    let mut i: i32 = 0
    while i + 16 <= len {
        let va: u8x16 = load(a, i)
        let vb: u8x16 = load(b, i)
        store(out, i, va .| vb)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_or(const uint8_t*, const uint8_t*, uint8_t*, int);

int main() {
    uint8_t a[16] = {0xF0, 0x0F, 0xAA, 0x00, 1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t b[16] = {0x0F, 0xF0, 0x55, 0xFF, 255,254,253,252,251,250,249,248,247,246,245,244};
    uint8_t out[16] = {0};
    test_or(a, b, out, 16);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "255 255 255 255",
        );
    }

    #[test]
    fn test_u8x16_xor_dot() {
        assert_c_interop(
            r#"
export func test_xor(a: *u8, b: *u8, out: *mut u8, len: i32) {
    let mut i: i32 = 0
    while i + 16 <= len {
        let va: u8x16 = load(a, i)
        let vb: u8x16 = load(b, i)
        store(out, i, va .^ vb)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_xor(const uint8_t*, const uint8_t*, uint8_t*, int);

int main() {
    uint8_t a[16] = {0xFF, 0xFF, 0xAA, 0x00, 1,2,3,4,5,6,7,8,9,10,11,12};
    uint8_t b[16] = {0xFF, 0x00, 0x55, 0xFF, 255,254,253,252,251,250,249,248,247,246,245,244};
    uint8_t out[16] = {0};
    test_xor(a, b, out, 16);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "0 255 255 255",
        );
    }

    #[test]
    fn test_i32x4_and_dot() {
        assert_c_interop(
            r#"
export func test_and_i32(a: *i32, b: *i32, out: *mut i32, len: i32) {
    let mut i: i32 = 0
    while i + 4 <= len {
        let va: i32x4 = load(a, i)
        let vb: i32x4 = load(b, i)
        store(out, i, va .& vb)
        i = i + 4
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_and_i32(const int32_t*, const int32_t*, int32_t*, int);

int main() {
    int32_t a[4] = {0x0F0F0F0F, -1, 0, 0x12345678};
    int32_t b[4] = {(int32_t)0xF0F0F0F0, 0x7FFFFFFF, -1, 0x0F0F0F0F};
    int32_t out[4] = {0};
    test_and_i32(a, b, out, 4);
    printf("%d %d\n", out[0], out[1]);
    return 0;
}
"#,
            "0 2147483647",
        );
    }

    #[test]
    fn test_i16x8_or_dot() {
        assert_c_interop(
            r#"
export func test_or_i16(a: *i16, b: *i16, out: *mut i16, len: i32) {
    let mut i: i32 = 0
    while i + 8 <= len {
        let va: i16x8 = load(a, i)
        let vb: i16x8 = load(b, i)
        store(out, i, va .| vb)
        i = i + 8
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void test_or_i16(const int16_t*, const int16_t*, int16_t*, int);

int main() {
    int16_t a[8] = {0x00F0, 0x0F00, 0, 0, 0, 0, 0, 0};
    int16_t b[8] = {0x000F, 0x00F0, 0, 0, 0, 0, 0, 0};
    int16_t out[8] = {0};
    test_or_i16(a, b, out, 8);
    printf("%d %d\n", out[0], out[1]);
    return 0;
}
"#,
            "255 4080",
        );
    }

    // === Hex and Binary Literals ===

    #[test]
    fn test_hex_literal() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 0xFF
                println(x)
            }
            "#,
            "255",
        );
    }

    #[test]
    fn test_binary_literal() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 0b11110000
                println(x)
            }
            "#,
            "240",
        );
    }

    #[test]
    fn test_hex_u8_splat() {
        assert_c_interop(
            r#"
export func mask_and(data: *u8, out: *mut u8, len: i32) {
    let mask: u8x16 = splat(0x0F)
    let mut i: i32 = 0
    while i + 16 <= len {
        let v: u8x16 = load(data, i)
        store(out, i, v .& mask)
        i = i + 16
    }
}
"#,
            r#"
#include <stdio.h>
#include <stdint.h>

extern void mask_and(const uint8_t*, uint8_t*, int);

int main() {
    uint8_t data[16] = {0xFF, 0xAB, 0x12, 0x00, 0,0,0,0,0,0,0,0,0,0,0,0};
    uint8_t out[16] = {0};
    mask_and(data, out, 16);
    printf("%d %d %d %d\n", out[0], out[1], out[2], out[3]);
    return 0;
}
"#,
            "15 11 2 0",
        );
    }

    #[test]
    fn test_negative_hex_literal() {
        assert_output(
            r#"
            func main() {
                let x: i32 = -0x01
                println(x)
            }
            "#,
            "-1",
        );
    }

    #[test]
    fn test_binary_bitmask() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 0b10101010
                println(x)
            }
            "#,
            "170",
        );
    }

    // === Unary negation ===

    #[test]
    fn test_negate_i32_variable() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 42
                let y: i32 = -x
                println(y)
            }
            "#,
            "-42",
        );
    }

    #[test]
    fn test_negate_f32_variable() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 3.5
                let y: f32 = -x
                println(y)
            }
            "#,
            "-3.5",
        );
    }

    #[test]
    fn test_negate_f64_variable() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 2.25
                let y: f64 = -x
                println(y)
            }
            "#,
            "-2.25",
        );
    }

    #[test]
    fn test_negate_in_expression() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 10
                let y: i32 = 5 + -x
                println(y)
            }
            "#,
            "-5",
        );
    }

    #[test]
    fn test_double_negate() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 7
                let y: i32 = - -x
                println(y)
            }
            "#,
            "7",
        );
    }

    // === sqrt / rsqrt ===

    #[test]
    fn test_sqrt_f32_scalar() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 4.0
                let y: f32 = sqrt(x)
                println(y)
            }
            "#,
            "2",
        );
    }

    #[test]
    fn test_sqrt_f64_scalar() {
        assert_output(
            r#"
            func main() {
                let x: f64 = 9.0
                let y: f64 = sqrt(x)
                println(y)
            }
            "#,
            "3",
        );
    }

    #[test]
    fn test_sqrt_f32x4_vector() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = splat(16.0)
                let r: f32x4 = sqrt(v)
                println(r[0])
            }
            "#,
            "4",
        );
    }

    #[test]
    fn test_rsqrt_f32_scalar() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 4.0
                let y: f32 = rsqrt(x)
                println(y)
            }
            "#,
            "0.5",
        );
    }

    #[test]
    fn test_rsqrt_f32x4_vector() {
        assert_output(
            r#"
            func main() {
                let v: f32x4 = splat(4.0)
                let r: f32x4 = rsqrt(v)
                println(r[0])
            }
            "#,
            "0.5",
        );
    }

    #[test]
    fn test_sqrt_in_magnitude() {
        assert_output(
            r#"
            func main() {
                let x: f32 = 3.0
                let y: f32 = 4.0
                let mag: f32 = sqrt(x * x + y * y)
                println(mag)
            }
            "#,
            "5",
        );
    }
}
