#[cfg(feature = "llvm")]
mod common;

#[cfg(feature = "llvm")]
mod tests {
    use super::common::*;

    // === Phase 3: Control Flow tests ===

    #[test]
    fn test_if_true_branch() {
        assert_output_lines(
            r#"
            func main() {
                let x: i32 = 10
                if x > 5 { println(1) }
                println(0)
            }
        "#,
            &["1", "0"],
        );
    }

    #[test]
    fn test_if_false_branch() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 3
                if x > 5 { println(1) }
                println(0)
            }
        "#,
            "0",
        );
    }

    #[test]
    fn test_if_else() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 3
                if x > 5 { println(1) } else { println(2) }
            }
        "#,
            "2",
        );
    }

    #[test]
    fn test_comparisons() {
        assert_output_lines(
            r#"
            func check(a: i32, b: i32) -> i32 {
                if a < b { return 1 }
                return 0
            }
            func main() {
                println(check(1, 2))
                println(check(2, 1))
            }
        "#,
            &["1", "0"],
        );
    }

    #[test]
    fn test_less_equal() {
        assert_output_lines(
            r#"
            func check(a: i32, b: i32) -> i32 {
                if a <= b { return 1 }
                return 0
            }
            func main() {
                println(check(3, 3))
                println(check(4, 3))
            }
        "#,
            &["1", "0"],
        );
    }

    #[test]
    fn test_equality() {
        assert_output_lines(
            r#"
            func check(a: i32, b: i32) -> i32 {
                if a == b { return 1 }
                if a != b { return 2 }
                return 0
            }
            func main() {
                println(check(5, 5))
                println(check(5, 6))
            }
        "#,
            &["1", "2"],
        );
    }

    #[test]
    fn test_while_loop() {
        assert_output(
            r#"
            func main() {
                let mut i: i32 = 0
                while i < 5 { i = i + 1 }
                println(i)
            }
        "#,
            "5",
        );
    }

    #[test]
    fn test_sum_range() {
        assert_output(
            r#"
            func main() {
                let mut sum: i32 = 0
                let mut i: i32 = 1
                while i <= 10 {
                    sum = sum + i
                    i = i + 1
                }
                println(sum)
            }
        "#,
            "55",
        );
    }

    #[test]
    fn test_logical_and() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 5
                if x > 0 && x < 10 { println(1) } else { println(0) }
            }
        "#,
            "1",
        );
    }

    #[test]
    fn test_logical_or() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 15
                if x < 0 || x > 10 { println(1) } else { println(0) }
            }
        "#,
            "1",
        );
    }

    #[test]
    fn test_not_operator() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 5
                if !(x > 10) { println(1) } else { println(0) }
            }
        "#,
            "1",
        );
    }

    #[test]
    fn test_nested_if_in_while() {
        assert_output(
            r#"
            func main() {
                let mut i: i32 = 0
                let mut evens: i32 = 0
                while i < 10 {
                    if i % 2 == 0 { evens = evens + 1 }
                    i = i + 1
                }
                println(evens)
            }
        "#,
            "5",
        );
    }

    #[test]
    fn test_early_return_from_if() {
        assert_output_lines(
            r#"
            func abs(x: i32) -> i32 {
                if x < 0 { return 0 - x }
                return x
            }
            func main() {
                println(abs(-5))
                println(abs(3))
            }
        "#,
            &["5", "3"],
        );
    }

    #[test]
    fn test_fizzbuzz_style() {
        assert_output_lines(
            r#"
            func classify(n: i32) -> i32 {
                if n % 3 == 0 {
                    return 3
                } else {
                    if n % 5 == 0 { return 5 } else { return n }
                }
            }
            func main() {
                println(classify(9))
                println(classify(10))
                println(classify(7))
            }
        "#,
            &["3", "5", "7"],
        );
    }

    // === Phase 3: C interop tests ===

    #[test]
    fn test_export_sum_range_c_interop() {
        assert_c_interop(
            r#"
            export func sum_range(start: i32, end: i32) -> i32 {
                let mut sum: i32 = 0
                let mut i: i32 = start
                while i <= end {
                    sum = sum + i
                    i = i + 1
                }
                return sum
            }
        "#,
            r#"#include <stdio.h>
            extern int sum_range(int, int);
            int main() { printf("%d\n", sum_range(1, 100)); return 0; }"#,
            "5050",
        );
    }

    #[test]
    fn test_export_max_c_interop() {
        assert_c_interop(
            r#"
            export func max(a: i32, b: i32) -> i32 {
                if a > b { return a }
                return b
            }
        "#,
            r#"#include <stdio.h>
            extern int max(int, int);
            int main() { printf("%d\n", max(10, 20)); printf("%d\n", max(30, 5)); return 0; }"#,
            "20\n30",
        );
    }

    #[test]
    fn test_export_clamp_c_interop() {
        assert_c_interop(
            r#"
            export func clamp(val: i32, lo: i32, hi: i32) -> i32 {
                if val < lo { return lo }
                if val > hi { return hi }
                return val
            }
        "#,
            r#"#include <stdio.h>
            extern int clamp(int, int, int);
            int main() { printf("%d\n", clamp(-5, 0, 10)); printf("%d\n", clamp(15, 0, 10)); printf("%d\n", clamp(5, 0, 10)); return 0; }"#,
            "0\n10\n5",
        );
    }

    // === else if ===

    #[test]
    fn test_else_if_basic() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 2
                if x == 1 {
                    println(10)
                } else if x == 2 {
                    println(20)
                } else {
                    println(30)
                }
            }
            "#,
            "20",
        );
    }

    #[test]
    fn test_else_if_chain() {
        assert_output(
            r#"
            func main() {
                let x: i32 = 4
                if x == 1 {
                    println(10)
                } else if x == 2 {
                    println(20)
                } else if x == 3 {
                    println(30)
                } else if x == 4 {
                    println(40)
                } else {
                    println(50)
                }
            }
            "#,
            "40",
        );
    }

    #[test]
    fn test_else_if_c_interop() {
        assert_c_interop(
            r#"
export func classify(x: i32) -> i32 {
    if x < 0 {
        return -1
    } else if x == 0 {
        return 0
    } else {
        return 1
    }
}
"#,
            r#"
#include <stdio.h>

extern int classify(int);

int main() {
    printf("%d %d %d\n", classify(-5), classify(0), classify(7));
    return 0;
}
"#,
            "-1 0 1",
        );
    }

    // === Short-circuit evaluation ===

    #[test]
    fn test_short_circuit_and_skips_right() {
        // false && side_effect(): must NOT execute right side
        assert_c_interop(
            r#"
export func test_and(data: *mut i32, len: i32, i: i32) -> i32 {
    if i < len && data[i] > 0 {
        return 1
    }
    return 0
}
"#,
            r#"
#include <stdio.h>
extern int test_and(int*, int, int);
int main() {
    int data[] = {10, 20, 30};
    printf("%d\n", test_and(data, 3, 1));
    printf("%d\n", test_and(data, 3, 5));
    return 0;
}
"#,
            "1\n0",
        );
    }

    #[test]
    fn test_short_circuit_or_skips_right() {
        // true || side_effect(): must NOT execute right side
        assert_c_interop(
            r#"
export func test_or(data: *mut i32, len: i32, i: i32) -> i32 {
    if i >= len || data[i] == 0 {
        return 1
    }
    return 0
}
"#,
            r#"
#include <stdio.h>
extern int test_or(int*, int, int);
int main() {
    int data[] = {10, 0, 30};
    printf("%d\n", test_or(data, 3, 5));
    printf("%d\n", test_or(data, 3, 1));
    printf("%d\n", test_or(data, 3, 0));
    return 0;
}
"#,
            "1\n1\n0",
        );
    }

    #[test]
    fn test_short_circuit_and_no_mutation() {
        // false && mutating_call(): mutation must NOT happen
        assert_c_interop(
            r#"
func set_flag(out: *mut i32) -> bool {
    out[0] = 999
    return true
}

export func test(out: *mut i32) {
    out[0] = 0
    let cond: bool = false
    if cond && set_flag(out) {
        out[0] = 1
    }
}
"#,
            r#"
#include <stdio.h>
extern void test(int*);
int main() {
    int flag = 42;
    test(&flag);
    printf("%d\n", flag);
    return 0;
}
"#,
            "0",
        );
    }

    #[test]
    fn test_short_circuit_or_no_mutation() {
        // true || mutating_call(): mutation must NOT happen
        assert_c_interop(
            r#"
func set_flag(out: *mut i32) -> bool {
    out[0] = 999
    return true
}

export func test(out: *mut i32) {
    out[0] = 0
    let cond: bool = true
    if cond || set_flag(out) {
        out[0] = 1
    }
}
"#,
            r#"
#include <stdio.h>
extern void test(int*);
int main() {
    int flag = 42;
    test(&flag);
    printf("%d\n", flag);
    return 0;
}
"#,
            "1",
        );
    }

    #[test]
    fn test_and_or_nested() {
        assert_output(
            r#"
            func main() {
                let a: i32 = 5
                let b: i32 = 3
                let c: i32 = 5
                if a > 0 && (b < 10 || c == 5) {
                    println(1)
                } else {
                    println(0)
                }
            }
            "#,
            "1",
        );
    }
}
