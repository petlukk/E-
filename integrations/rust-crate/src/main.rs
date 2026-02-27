extern "C" {
    fn dot_product(a: *const f32, b: *const f32, n: i32) -> f32;
}

fn main() {
    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let b = vec![2.0f32, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0];

    let result = unsafe { dot_product(a.as_ptr(), b.as_ptr(), a.len() as i32) };
    println!("dot product: {result}"); // 72.0

    let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    assert!(
        (result - expected).abs() < 1e-6,
        "mismatch: got {result}, expected {expected}"
    );
    println!("verified: matches Rust computation");
}
