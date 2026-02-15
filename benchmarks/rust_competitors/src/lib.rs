#![feature(portable_simd)]

use std::simd::prelude::*;
use std::simd::StdFloat;

// ---------------------------------------------------------------------------
// FMA kernel: result[i] = a[i] * b[i] + c[i]
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn fma_kernel_f32x4_rust(
    a: *const f32,
    b: *const f32,
    c: *const f32,
    result: *mut f32,
    len: i32,
) {
    let len = len as usize;
    let a = unsafe { std::slice::from_raw_parts(a, len) };
    let b = unsafe { std::slice::from_raw_parts(b, len) };
    let c = unsafe { std::slice::from_raw_parts(c, len) };
    let result = unsafe { std::slice::from_raw_parts_mut(result, len) };

    let chunks = len / 4;
    for i in 0..chunks {
        let off = i * 4;
        let va = f32x4::from_slice(&a[off..off + 4]);
        let vb = f32x4::from_slice(&b[off..off + 4]);
        let vc = f32x4::from_slice(&c[off..off + 4]);
        let vr = va.mul_add(vb, vc);
        vr.copy_to_slice(&mut result[off..off + 4]);
    }
    // Remainder
    for i in (chunks * 4)..len {
        result[i] = a[i].mul_add(b[i], c[i]);
    }
}

// ---------------------------------------------------------------------------
// Sum reduction
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn sum_f32x4_rust(data: *const f32, len: i32) -> f32 {
    let len = len as usize;
    let data = unsafe { std::slice::from_raw_parts(data, len) };

    let mut acc = f32x4::splat(0.0);
    let chunks = len / 4;
    for i in 0..chunks {
        let off = i * 4;
        let v = f32x4::from_slice(&data[off..off + 4]);
        acc += v;
    }
    let mut total = acc.reduce_sum();
    for i in (chunks * 4)..len {
        total += data[i];
    }
    total
}

#[no_mangle]
pub extern "C" fn sum_f32x8_rust(data: *const f32, len: i32) -> f32 {
    let len = len as usize;
    let data = unsafe { std::slice::from_raw_parts(data, len) };

    let mut acc = f32x8::splat(0.0);
    let chunks = len / 8;
    for i in 0..chunks {
        let off = i * 8;
        let v = f32x8::from_slice(&data[off..off + 8]);
        acc += v;
    }
    let mut total = acc.reduce_sum();
    for i in (chunks * 8)..len {
        total += data[i];
    }
    total
}

// ---------------------------------------------------------------------------
// Max reduction
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn max_f32x4_rust(data: *const f32, len: i32) -> f32 {
    let len = len as usize;
    let data = unsafe { std::slice::from_raw_parts(data, len) };

    if len == 0 {
        return f32::NEG_INFINITY;
    }

    let mut acc = f32x4::splat(f32::NEG_INFINITY);
    let chunks = len / 4;
    for i in 0..chunks {
        let off = i * 4;
        let v = f32x4::from_slice(&data[off..off + 4]);
        acc = acc.simd_max(v);
    }
    let mut result = acc.reduce_max();
    for i in (chunks * 4)..len {
        if data[i] > result {
            result = data[i];
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Min reduction
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn min_f32x4_rust(data: *const f32, len: i32) -> f32 {
    let len = len as usize;
    let data = unsafe { std::slice::from_raw_parts(data, len) };

    if len == 0 {
        return f32::INFINITY;
    }

    let mut acc = f32x4::splat(f32::INFINITY);
    let chunks = len / 4;
    for i in 0..chunks {
        let off = i * 4;
        let v = f32x4::from_slice(&data[off..off + 4]);
        acc = acc.simd_min(v);
    }
    let mut result = acc.reduce_min();
    for i in (chunks * 4)..len {
        if data[i] < result {
            result = data[i];
        }
    }
    result
}
