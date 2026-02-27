use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let kernel_src = manifest_dir.join("kernel.ea");

    // Recompile when the kernel source changes
    println!("cargo:rerun-if-changed=kernel.ea");

    // Compile kernel.ea → kernel.o
    let status = Command::new("ea")
        .arg(kernel_src.to_str().unwrap())
        .current_dir(&out_dir)
        .status()
        .expect("failed to run `ea` — is it on PATH?");
    assert!(status.success(), "Eä compilation failed");

    let obj_file = out_dir.join("kernel.o");

    // Create a static library from the object file
    let lib_file = out_dir.join("libkernel.a");
    let status = Command::new("ar")
        .args(["rcs", lib_file.to_str().unwrap(), obj_file.to_str().unwrap()])
        .status()
        .expect("failed to run `ar`");
    assert!(status.success(), "ar failed");

    // Tell cargo where to find the library
    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=kernel");
}
