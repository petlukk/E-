# Running Tests on Linux

These instructions assume an Ubuntu/Debian-based system. Adjust package manager commands if using Fedora/Arch/etc.

## 1. Install Prerequisites

You need Rust and LLVM 14 development libraries.

### Install Rust (if not installed)
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### Install LLVM 14 & Dependencies
The `inkwell` crate requires LLVM 14 specifically.
```bash
sudo apt-get update
sudo apt-get install -y llvm-14-dev libpolly-14-dev libzstd-dev libffi-dev zlib1g-dev build-essential
```

If `llvm-14-dev` is not found, you may need to add the LLVM repository:
```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 14
```

## 2. Transfer Code

Copy the entire `EA2` directory to your Linux machine.
Ensure `Cargo.toml`, `src/`, and `tests/` are present.

## 3. Run Tests

Navigate to the project directory and run:

```bash
# Set LLVM config path explicitly if needed (usually auto-detected)
export LLVM_SYS_140_PREFIX=/usr/lib/llvm-14

# Run all tests
cargo test --features=llvm
```

## 4. Expected Output

You should see output ending with something like:

```text
test result: ok. 56 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 2.34s
```

If specific tests fail, please copy the failure output and share it.

## 5. Code Quality Check (Clippy)

To ensure the code is idiomatic and bug-free, run Clippy:

```bash
cargo clippy --features=llvm -- -D warnings
```

If this command produces no output (or just "Checking..." then finishes), the code is clean. If errors/warnings appear, please share them.
