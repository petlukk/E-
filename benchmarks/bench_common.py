"""
Shared helpers for competitor benchmarks.

Provides tool detection, compilation, and library loading for:
- Clang (various versions)
- ISPC
- Rust nightly (std::simd)

All functions return bool/path/None — never sys.exit().
Callers decide whether to skip a competitor.
"""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path


# ---------------------------------------------------------------------------
# Tool detection
# ---------------------------------------------------------------------------

def has_clang(version: int = 14) -> str | None:
    """Return the clang binary name if the requested version is installed.

    Tries clang-<version> first, then falls back to unversioned 'clang'
    only if it actually reports the requested major version.
    """
    # Try versioned binary first
    versioned = f"clang-{version}"
    if shutil.which(versioned):
        try:
            out = subprocess.run(
                [versioned, "--version"], capture_output=True, text=True,
                timeout=5,
            )
            if out.returncode == 0:
                return versioned
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Fall back to unversioned 'clang' only if its major version matches
    if shutil.which("clang"):
        try:
            out = subprocess.run(
                ["clang", "--version"], capture_output=True, text=True,
                timeout=5,
            )
            if out.returncode == 0 and f"version {version}." in out.stdout:
                return "clang"
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    return None


def has_ispc() -> str | None:
    """Return the ispc binary path if installed."""
    path = shutil.which("ispc")
    return path


def has_rust_nightly() -> bool:
    """Return True if cargo +nightly is available."""
    try:
        out = subprocess.run(
            ["cargo", "+nightly", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        return out.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Compilation helpers
# ---------------------------------------------------------------------------

def compile_with_clang(
    clang_bin: str,
    src: str | Path,
    output: str | Path,
    cwd: str | Path,
    extra_flags: list[str] | None = None,
) -> bool:
    """Compile a C source file with clang to a shared library.

    Returns True on success.
    """
    cmd = [
        clang_bin, "-O3", "-march=native", "-ffast-math",
        "-shared", "-fPIC",
        str(src), "-o", str(output),
    ]
    if extra_flags:
        cmd.extend(extra_flags)
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(cwd), timeout=30
        )
        if result.returncode != 0:
            print(f"  Clang compilation failed: {result.stderr.strip()}")
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  Clang compilation error: {e}")
        return False


def compile_ispc(
    src: str | Path,
    output_so: str | Path,
    cwd: str | Path,
    target: str = "avx2-i32x8",
) -> bool:
    """Compile an ISPC source to a shared library (.ispc -> .o -> .so).

    Returns True on success.
    """
    src = Path(src)
    obj = src.with_suffix(".o")

    # Step 1: ispc -> .o
    try:
        r1 = subprocess.run(
            ["ispc", str(src), "-o", str(obj),
             f"--target={target}", "-O2", "--pic"],
            capture_output=True, text=True, cwd=str(cwd), timeout=30,
        )
        if r1.returncode != 0:
            print(f"  ISPC compile failed: {r1.stderr.strip()}")
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  ISPC error: {e}")
        return False

    # Step 2: .o -> .so  (use gcc to link)
    try:
        r2 = subprocess.run(
            ["gcc", "-shared", str(obj), "-o", str(output_so)],
            capture_output=True, text=True, cwd=str(cwd), timeout=15,
        )
        if r2.returncode != 0:
            print(f"  ISPC link failed: {r2.stderr.strip()}")
            return False
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  ISPC link error: {e}")
        return False


def compile_rust_competitors(crate_dir: str | Path) -> Path | None:
    """Build the Rust competitors cdylib crate.

    Returns the path to the .so on success, None otherwise.
    """
    crate_dir = Path(crate_dir)
    env = os.environ.copy()
    env["RUSTFLAGS"] = "-C target-cpu=native"

    try:
        result = subprocess.run(
            ["cargo", "+nightly", "build", "--release"],
            capture_output=True, text=True,
            cwd=str(crate_dir), timeout=120, env=env,
        )
        if result.returncode != 0:
            print(f"  Rust build failed: {result.stderr.strip()}")
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  Rust build error: {e}")
        return None

    so_path = crate_dir / "target" / "release" / "libea_competitors.so"
    if so_path.exists():
        return so_path
    # Try .dylib on macOS
    dylib = so_path.with_suffix(".dylib")
    if dylib.exists():
        return dylib
    print(f"  Rust build produced no library at {so_path}")
    return None


# ---------------------------------------------------------------------------
# Library loading
# ---------------------------------------------------------------------------

def try_load(path: str | Path, label: str) -> ctypes.CDLL | None:
    """Load a shared library via ctypes, returning None on failure."""
    try:
        lib = ctypes.CDLL(str(path))
        return lib
    except OSError as e:
        print(f"  Could not load {label}: {e}")
        return None


# ---------------------------------------------------------------------------
# Version reporting
# ---------------------------------------------------------------------------

def _run_version(cmd: list[str]) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            return r.stdout.splitlines()[0].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def print_competitor_versions():
    """Print detected versions of all competitor toolchains."""
    print("--- Competitor Toolchains ---")

    # Clang (deduplicate — don't print the same binary twice)
    seen_clang = set()
    for ver in [14, 16, 17, 18]:
        name = has_clang(ver)
        if name and name not in seen_clang:
            seen_clang.add(name)
            v = _run_version([name, "--version"])
            print(f"  {name}: {v or 'found'}")

    # ISPC
    if has_ispc():
        v = _run_version(["ispc", "--version"])
        print(f"  ispc: {v or 'found'}")
    else:
        print("  ispc: not found (skipping)")

    # Rust nightly
    if has_rust_nightly():
        v = _run_version(["cargo", "+nightly", "--version"])
        print(f"  cargo nightly: {v or 'found'}")
        v = _run_version(["rustup", "run", "nightly", "rustc", "--version"])
        if v:
            print(f"  rustc nightly: {v}")
    else:
        print("  rust nightly: not found (skipping)")

    print()
