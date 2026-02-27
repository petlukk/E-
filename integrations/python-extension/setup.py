"""Custom build that compiles Eä kernels before packaging."""

import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildEaKernels(build_py):
    """Compile .ea kernels into .so before building the Python package."""

    def run(self):
        src_dir = Path(__file__).parent
        kernel_src = src_dir / "kernel.ea"
        pkg_dir = src_dir / "ea_kernels"

        # Compile kernel.ea → kernel.so
        print(f"Compiling {kernel_src} with Eä compiler...")
        result = subprocess.run(
            ["ea", str(kernel_src), "--lib"],
            cwd=str(src_dir),
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(result.stderr, file=sys.stderr)
            raise RuntimeError("Eä compilation failed")
        print(result.stderr, end="")  # compiler prints status to stderr

        # Move .so into the package directory
        so_file = src_dir / "kernel.so"
        dest = pkg_dir / "kernel.so"
        shutil.copy2(str(so_file), str(dest))
        so_file.unlink()
        print(f"Installed {dest}")

        super().run()


setup(cmdclass={"build_py": BuildEaKernels})
