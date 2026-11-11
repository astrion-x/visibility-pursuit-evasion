"""Build the shared geometry kernel with Clang, GCC or Zig cc."""

from pathlib import Path
import shutil
import subprocess
import sys


def build_kernel(output, compiler=None):
    compiler = (
        compiler or shutil.which("clang") or shutil.which("gcc") or shutil.which("zig")
    )
    if not compiler:
        raise RuntimeError(
            "Install Clang, GCC or Zig, or supply --compiler with its executable path"
        )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("forecast_kernel.c")
    suffix = (
        ".dll"
        if sys.platform == "win32"
        else ".dylib" if sys.platform == "darwin" else ".so"
    )
    target = output / ("forecast" + suffix)
    command = [str(compiler)]
    if Path(compiler).stem.lower() == "zig":
        command.append("cc")
    command += ["-O3", "-ffp-contract=off", "-shared", str(source), "-o", str(target)]
    if sys.platform == "win32" and "clang" in Path(compiler).name.lower():
        command += ["-fuse-ld=lld", "-nostdlib", "-Wl,/noentry"]
    elif sys.platform != "win32":
        command += ["-fPIC", "-lm"]
    subprocess.run(command, check=True)
    return target
