"""Run the requested methods sequentially for one generated initial pair."""

import json
from pathlib import Path
import sys
from .config import PARAMETERS, Settings, load_scenes
from .acceleration import CompiledGeometry
from .initial_states import generate


def write_json(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(
        json.dumps(content, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def peak_memory():
    """Process-lifetime peak resident/working-set memory in bytes."""
    if sys.platform == "win32":
        import ctypes as ct
        from ctypes import wintypes as wt

        class Counters(ct.Structure):
            _fields_ = [("cb", wt.DWORD), ("PageFaultCount", wt.DWORD)] + [
                (name, ct.c_size_t)
                for name in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        kernel = ct.WinDLL("kernel32", use_last_error=True)
        psapi = ct.WinDLL("psapi", use_last_error=True)
        kernel.GetCurrentProcess.restype = wt.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wt.HANDLE,
            ct.POINTER(Counters),
            wt.DWORD,
        ]
        psapi.GetProcessMemoryInfo.restype = wt.BOOL
        counters = Counters()
        counters.cb = ct.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(
            kernel.GetCurrentProcess(), ct.byref(counters), counters.cb
        ):
            raise ct.WinError(ct.get_last_error())
        return int(counters.PeakWorkingSetSize)
    import resource

    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def execute_case(scene, seed, methods, output, kernel):
    from .rollout import run

    initial = generate(scene, seed)
    write_json(Path(output) / "initial_states" / f"{scene}_{seed}.json", initial)
    settings = Settings()
    geometry = None
    statuses = {}
    for method in methods:
        dest = Path(output) / "runs" / scene / f"{seed}_{method}.json"
        if dest.exists():
            previous = json.loads(dest.read_text(encoding="utf-8"))
            if (previous["scene"], previous["seed"], previous["method"]) != (
                scene,
                seed,
                method,
            ):
                raise ValueError(f"Unexpected record identity: {dest.name}")
            statuses[method] = previous["status"]
            continue
        if geometry is None:
            geometry = CompiledGeometry(
                load_scenes(),
                scene,
                [initial["P"], initial["E"]],
                parameters=PARAMETERS,
                settings=settings,
                kernel=kernel,
            )
        result = run(initial, method, geometry, settings=settings)
        result["process_peak_memory_bytes"] = peak_memory()
        write_json(dest, result)
        statuses[method] = result["status"]
    return statuses
