"""What a benchmark ran on: the Mac's hardware and the software versions.

Every value comes from a probe that may fail (an older macOS, a sandbox);
a value that cannot be read is None, never an error.
"""

import json
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

TIMEOUT_S = 10


def _command(*args: str) -> str | None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=TIMEOUT_S)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 and result.stdout.strip() else None


def sysctl(name: str) -> str | None:
    return _command("sysctl", "-n", name)


def _int(value: str | None) -> int | None:
    try:
        return int(value) if value is not None else None
    except ValueError:
        return None


def cpu_cores() -> list[dict] | None:
    """Core count per performance level, named as macOS names them.

    "Performance" and "Efficiency" on M1 to M4; the M5 generation reports
    "Super" and "Performance".
    """
    levels = _int(sysctl("hw.nperflevels"))
    if not levels:
        return None
    cores = []
    for level in range(levels):
        count = _int(sysctl(f"hw.perflevel{level}.physicalcpu"))
        if count is None:
            return None
        cores.append({"name": sysctl(f"hw.perflevel{level}.name"), "count": count})
    return cores


def gpu_cores() -> int | None:
    output = _command("system_profiler", "SPDisplaysDataType", "-json")
    try:
        displays = json.loads(output)["SPDisplaysDataType"] if output else []
    except (ValueError, KeyError, TypeError):
        return None
    for display in displays:
        if isinstance(display, dict) and "sppci_cores" in display:
            return _int(str(display["sppci_cores"]))
    return None


def working_set_bytes() -> int | None:
    try:
        import mlx.core as mx

        return int(mx.device_info()["max_recommended_working_set_size"])
    except Exception:
        return None


def machine_info() -> dict:
    return {
        "model": sysctl("hw.model"),
        "chip": sysctl("machdep.cpu.brand_string"),
        "cpu_cores": cpu_cores(),
        "gpu_cores": gpu_cores(),
        "memory_bytes": _int(sysctl("hw.memsize")),
        "gpu_working_set_bytes": working_set_bytes(),
        "macos": platform.mac_ver()[0] or None,
    }


def _version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def software_info() -> dict:
    from . import __version__

    return {
        "mlx_decision": __version__,
        "mlx": _version("mlx"),
        "python": sys.version.split()[0],
    }


def describe(machine: dict) -> str:
    """One line, e.g. "Apple M5 Pro (Mac17,9), 6 Super + 12 Performance CPU cores, ..."."""
    parts = [machine.get("chip") or "unknown chip"]
    if machine.get("model"):
        parts[0] += f" ({machine['model']})"
    if machine.get("cpu_cores"):
        levels = " + ".join(f"{c['count']} {c['name'] or 'other'}" for c in machine["cpu_cores"])
        parts.append(f"{levels} CPU cores")
    if machine.get("gpu_cores"):
        parts.append(f"{machine['gpu_cores']} GPU cores")
    if machine.get("memory_bytes"):
        parts.append(f"{machine['memory_bytes'] / 2**30:.0f} GB memory")
    if machine.get("gpu_working_set_bytes"):
        parts.append(
            f"{machine['gpu_working_set_bytes'] / 1e9:.1f} GB GPU working set"
        )  # as the fit check
    if machine.get("macos"):
        parts.append(f"macOS {machine['macos']}")
    return ", ".join(parts)
