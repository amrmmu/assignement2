"""
Collect hardware and container specification information.
Reads Docker cgroup limits (v1 and v2) when available; falls back
to host-level psutil values when running outside a container.
"""

import os
import platform
import sys

import psutil


def _read_file(path: str) -> str:
    """Read a text file and return stripped contents, or '' on error."""
    try:
        with open(path, "r") as fh:
            return fh.read().strip()
    except Exception:
        return ""


def _detect_cpu_limit() -> str:
    """
    Try to read the container CPU limit from cgroup v1 or v2.
    Returns a human-readable string like '4.0 CPUs' or 'No limit detected'.
    """
    # cgroup v1
    quota = _read_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota and period:
        try:
            q, p = int(quota), int(period)
            if q > 0:
                return f"{q / p:.1f} CPUs"
        except ValueError:
            pass

    # cgroup v2
    cpu_max = _read_file("/sys/fs/cgroup/cpu.max")
    if cpu_max:
        parts = cpu_max.split()
        if len(parts) == 2 and parts[0] != "max":
            try:
                return f"{int(parts[0]) / int(parts[1]):.1f} CPUs"
            except ValueError:
                pass

    return "No limit detected (host logical CPUs: {})".format(
        psutil.cpu_count(logical=True)
    )


def _detect_ram_limit() -> str:
    """
    Try to read the container memory limit from cgroup v1 or v2.
    Returns a human-readable string like '4.00 GB' or 'No limit detected'.
    """
    UNLIMITED = 2 ** 62  # sentinel for "no limit" in cgroup v1

    # cgroup v1
    mem_limit = _read_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if mem_limit:
        try:
            val = int(mem_limit)
            if val < UNLIMITED:
                return f"{val / (1024 ** 3):.2f} GB"
        except ValueError:
            pass

    # cgroup v2
    mem_max = _read_file("/sys/fs/cgroup/memory.max")
    if mem_max and mem_max != "max":
        try:
            return f"{int(mem_max) / (1024 ** 3):.2f} GB"
        except ValueError:
            pass

    total = psutil.virtual_memory().total
    return f"No limit detected (host total: {total / (1024 ** 3):.2f} GB)"


def get_system_info() -> dict:
    """
    Return a dictionary with hardware and environment information.
    Fields: cpu_model, cpu_logical_cores, cpu_physical_cores, cpu_limit,
            total_ram_gb, ram_limit, os, python_version, storage.
    """
    try:
        cpu_model = platform.processor() or "N/A (not exposed in container)"
    except Exception:
        cpu_model = "N/A"

    total_ram = psutil.virtual_memory().total

    return {
        "cpu_model": cpu_model,
        "cpu_logical_cores": psutil.cpu_count(logical=True),
        "cpu_physical_cores": psutil.cpu_count(logical=False),
        "cpu_limit": _detect_cpu_limit(),
        "total_ram_gb": f"{total_ram / (1024 ** 3):.2f} GB",
        "ram_limit": _detect_ram_limit(),
        "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python_version": sys.version,
        "storage": (
            "Docker overlay2 filesystem — container read/write layer backed "
            "by the host Docker storage driver. Results directory is bind-mounted "
            "from the host, so output files persist after the container exits."
        ),
    }


if __name__ == "__main__":
    import json
    info = get_system_info()
    print(json.dumps(info, indent=2))
