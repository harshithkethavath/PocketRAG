import platform
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
    import psutil
except ImportError:
    psutil = None

try:
    import torch
except ImportError:
    torch = None

@dataclass
class HardwareReport:
    os: str
    os_version: str
    cpu: str
    cpu_count_logical: Optional[int]
    cpu_count_physical: Optional[int]
    total_ram_gb: Optional[float]
    has_cuda: bool
    cuda_device_count: int
    has_mps: bool
    torch_version: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _get_cpu_name() -> str:
    """
    Try to get a human-readable CPU name. This is not perfect, but good enough
    for reproducible reporting on macOS/Linux/Windows.
    """
    system = platform.system()

    if system == "Darwin":
        return platform.processor() or "Unknown Apple CPU"
    elif system == "Windows":
        return platform.processor() or "Unknown Windows CPU"
    else:
        try:
            import subprocess

            out = subprocess.check_output(
                ["grep", "-m", "1", "model name", "/proc/cpuinfo"],
                text=True,
            )
            return out.split(":", 1)[1].strip()
        except Exception:
            return platform.processor() or "Unknown CPU"


def generate_hardware_report() -> HardwareReport:
    system = platform.system()
    os_version = platform.version()

    cpu_name = _get_cpu_name()
    logical = physical = None
    total_ram_gb = None

    if psutil is not None:
        logical = psutil.cpu_count(logical=True)
        physical = psutil.cpu_count(logical=False)
        total_ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)

    has_cuda = False
    cuda_device_count = 0
    has_mps = False
    torch_version = None

    if torch is not None:
        torch_version = torch.__version__
        has_cuda = torch.cuda.is_available()
        cuda_device_count = torch.cuda.device_count() if has_cuda else 0
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    return HardwareReport(
        os=system,
        os_version=os_version,
        cpu=cpu_name,
        cpu_count_logical=logical,
        cpu_count_physical=physical,
        total_ram_gb=total_ram_gb,
        has_cuda=has_cuda,
        cuda_device_count=cuda_device_count,
        has_mps=has_mps,
        torch_version=torch_version,
    )


def format_hardware_report(report: HardwareReport) -> str:
    """
    Turn the report into a human-readable multi-line string for the CLI.
    """
    lines = [
        "PocketRAG Hardware & Software Report",
        "------------------------------------",
        f"OS              : {report.os} ({report.os_version})",
        f"CPU             : {report.cpu}",
        f"Logical cores   : {report.cpu_count_logical}",
        f"Physical cores  : {report.cpu_count_physical}",
        f"Total RAM (GB)  : {report.total_ram_gb}",
        "",
        f"PyTorch version : {report.torch_version}",
        f"CUDA available  : {report.has_cuda} (devices: {report.cuda_device_count})",
        f"MPS available   : {report.has_mps}",
    ]
    return "\n".join(lines)