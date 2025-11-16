import argparse
import json
from typing import Any, Dict

from . import __version__
from .utils.hardware import generate_hardware_report, format_hardware_report
from .utils.logging import get_logger

logger = get_logger(__name__)


def cmd_version(_: argparse.Namespace) -> None:
    """Print PocketRAG version."""
    print(f"PocketRAG version {__version__}")


def cmd_hardware_report(args: argparse.Namespace) -> None:
    """
    Print hardware/software info, optionally as JSON.
    """
    report = generate_hardware_report()

    if args.json:
        as_dict: Dict[str, Any] = report.to_dict()
        print(json.dumps(as_dict, indent=2))
    else:
        print(format_hardware_report(report))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pocketrag",
        description="PocketRAG: RAG benchmarking on consumer hardware.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_version = subparsers.add_parser("version", help="Show PocketRAG version.")
    p_version.set_defaults(func=cmd_version)

    p_hw = subparsers.add_parser(
        "hardware-report",
        help="Show hardware and software environment information.",
    )
    p_hw.add_argument(
        "--json",
        action="store_true",
        help="Output the hardware report as JSON.",
    )
    p_hw.set_defaults(func=cmd_hardware_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        func = getattr(args, "func")
    except AttributeError:
        parser.print_help()
        return

    func(args)