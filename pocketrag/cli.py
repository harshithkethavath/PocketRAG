import argparse
import json
from typing import Any, Dict
import os
import json
from pathlib import Path

from . import __version__
from .utils.hardware import generate_hardware_report, format_hardware_report
from .utils.logging import get_logger
from .data import (
    load_documents_from_dir,
    FixedSizeWordChunker,
    chunk_corpus,
)

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


    p_build = subparsers.add_parser(
        "build-chunks",
        help="Ingest a directory of text files and build word-based chunks.",
    )
    p_build.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory containing .txt files.",
    )
    p_build.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSONL file with chunks.",
    )
    p_build.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="Max words per chunk (default: 200).",
    )
    p_build.add_argument(
        "--overlap",
        type=int,
        default=40,
        help="Word overlap between chunks (default: 40).",
    )
    p_build.set_defaults(func=cmd_build_chunks)

    return parser


def cmd_build_chunks(args: argparse.Namespace) -> None:
    """
    Ingest a directory of text files, chunk them, and write to a JSONL file.
    """
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input dir does not exist or is not a directory: {input_dir}")

    logger.info(f"Loading documents from {input_dir} ...")
    documents = load_documents_from_dir(input_dir)

    if not documents:
        logger.warning("No documents found. Nothing to do.")
        return

    logger.info(f"Loaded {len(documents)} documents.")

    chunker = FixedSizeWordChunker(
        max_words=args.chunk_size,
        overlap=args.overlap,
    )

    logger.info(
        f"Chunking documents with FixedSizeWordChunker(max_words={args.chunk_size}, "
        f"overlap={args.overlap}) ..."
    )
    chunks = chunk_corpus(documents, chunker)
    logger.info(f"Generated {len(chunks)} chunks.")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Writing chunks to {output_path} (JSONL) ...")
    with output_path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + os.linesep)

    logger.info("Done.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        func = getattr(args, "func")
    except AttributeError:
        parser.print_help()
        return

    func(args)