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
    load_chunks_from_jsonl,
)
from .retrieval import BM25Index

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


    p_bm25_build = subparsers.add_parser(
        "build-bm25-index",
        help="Build a BM25 index from a chunks JSONL file.",
    )
    p_bm25_build.add_argument(
        "--chunks-file",
        type=str,
        required=True,
        help="Path to JSONL file with DocumentChunks (from build-chunks).",
    )
    p_bm25_build.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output BM25 index pickle file.",
    )
    p_bm25_build.add_argument(
        "--k1",
        type=float,
        default=1.5,
        help="BM25 k1 parameter (default: 1.5).",
    )
    p_bm25_build.add_argument(
        "--b",
        type=float,
        default=0.75,
        help="BM25 b parameter (default: 0.75).",
    )
    p_bm25_build.set_defaults(func=cmd_build_bm25_index)

    # pocketrag search-bm25
    p_bm25_search = subparsers.add_parser(
        "search-bm25",
        help="Run an ad-hoc search query against a BM25 index.",
    )
    p_bm25_search.add_argument(
        "--index",
        type=str,
        required=True,
        help="Path to BM25 index pickle file.",
    )
    p_bm25_search.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query string.",
    )
    p_bm25_search.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5).",
    )
    p_bm25_search.set_defaults(func=cmd_search_bm25)

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


def cmd_build_bm25_index(args: argparse.Namespace) -> None:
    """
    Build a BM25 index from a chunks JSONL file and save it as a pickle.
    """
    chunks_file = Path(args.chunks_file).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not chunks_file.exists():
        raise SystemExit(f"Chunks file does not exist: {chunks_file}")

    logger.info(f"Loading chunks from {chunks_file} ...")
    chunks = load_chunks_from_jsonl(chunks_file)
    if not chunks:
        logger.warning("No chunks loaded. Nothing to index.")
        return

    logger.info(f"Loaded {len(chunks)} chunks. Building BM25 index ...")

    index = BM25Index.from_chunks(
        chunks,
        k1=args.k1,
        b=args.b,
    )

    logger.info(
        f"Built BM25 index with N={index.N}, avgdl={index.avgdl:.2f}, "
        f"k1={index.k1}, b={index.b}"
    )

    logger.info(f"Saving BM25 index to {output_path} ...")
    index.save(output_path)
    logger.info("Done.")


def cmd_search_bm25(args: argparse.Namespace) -> None:
    """
    Load a BM25 index and run an ad-hoc query.
    """
    index_path = Path(args.index).expanduser().resolve()
    if not index_path.exists():
        raise SystemExit(f"Index file does not exist: {index_path}")

    logger.info(f"Loading BM25 index from {index_path} ...")
    index = BM25Index.load(index_path)
    logger.info(f"Loaded BM25 index with N={index.N}, avgdl={index.avgdl:.2f}")

    query = args.query
    if not query:
        raise SystemExit("Query string is empty.")

    logger.info(f"Running query: {query!r}")
    results = index.score_query(query, top_k=args.top_k)

    if not results:
        print("No results.")
        return

    print(f"Top {len(results)} results:")
    print("-" * 60)
    for rank, r in enumerate(results, start=1):
        snippet = r.text[:200].replace("\n", " ")
        if len(r.text) > 200:
            snippet += " ..."
        print(
            f"{rank:2d}. score={r.score:.4f}  "
            f"doc_id={r.doc_id}  chunk_id={r.chunk_id}"
        )
        print(f"    {snippet}")
        print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        func = getattr(args, "func")
    except AttributeError:
        parser.print_help()
        return

    func(args)