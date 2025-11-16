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
from .retrieval import (
    BM25Index,
    DenseIndex,
    EmbeddingModel,
    EmbeddingConfig,
)
from .generation import (
    GenerationConfig,
    LocalHFGenerator,
)
from .generation.rag_pipeline import RAGPipeline
from .eval import (
    load_qrels_jsonl,
    evaluate_retrieval,
    load_qa_jsonl,
    evaluate_generation as eval_gen_fn,
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


    p_dense_build = subparsers.add_parser(
        "build-dense-index",
        help="Build a dense (embedding) index from a chunks JSONL file.",
    )
    p_dense_build.add_argument(
        "--chunks-file",
        type=str,
        required=True,
        help="Path to JSONL file with DocumentChunks (from build-chunks).",
    )
    p_dense_build.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output dense index file (torch save).",
    )
    p_dense_build.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model to use (default: all-MiniLM-L6-v2).",
    )
    p_dense_build.add_argument(
        "--device-pref",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference for encoding (default: auto).",
    )
    p_dense_build.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32).",
    )
    p_dense_build.set_defaults(func=cmd_build_dense_index)

    p_dense_search = subparsers.add_parser(
        "search-dense",
        help="Run an ad-hoc dense retrieval query against a dense index.",
    )
    p_dense_search.add_argument(
        "--index",
        type=str,
        required=True,
        help="Path to dense index file (torch save).",
    )
    p_dense_search.add_argument(
        "--query",
        type=str,
        required=True,
        help="Search query string.",
    )
    p_dense_search.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top results to return (default: 5).",
    )
    p_dense_search.add_argument(
        "--device-pref",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference when running queries (default: auto).",
    )
    p_dense_search.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for query encoding (default: 32).",
    )
    p_dense_search.set_defaults(func=cmd_search_dense)


    p_eval = subparsers.add_parser(
        "eval-retrieval",
        help="Evaluate retrieval quality (Recall@K, MRR) for BM25 or Dense.",
    )
    p_eval.add_argument(
        "--qrels-file",
        type=str,
        required=True,
        help="Path to JSONL file with qrels (queries + relevant_doc_ids).",
    )
    p_eval.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["bm25", "dense"],
        help="Retrieval mode to evaluate: 'bm25' or 'dense'.",
    )
    p_eval.add_argument(
        "--bm25-index",
        type=str,
        help="Path to BM25 index pickle file (required if mode=bm25).",
    )
    p_eval.add_argument(
        "--dense-index",
        type=str,
        help="Path to dense index file (torch save, required if mode=dense).",
    )
    p_eval.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Cutoff K for metrics (default: 10).",
    )
    p_eval.add_argument(
        "--device-pref",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference for dense mode (default: auto).",
    )
    p_eval.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for dense encoding during evaluation (default: 32).",
    )
    p_eval.set_defaults(func=cmd_eval_retrieval)


    p_rag = subparsers.add_parser(
        "rag-answer",
        help="Run a single RAG query (retrieve + generate).",
    )
    p_rag.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["bm25", "dense"],
        help="Retrieval mode to use: 'bm25' or 'dense'.",
    )
    p_rag.add_argument(
        "--bm25-index",
        type=str,
        help="Path to BM25 index (required if mode=bm25).",
    )
    p_rag.add_argument(
        "--dense-index",
        type=str,
        help="Path to dense index (required if mode=dense).",
    )
    p_rag.add_argument(
        "--query",
        type=str,
        required=True,
        help="Question to ask.",
    )
    p_rag.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve as context (default: 5).",
    )
    p_rag.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF causal LM model name for generation.",
    )
    p_rag.add_argument(
        "--device-pref",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference for generation (default: auto).",
    )
    p_rag.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for generation (default: 256).",
    )
    p_rag.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    p_rag.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9).",
    )
    p_rag.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Max characters of concatenated context (default: 4000).",
    )
    p_rag.set_defaults(func=cmd_rag_answer)

    p_eval_gen = subparsers.add_parser(
        "eval-generation",
        help="Evaluate RAG generation (EM/F1) on a QA set.",
    )
    p_eval_gen.add_argument(
        "--qa-file",
        type=str,
        required=True,
        help="Path to QA JSONL file (q_id, question, answers).",
    )
    p_eval_gen.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["bm25", "dense"],
        help="Retrieval mode to use: 'bm25' or 'dense'.",
    )
    p_eval_gen.add_argument(
        "--bm25-index",
        type=str,
        help="Path to BM25 index (required if mode=bm25).",
    )
    p_eval_gen.add_argument(
        "--dense-index",
        type=str,
        help="Path to dense index (required if mode=dense).",
    )
    p_eval_gen.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve as context (default: 5).",
    )
    p_eval_gen.add_argument(
        "--model-name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HF causal LM model name for generation.",
    )
    p_eval_gen.add_argument(
        "--device-pref",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device preference for generation (default: auto).",
    )
    p_eval_gen.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Max new tokens for generation (default: 256).",
    )
    p_eval_gen.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    p_eval_gen.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p nucleus sampling (default: 0.9).",
    )
    p_eval_gen.add_argument(
        "--max-context-chars",
        type=int,
        default=4000,
        help="Max characters of concatenated context (default: 4000).",
    )
    p_eval_gen.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="(Used for dense retrieval encoder.)",
    )
    p_eval_gen.set_defaults(func=cmd_eval_generation)

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


def cmd_build_dense_index(args: argparse.Namespace) -> None:
    """
    Build a dense (embedding) index from a chunks JSONL file.
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

    logger.info(
        f"Loaded {len(chunks)} chunks. Building dense index with "
        f"model={args.model_name!r}, device_pref={args.device_pref!r}, "
        f"batch_size={args.batch_size} ..."
    )

    encoder = EmbeddingModel(
        EmbeddingConfig(
            model_name=args.model_name,
            device_pref=args.device_pref,
            batch_size=args.batch_size,
        )
    )

    index = DenseIndex.from_chunks(
        chunks=chunks,
        encoder=encoder,
        batch_size=args.batch_size,
    )

    logger.info(
        f"Dense index built: "
        f"N={index.embeddings.shape[0]}, "
        f"D={index.embeddings.shape[1]}, "
        f"model={index.model_name}"
    )

    logger.info(f"Saving dense index to {output_path} ...")
    index.save(output_path)
    logger.info("Done.")


def cmd_search_dense(args: argparse.Namespace) -> None:
    """
    Load a dense index and run an ad-hoc query via cosine similarity.
    """
    index_path = Path(args.index).expanduser().resolve()
    if not index_path.exists():
        raise SystemExit(f"Index file does not exist: {index_path}")

    logger.info(f"Loading dense index from {index_path} ...")
    index = DenseIndex.load(index_path)
    logger.info(
        f"Loaded dense index with "
        f"N={index.embeddings.shape[0]}, "
        f"D={index.embeddings.shape[1]}, "
        f"model={index.model_name}"
    )

    encoder = EmbeddingModel(
        EmbeddingConfig(
            model_name=index.model_name,
            device_pref=args.device_pref,
            batch_size=args.batch_size,
        )
    )

    query = args.query
    if not query:
        raise SystemExit("Query string is empty.")

    logger.info(f"Running dense query: {query!r}")
    results = index.search(
        query=query,
        encoder=encoder,
        top_k=args.top_k,
    )

    if not results:
        print("No results.")
        return

    print(f"Top {len(results)} dense results:")
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


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    """
    Evaluate retrieval quality for BM25 or Dense index using a qrels file.
    """
    qrels_path = Path(args.qrels_file).expanduser().resolve()
    if not qrels_path.exists():
        raise SystemExit(f"Qrels file does not exist: {qrels_path}")

    logger.info(f"Loading qrels from {qrels_path} ...")
    samples = load_qrels_jsonl(qrels_path)
    logger.info(f"Loaded {len(samples)} evaluation queries.")

    mode = args.mode.lower()
    top_k = args.top_k

    if mode == "bm25":
        index_path = Path(args.bm25_index).expanduser().resolve()
        if not index_path.exists():
            raise SystemExit(f"BM25 index file does not exist: {index_path}")

        logger.info(f"Loading BM25 index from {index_path} ...")
        bm25_index = BM25Index.load(index_path)
        logger.info(
            f"BM25 index: N={bm25_index.N}, avgdl={bm25_index.avgdl:.2f}"
        )

        def retrieve_fn(query: str, k: int):
            return bm25_index.score_query(query, top_k=k)

    elif mode == "dense":
        index_path = Path(args.dense_index).expanduser().resolve()
        if not index_path.exists():
            raise SystemExit(f"Dense index file does not exist: {index_path}")

        logger.info(f"Loading dense index from {index_path} ...")
        dense_index = DenseIndex.load(index_path)
        logger.info(
            f"Dense index: N={dense_index.embeddings.shape[0]}, "
            f"D={dense_index.embeddings.shape[1]}, model={dense_index.model_name}"
        )

        encoder = EmbeddingModel(
            EmbeddingConfig(
                model_name=dense_index.model_name,
                device_pref=args.device_pref,
                batch_size=args.batch_size,
            )
        )

        def retrieve_fn(query: str, k: int):
            return dense_index.search(query=query, encoder=encoder, top_k=k)

    else:
        raise SystemExit(f"Unknown mode: {mode}. Expected 'bm25' or 'dense'.")

    logger.info(
        f"Evaluating retrieval: mode={mode}, top_k={top_k}, "
        f"num_queries={len(samples)} ..."
    )

    metrics = evaluate_retrieval(samples, retrieve_fn, top_k=top_k)

    print()
    print("Retrieval Evaluation Results")
    print("----------------------------")
    print(f"Mode           : {mode}")
    print(f"Num queries    : {metrics.num_queries}")
    print(f"Top-K          : {metrics.top_k}")
    print(f"Mean Recall@K  : {metrics.mean_recall:.4f}")
    print(f"Mean HitRate@K : {metrics.mean_hit_rate:.4f}")
    print(f"Mean MRR       : {metrics.mean_mrr:.4f}")
    print()


def build_retrieve_fn_for_mode(args: argparse.Namespace):
    mode = args.mode.lower()

    if mode == "bm25":
        index_path = Path(args.bm25_index).expanduser().resolve()
        if not index_path.exists():
            raise SystemExit(f"BM25 index file does not exist: {index_path}")

        logger.info(f"Loading BM25 index from {index_path} ...")
        bm25_index = BM25Index.load(index_path)
        logger.info(
            f"BM25 index: N={bm25_index.N}, avgdl={bm25_index.avgdl:.2f}"
        )

        def retrieve_fn(query: str, k: int):
            return bm25_index.score_query(query, top_k=k)

    elif mode == "dense":
        index_path = Path(args.dense_index).expanduser().resolve()
        if not index_path.exists():
            raise SystemExit(f"Dense index file does not exist: {index_path}")

        logger.info(f"Loading dense index from {index_path} ...")
        dense_index = DenseIndex.load(index_path)
        logger.info(
            f"Dense index: N={dense_index.embeddings.shape[0]}, "
            f"D={dense_index.embeddings.shape[1]}, "
            f"model={dense_index.model_name}"
        )

        encoder = EmbeddingModel(
            EmbeddingConfig(
                model_name=dense_index.model_name,
                device_pref=args.device_pref,
                batch_size=args.batch_size,
            )
        )

        def retrieve_fn(query: str, k: int):
            return dense_index.search(query=query, encoder=encoder, top_k=k)

    else:
        raise SystemExit(f"Unknown mode: {mode}. Expected 'bm25' or 'dense'.")

    return retrieve_fn


def cmd_rag_answer(args: argparse.Namespace) -> None:
    """
    Run a single RAG query (retrieve + generate) and print the answer and context.
    """
    retrieve_fn = build_retrieve_fn_for_mode(args)

    gen_cfg = GenerationConfig(
        model_name=args.model_name,
        device_pref=args.device_pref,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generator = LocalHFGenerator(gen_cfg)

    pipeline = RAGPipeline(
        retrieve_fn=retrieve_fn,
        generator=generator,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
    )

    question = args.query
    rag_answer = pipeline.answer(question)

    print("\nQuestion")
    print("--------")
    print(question)
    print("\nAnswer")
    print("------")
    print(rag_answer.answer)
    print("\nRetrieved contexts")
    print("------------------")
    for i, r in enumerate(rag_answer.retrieved, start=1):
        snippet = r.text[:200].replace("\n", " ")
        if len(r.text) > 200:
            snippet += " ..."
        print(
            f"{i:2d}. score={r.score:.4f} doc_id={r.doc_id} chunk_id={r.chunk_id}"
        )
        print(f"    {snippet}")
    print()


def cmd_eval_generation(args: argparse.Namespace) -> None:
    """
    Evaluate end-to-end RAG generation with EM/F1 against reference answers.
    """
    qa_path = Path(args.qa_file).expanduser().resolve()
    if not qa_path.exists():
        raise SystemExit(f"QA file does not exist: {qa_path}")

    logger.info(f"Loading QA samples from {qa_path} ...")
    samples = load_qa_jsonl(qa_path)
    logger.info(f"Loaded {len(samples)} QA samples.")

    retrieve_fn = build_retrieve_fn_for_mode(args)

    gen_cfg = GenerationConfig(
        model_name=args.model_name,
        device_pref=args.device_pref,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    generator = LocalHFGenerator(gen_cfg)

    pipeline = RAGPipeline(
        retrieve_fn=retrieve_fn,
        generator=generator,
        top_k=args.top_k,
        max_context_chars=args.max_context_chars,
    )

    logger.info(
        f"Evaluating generation: mode={args.mode}, "
        f"top_k={args.top_k}, num_questions={len(samples)} ..."
    )

    metrics = eval_gen_fn(samples, pipeline)

    print()
    print("Generation Evaluation Results")
    print("----------------------------")
    print(f"Mode          : {args.mode}")
    print(f"Num questions : {metrics.num_questions}")
    print(f"Mean EM       : {metrics.mean_em:.4f}")
    print(f"Mean F1       : {metrics.mean_f1:.4f}")
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