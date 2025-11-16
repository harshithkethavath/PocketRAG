from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence, Set

from ..retrieval import ScoredChunk


@dataclass
class QuerySample:
    """
    Single evaluation query.

    - q_id: query id
    - query: natural-language query string
    - relevant_doc_ids: set of doc_ids considered relevant
    """
    q_id: str
    query: str
    relevant_doc_ids: Set[str]


def load_qrels_jsonl(path: Path | str) -> List[QuerySample]:
    """
    Load query + relevance info from a JSONL file.

    Each line must be a JSON object with:
    - q_id: str
    - query: str
    - relevant_doc_ids: List[str]
    """
    p = Path(path).expanduser().resolve()
    samples: List[QuerySample] = []

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            q_id = obj["q_id"]
            query = obj["query"]
            rel_ids = set(obj["relevant_doc_ids"])
            samples.append(
                QuerySample(
                    q_id=q_id,
                    query=query,
                    relevant_doc_ids=rel_ids,
                )
            )

    return samples


@dataclass
class RetrievalMetrics:
    num_queries: int
    top_k: int
    mean_recall: float
    mean_hit_rate: float
    mean_mrr: float


def _compute_metrics_for_query(
    relevant_doc_ids: Set[str],
    retrieved: Sequence[ScoredChunk],
    top_k: int,
) -> Dict[str, float]:
    """
    Compute Recall@K, Hit@K, and MRR for a single query.
    """
    if not relevant_doc_ids:
        return {"recall": 0.0, "hit": 0.0, "rr": 0.0}

    retrieved_doc_ids: List[str] = []
    for r in retrieved[:top_k]:
        if r.doc_id not in retrieved_doc_ids:
            retrieved_doc_ids.append(r.doc_id)

    retrieved_relevant = [d for d in retrieved_doc_ids if d in relevant_doc_ids]
    recall = len(retrieved_relevant) / float(len(relevant_doc_ids))

    hit = 1.0 if retrieved_relevant else 0.0

    rr = 0.0
    for rank, doc_id in enumerate(retrieved_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            rr = 1.0 / rank
            break

    return {"recall": recall, "hit": hit, "rr": rr}


def evaluate_retrieval(
    samples: Iterable[QuerySample],
    retrieve_fn: Callable[[str, int], List[ScoredChunk]],
    top_k: int,
) -> RetrievalMetrics:
    """
    Evaluate a retrieval system using a generic retrieve_fn(query, top_k).

    - samples: list of QuerySample
    - retrieve_fn: function that returns top_k ScoredChunk objects for a query
    - top_k: cutoff for metrics
    """
    samples = list(samples)
    if not samples:
        raise ValueError("No evaluation samples provided.")

    total_recall = 0.0
    total_hit = 0.0
    total_rr = 0.0

    for s in samples:
        retrieved = retrieve_fn(s.query, top_k)
        metrics = _compute_metrics_for_query(
            relevant_doc_ids=s.relevant_doc_ids,
            retrieved=retrieved,
            top_k=top_k,
        )
        total_recall += metrics["recall"]
        total_hit += metrics["hit"]
        total_rr += metrics["rr"]

    n = len(samples)
    return RetrievalMetrics(
        num_queries=n,
        top_k=top_k,
        mean_recall=total_recall / n,
        mean_hit_rate=total_hit / n,
        mean_mrr=total_rr / n,
    )