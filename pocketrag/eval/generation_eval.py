from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

from ..generation.rag_pipeline import RAGPipeline


@dataclass
class QASample:
    """
    Single QA sample for generation eval.

    - q_id: query id
    - question: the question
    - answers: list of reference answers (text)
    """
    q_id: str
    question: str
    answers: List[str]


@dataclass
class GenerationMetrics:
    num_questions: int
    mean_em: float
    mean_f1: float


def load_qa_jsonl(path: Path | str) -> List[QASample]:
    """
    JSONL format:
    {"q_id": "...", "question": "...", "answers": ["...", "..."]}
    """
    p = Path(path).expanduser().resolve()
    samples: List[QASample] = []

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(
                QASample(
                    q_id=obj["q_id"],
                    question=obj["question"],
                    answers=list(obj["answers"]),
                )
            )

    return samples


# ----- Text normalization + metrics (SQuAD-style) -----

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WHITESPACE = re.compile(r"\s+", re.UNICODE)
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)


def _normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT.sub(" ", text)
    text = _ARTICLES.sub(" ", text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


def _f1_single(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    common = {}
    for t in pred_tokens:
        common[t] = common.get(t, 0) + 1
    overlap = 0
    for t in gold_tokens:
        if common.get(t, 0) > 0:
            overlap += 1
            common[t] -= 1

    if overlap == 0:
        return 0.0

    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, golds: Sequence[str]) -> float:
    pred_norm = _normalize(pred)
    for g in golds:
        if pred_norm == _normalize(g):
            return 1.0
    return 0.0


def f1_score(pred: str, golds: Sequence[str]) -> float:
    return max((_f1_single(pred, g) for g in golds), default=0.0)


def evaluate_generation(
    samples: Iterable[QASample],
    pipeline: RAGPipeline,
) -> GenerationMetrics:
    samples = list(samples)
    if not samples:
        raise ValueError("No QA samples provided for generation eval.")

    total_em = 0.0
    total_f1 = 0.0

    for s in samples:
        rag_ans = pipeline.answer(s.question)
        em = exact_match(rag_ans.answer, s.answers)
        f1 = f1_score(rag_ans.answer, s.answers)
        total_em += em
        total_f1 += f1

    n = len(samples)
    return GenerationMetrics(
        num_questions=n,
        mean_em=total_em / n,
        mean_f1=total_f1 / n,
    )