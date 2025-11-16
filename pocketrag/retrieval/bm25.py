from __future__ import annotations

import math
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ..data import DocumentChunk


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    """
    Very simple tokenizer: lowercase + word characters.
    """
    return _TOKEN_PATTERN.findall(text.lower())


@dataclass
class ScoredChunk:
    chunk_id: str
    doc_id: str
    score: float
    text: str


@dataclass
class BM25Index:
    """
    Simple BM25 index over DocumentChunk objects.

    We keep:
    - chunks: the original chunks (for text + ids)
    - tokenized_docs: list of token lists
    - term_freqs: per-doc term frequency dicts
    - doc_lens: length of each doc in tokens
    - df: document frequency per term
    - N: number of documents
    - avgdl: average document length
    """

    chunks: List[DocumentChunk]
    tokenized_docs: List[List[str]]
    term_freqs: List[Dict[str, int]]
    doc_lens: List[int]
    df: Dict[str, int]
    N: int
    avgdl: float
    k1: float = 1.5
    b: float = 0.75


    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[DocumentChunk],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        chunks = list(chunks)
        tokenized_docs: List[List[str]] = []
        term_freqs: List[Dict[str, int]] = []
        doc_lens: List[int] = []
        df: Dict[str, int] = {}

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            tokenized_docs.append(tokens)

            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            term_freqs.append(tf)

            doc_len = len(tokens)
            doc_lens.append(doc_len)

            for t in tf.keys():
                df[t] = df.get(t, 0) + 1

        N = len(chunks)
        avgdl = sum(doc_lens) / N if N > 0 else 0.0

        return cls(
            chunks=chunks,
            tokenized_docs=tokenized_docs,
            term_freqs=term_freqs,
            doc_lens=doc_lens,
            df=df,
            N=N,
            avgdl=avgdl,
            k1=k1,
            b=b,
        )


    def save(self, path: Path | str) -> None:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "BM25Index":
        p = Path(path).expanduser().resolve()
        with p.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded object is not a BM25Index: {type(obj)}")
        return obj


    def _idf(self, term: str) -> float:
        """
        BM25 idf with 1+ smoothing to avoid negatives.
        """
        n_qi = self.df.get(term, 0)
        if n_qi == 0:
            return 0.0
        return math.log(1.0 + (self.N - n_qi + 0.5) / (n_qi + 0.5))

    def score_query(self, query: str, top_k: int = 10) -> List[ScoredChunk]:
        """
        Score all chunks for a given query and return top_k.

        This is a simple baseline: iterate over all docs.
        Good enough for small/medium corpora on a laptop.
        """
        q_tokens = tokenize(query)
        if not q_tokens or self.N == 0:
            return []

        scores: List[Tuple[int, float]] = []
        for i in range(self.N):
            tf = self.term_freqs[i]
            doc_len = self.doc_lens[i]

            score = 0.0
            for term in q_tokens:
                f = tf.get(term, 0)
                if f == 0:
                    continue

                idf = self._idf(term)
                if idf == 0.0:
                    continue

                denom = f + self.k1 * (
                    1.0 - self.b + self.b * doc_len / (self.avgdl or 1.0)
                )
                score += idf * (f * (self.k1 + 1.0) / denom)

            if score != 0.0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        scores = scores[:top_k]

        results: List[ScoredChunk] = []
        for idx, s in scores:
            chunk = self.chunks[idx]
            results.append(
                ScoredChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    score=float(s),
                    text=chunk.text,
                )
            )
        return results