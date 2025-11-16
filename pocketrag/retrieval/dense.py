from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch

from ..data import DocumentChunk
from .embeddings import EmbeddingModel, EmbeddingConfig
from .bm25 import ScoredChunk  # reuse the same scored result type


@dataclass
class DenseIndex:
    """
    Simple dense retrieval index.

    - embeddings: [N, D] float32 tensor on CPU
    - chunk_ids / doc_ids / texts: aligned metadata lists
    - model_name: sentence embedding model that produced the vectors
    """

    embeddings: torch.Tensor
    chunk_ids: List[str]
    doc_ids: List[str]
    texts: List[str]
    model_name: str

    # ---------- Construction ----------

    @classmethod
    def from_chunks(
        cls,
        chunks: List[DocumentChunk],
        encoder: EmbeddingModel,
        batch_size: int | None = None,
    ) -> "DenseIndex":
        if not chunks:
            raise ValueError("Cannot build DenseIndex from empty chunk list.")

        texts = [c.text for c in chunks]
        chunk_ids = [c.chunk_id for c in chunks]
        doc_ids = [c.doc_id for c in chunks]

        embeddings = encoder.encode(texts, batch_size=batch_size, normalize=True)

        return cls(
            embeddings=embeddings,
            chunk_ids=chunk_ids,
            doc_ids=doc_ids,
            texts=texts,
            model_name=encoder.model_name,
        )

    # ---------- Persistence ----------

    def save(self, path: Path | str) -> None:
        """
        Save the index using torch.save. This is fine for small/medium corpora.
        """
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)

        obj = {
            "embeddings": self.embeddings,
            "chunk_ids": self.chunk_ids,
            "doc_ids": self.doc_ids,
            "texts": self.texts,
            "model_name": self.model_name,
        }
        torch.save(obj, p)

    @classmethod
    def load(cls, path: Path | str) -> "DenseIndex":
        p = Path(path).expanduser().resolve()
        obj = torch.load(p, map_location="cpu")

        return cls(
            embeddings=obj["embeddings"],
            chunk_ids=list(obj["chunk_ids"]),
            doc_ids=list(obj["doc_ids"]),
            texts=list(obj["texts"]),
            model_name=obj["model_name"],
        )

    # ---------- Search ----------

    def search(
        self,
        query: str,
        encoder: EmbeddingModel,
        top_k: int = 10,
    ) -> List[ScoredChunk]:
        """
        Brute-force dense search (cosine similarity) over all chunks.

        - Embeddings are L2-normalized at build time, and we normalize
          the query as well, so dot product == cosine similarity.
        """
        if self.embeddings.numel() == 0:
            return []

        # Safety check: warn if model names differ (but still allow)
        if encoder.model_name != self.model_name:
            # You could raise here; for now we just print a warning.
            print(
                f"[warning] Query encoder model ({encoder.model_name}) "
                f"differs from index model ({self.model_name}). "
                "Results may be meaningless."
            )

        with torch.inference_mode():
            q_emb = encoder.encode([query], normalize=True)[0]  # [D]
            # [N, D] · [D] -> [N]
            scores = torch.matmul(self.embeddings, q_emb)
            scores = scores.tolist()

        # Get top_k indices
        idxs = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )[:top_k]

        results: List[ScoredChunk] = []
        for i in idxs:
            results.append(
                ScoredChunk(
                    chunk_id=self.chunk_ids[i],
                    doc_id=self.doc_ids[i],
                    score=float(scores[i]),
                    text=self.texts[i],
                )
            )
        return results


def build_dense_index_from_chunks_file(
    chunks_file: Path | str,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    device_pref: str = "auto",
    batch_size: int = 32,
) -> DenseIndex:
    """
    Convenience helper for CLI: loads chunks, builds encoder, builds index.
    """
    from ..data import load_chunks_from_jsonl  # local import to avoid cycles

    cf = Path(chunks_file).expanduser().resolve()
    chunks = load_chunks_from_jsonl(cf)

    encoder = EmbeddingModel(
        EmbeddingConfig(
            model_name=model_name,
            device_pref=device_pref,
            batch_size=batch_size,
        )
    )

    index = DenseIndex.from_chunks(
        chunks=chunks,
        encoder=encoder,
        batch_size=batch_size,
    )
    return index