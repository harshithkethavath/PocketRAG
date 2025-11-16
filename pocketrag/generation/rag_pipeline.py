from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List

from ..retrieval import ScoredChunk
from .base import BaseGenerator


RetrieveFn = Callable[[str, int], List[ScoredChunk]]


@dataclass
class RAGAnswer:
    question: str
    answer: str
    retrieved: List[ScoredChunk]
    prompt: str


class RAGPipeline:
    """
    Minimal RAG pipeline:
    - retrieve_fn: query -> top-k ScoredChunk
    - generator: BaseGenerator (LLM)
    """

    def __init__(
        self,
        retrieve_fn: RetrieveFn,
        generator: BaseGenerator,
        top_k: int = 5,
        max_context_chars: int = 4000,
    ):
        self.retrieve_fn = retrieve_fn
        self.generator = generator
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    def _build_prompt(self, question: str, contexts: List[ScoredChunk]) -> str:
        blocks = []
        for r in contexts:
            blocks.append(
                f"[doc_id={r.doc_id} | chunk_id={r.chunk_id} | score={r.score:.4f}]\n"
                f"{r.text}"
            )

        context_text = "\n\n".join(blocks)
        if len(context_text) > self.max_context_chars:
            context_text = context_text[: self.max_context_chars] + "\n...[truncated]"

        prompt = (
            "You are a helpful assistant answering questions based ONLY on the "
            "provided context. If the answer is not present in the context, say "
            "\"I don't know based on the given context.\".\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        return prompt

    def answer(self, question: str) -> RAGAnswer:
        retrieved = self.retrieve_fn(question, self.top_k)
        prompt = self._build_prompt(question, retrieved)
        answer = self.generator.generate(prompt)
        return RAGAnswer(
            question=question,
            answer=answer,
            retrieved=retrieved,
            prompt=prompt,
        )