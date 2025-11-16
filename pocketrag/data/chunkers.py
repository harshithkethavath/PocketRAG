from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, List

from .schemas import Document, DocumentChunk


class BaseChunker(ABC):
    """
    Abstract base class for chunkers.
    """

    @abstractmethod
    def chunk_document(self, doc: Document) -> List[DocumentChunk]:
        """
        Turn a Document into a list of DocumentChunk objects.
        """
        raise NotImplementedError


class FixedSizeWordChunker(BaseChunker):
    """
    Simple word-based chunker.

    - Splits on whitespace into tokens.
    - Groups into windows of `max_words`, with `overlap` words overlap
      between consecutive chunks.
    - Does NOT do anything fancy with sentences or tokens; that's for later.
    """

    def __init__(self, max_words: int = 200, overlap: int = 40):
        assert max_words > 0, "max_words must be > 0"
        assert 0 <= overlap < max_words, "overlap must be >= 0 and < max_words"
        self.max_words = max_words
        self.overlap = overlap

    def chunk_document(self, doc: Document) -> List[DocumentChunk]:
        words = doc.text.split()
        n = len(words)

        if n == 0:
            return []

        chunks: List[DocumentChunk] = []
        start_idx = 0
        position = 0

        step = self.max_words - self.overlap

        while start_idx < n:
            end_idx = min(start_idx + self.max_words, n)
            chunk_words = words[start_idx:end_idx]
            chunk_text = " ".join(chunk_words)

            chunk_id = f"{doc.doc_id}::chunk-{position}"

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc.doc_id,
                position=position,
                text=chunk_text,
                start_char=None,
                end_char=None,
                metadata={
                    "max_words": self.max_words,
                    "overlap": self.overlap,
                    "start_word_index": start_idx,
                    "end_word_index": end_idx,
                },
            )
            chunks.append(chunk)

            position += 1
            start_idx += step

        return chunks


def chunk_corpus(
    documents: Iterable[Document],
    chunker: BaseChunker,
) -> List[DocumentChunk]:
    """
    Apply the chunker to all documents and flatten the result.
    """
    all_chunks: List[DocumentChunk] = []
    for doc in documents:
        doc_chunks = chunker.chunk_document(doc)
        all_chunks.extend(doc_chunks)
    return all_chunks