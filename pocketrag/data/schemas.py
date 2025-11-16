from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional


@dataclass
class Document:
    """
    A single raw document in the corpus.

    - doc_id: stable ID used everywhere else (indexes, metrics, etc.).
    - text: full text of the document.
    - title: optional human-readable title.
    - metadata: arbitrary key/value pairs (e.g., source, path).
    """
    doc_id: str
    text: str
    title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DocumentChunk:
    """
    A contiguous chunk of a Document.

    - chunk_id: globally unique chunk ID.
    - doc_id: back-reference to the original document.
    - position: 0-based index of chunk in its document.
    - text: text content of the chunk.
    - start_char / end_char: optional char offsets in original document.
    - metadata: arbitrary per-chunk metadata.
    """
    chunk_id: str
    doc_id: str
    position: int
    text: str
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)