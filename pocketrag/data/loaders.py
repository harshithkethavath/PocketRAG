from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Sequence

from .schemas import Document


def iter_text_files(
    root_dir: Path | str,
    suffixes: Sequence[str] = (".txt",),
) -> Iterable[Path]:
    """
    Recursively yield all text files under root_dir with given suffixes.
    """
    root = Path(root_dir)
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            yield path


def load_documents_from_dir(
    root_dir: Path | str,
    suffixes: Sequence[str] = (".txt",),
) -> List[Document]:
    """
    Load all text files under root_dir into Document objects.

    doc_id is the POSIX-style relative path from root_dir, e.g.:
    'topic1/file1.txt'
    """
    root = Path(root_dir).resolve()
    documents: List[Document] = []

    for file_path in iter_text_files(root, suffixes=suffixes):
        rel_path = file_path.relative_to(root).as_posix()
        doc_id = rel_path  # stable, human-readable ID

        text = file_path.read_text(encoding="utf-8", errors="ignore")
        title = file_path.stem

        metadata = {
            "source": "filesystem",
            "root_dir": str(root),
            "relative_path": rel_path,
            "suffix": file_path.suffix,
        }

        documents.append(
            Document(
                doc_id=doc_id,
                text=text,
                title=title,
                metadata=metadata,
            )
        )

    return documents