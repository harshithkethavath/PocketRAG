from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from sentence_transformers import SentenceTransformer

from ..utils.hardware import pick_device


@dataclass
class EmbeddingConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device_pref: str = "auto"
    batch_size: int = 32


class EmbeddingModel:
    """
    Thin wrapper around SentenceTransformer for sentence embeddings.

    We make this a class so:
    - it can be reused across multiple queries without re-loading weights
    - we control device placement (cpu / cuda / mps)
    """

    def __init__(self, config: EmbeddingConfig | None = None):
        self.config = config or EmbeddingConfig()
        self.device = pick_device(self.config.device_pref)

        self.model = SentenceTransformer(self.config.model_name, device=self.device)

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def encode(
        self,
        texts: List[str],
        batch_size: int | None = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """
        Encode a list of texts into a [N, D] float32 tensor on CPU.

        We ALWAYS return CPU tensors so they can be saved easily; the model
        still runs on the selected device internally.
        """
        bs = batch_size or self.config.batch_size

        with torch.inference_mode():
            emb = self.model.encode(
                texts,
                batch_size=bs,
                convert_to_tensor=True,
                normalize_embeddings=normalize,
            )
            
        emb = emb.to(device="cpu", dtype=torch.float32)
        return emb