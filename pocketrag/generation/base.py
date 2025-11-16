from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..utils.hardware import pick_device


@dataclass
class GenerationConfig:
    model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    device_pref: str = "auto"
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Generate a response string for a given prompt.
        """
        raise NotImplementedError