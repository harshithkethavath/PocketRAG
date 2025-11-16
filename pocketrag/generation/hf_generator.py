from __future__ import annotations

from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseGenerator, GenerationConfig
from ..utils.hardware import pick_device


class LocalHFGenerator(BaseGenerator):
    """
    Local HuggingFace causal LM generator.

    Default model is TinyLlama (small enough for a laptop), but you can
    override via GenerationConfig.
    """

    def __init__(self, config: Optional[GenerationConfig] = None):
        self.config = config or GenerationConfig()
        self.device = pick_device(self.config.device_pref)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        torch_dtype = (
            torch.float16
            if self.device in ("cuda", "mps")
            else torch.float32
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch_dtype,
        ).to(self.device)
        self.model.eval()

    def generate(self, prompt: str) -> str:
        """
        Basic text generation: single-turn prompt -> completion.
        """
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
        )
        input_ids = inputs["input_ids"].to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=(self.config.temperature > 0.0),
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = output_ids[0, input_ids.shape[1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()