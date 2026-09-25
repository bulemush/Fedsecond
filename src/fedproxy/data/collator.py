from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .tasks import TaskExample
from .truncation import PromptEncoding, encode_prompt


@dataclass
class ResponseOnlyCollator:
    tokenizer: object
    max_input_length: int = 64
    max_target_length: int = 128
    prompt_truncation_strategy: str = "prefix"
    input_truncation_side: str = "right"
    _prompt_cache: dict[tuple[str, str], PromptEncoding] = field(default_factory=dict, init=False, repr=False)

    def prepare_prompt(self, example: TaskExample) -> PromptEncoding:
        key = (example.task_name, example.prompt)
        prepared = self._prompt_cache.get(key)
        if prepared is None:
            prepared = encode_prompt(
                example,
                self.tokenizer,
                self.max_input_length,
                strategy=self.prompt_truncation_strategy,
                truncation_side=self.input_truncation_side,
            )
            self._prompt_cache[key] = prepared
        return prepared

    def _encode(self, example: TaskExample):
        prompt = self.prepare_prompt(example).input_ids
        target = self.tokenizer.encode(example.response, add_special_tokens=False)
        eos = self.tokenizer.eos_token_id
        target = target[: max(0, self.max_target_length - 1)] + [eos]
        if not target:
            raise ValueError("Every example must contain at least one supervised target token")
        ids = prompt + target
        labels = [-100] * len(prompt) + target
        return ids, labels

    def __call__(self, examples: list[TaskExample]):
        encoded = [self._encode(example) for example in examples]
        max_length = max(len(ids) for ids, _ in encoded)
        pad = self.tokenizer.pad_token_id
        input_ids, labels, attention = [], [], []
        for ids, labs in encoded:
            padding = max_length - len(ids)
            input_ids.append(ids + [pad] * padding)
            labels.append(labs + [-100] * padding)
            attention.append([1] * len(ids) + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }
