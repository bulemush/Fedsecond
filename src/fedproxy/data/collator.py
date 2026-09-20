from __future__ import annotations

from dataclasses import dataclass

import torch

from .tasks import TaskExample


@dataclass
class ResponseOnlyCollator:
    tokenizer: object
    max_input_length: int = 64
    max_target_length: int = 128

    def _encode(self, example: TaskExample):
        prompt = self.tokenizer.encode(example.prompt, add_special_tokens=True)[: self.max_input_length]
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

