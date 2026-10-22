from __future__ import annotations

import re
from dataclasses import dataclass

from fedproxy.federated.state import ParameterSpec


_BLOCK = re.compile(r"(?:^|\.)layers\.(\d+)\.")


@dataclass(frozen=True)
class BlockRegistry:
    spec: ParameterSpec
    blocks: tuple[int, ...]
    tensor_block: dict[str, int]
    tensor_factor: dict[str, str]

    @classmethod
    def from_spec(cls, spec: ParameterSpec) -> "BlockRegistry":
        tensor_block: dict[str, int] = {}
        tensor_factor: dict[str, str] = {}
        for entry in spec.entries:
            match = _BLOCK.search(entry.name)
            if match is None:
                raise ValueError(f"LoRA tensor has no decoder block: {entry.name}")
            factor = "A" if "lora_A" in entry.name else "B" if "lora_B" in entry.name else None
            if factor is None:
                raise ValueError(f"Non-LoRA tensor in adapter: {entry.name}")
            tensor_block[entry.name] = int(match.group(1))
            tensor_factor[entry.name] = factor
        blocks = tuple(sorted(set(tensor_block.values())))
        if blocks != tuple(range(len(blocks))):
            raise ValueError(f"Proxy block IDs must be contiguous from zero: {blocks}")
        return cls(spec, blocks, tensor_block, tensor_factor)

    def names_in(self, blocks: set[int] | frozenset[int]) -> list[str]:
        return [entry.name for entry in self.spec.entries if self.tensor_block[entry.name] in blocks]


@dataclass(frozen=True)
class ClientPlan:
    client_id: str
    round_id: int
    train_blocks: frozenset[int]
    upload_blocks: frozenset[int]
    anchor_blocks: frozenset[int]

    def validate(self, registry: BlockRegistry) -> None:
        known = set(registry.blocks)
        if not self.anchor_blocks or not self.train_blocks <= known:
            raise ValueError("Invalid train/anchor blocks")
        if not self.anchor_blocks <= self.upload_blocks <= self.train_blocks:
            raise ValueError("Expected anchors <= uploads <= trained blocks")
