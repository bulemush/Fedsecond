from __future__ import annotations

import json
from pathlib import Path


def evaluate_with_lm_eval(model_path: str, tasks: list[str], output_path: str | Path, batch_size: int = 1, num_fewshot: int = 0):
    from lm_eval import evaluator

    results = evaluator.simple_evaluate(
        model="hf",
        model_args=f"pretrained={model_path}",
        tasks=tasks,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    return results

