"""Lossless little-endian FP32 payloads; signed zero is significant."""
from __future__ import annotations

import numpy as np


def encode_values(values: np.ndarray, *, elide_zero: bool = False,
                  elide_constant: bool = False) -> tuple[str, bytes]:
    values = np.asarray(values, dtype="<f4").reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite state value")
    bits = values.view("<u4")
    if elide_zero and len(bits) and np.all(bits == 0):
        return "zero", b""
    if elide_constant and len(bits) > 1 and np.all(bits == bits[0]):
        return "constant", values[:1].tobytes()
    return "fp32", values.tobytes()


def decode_values(raw: bytes, count: int, codec: str) -> np.ndarray:
    if count < 0:
        raise ValueError("Invalid value count")
    if codec == "zero" and not raw and count > 0:
        values = np.zeros(count, dtype="<f4")
    elif codec == "constant" and len(raw) == 4 and count > 1:
        # Repeat the original bits, including -0, without numeric conversion.
        values = np.full(count, np.frombuffer(raw, dtype="<u4")[0], dtype="<u4").view("<f4")
    elif codec == "fp32" and len(raw) == 4 * count:
        values = np.frombuffer(raw, dtype="<f4").copy()
    else:
        raise ValueError("Invalid value encoding or length")
    if not np.isfinite(values).all():
        raise ValueError("Non-finite state value")
    return values
