"""Tensor operations and properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "is_equal",
]


def is_equal(t1: Any, t2: Any) -> bool:
    """Check if two tensors are equal.

    Parameters
    ----------
    t1
        First tensor.
    t2
        Second tensor.
    """
    if isinstance(t1, np.ndarray | jnp.ndarray) and isinstance(t2, np.ndarray | jnp.ndarray):
        return np.array_equal(t1, t2)
    return t1 == t2
