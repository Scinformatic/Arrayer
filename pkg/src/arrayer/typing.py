from functools import partial
from typing import TypeAlias

import jax
import numpy as np
from beartype import beartype
from beartype.vale import Is
from jaxtyping import jaxtyped, Num, Bool, Float


__all__ = [
    "Num",
    "Bool",
    "Float",
    "typecheck",
    "atypecheck",
    "Array",
    "JAXArray",
]


typecheck = beartype
atypecheck = partial(jaxtyped, typechecker=beartype)

Array: TypeAlias = jax.Array | np.ndarray
JAXArray: TypeAlias = jax.Array
