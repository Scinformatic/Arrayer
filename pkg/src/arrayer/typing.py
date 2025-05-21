from functools import partial
from typing import TypeAlias

import jax
import numpy as np
from beartype import beartype
from jaxtyping import jaxtyped, Num

typecheck = beartype
atypecheck = partial(jaxtyped, typechecker=beartype)

Array: TypeAlias = jax.Array | np.ndarray
JAXArray: TypeAlias = jax.Array
