"""Tensor operations and properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from arrayer.typing import Array
from arrayer import exception

if TYPE_CHECKING:
    from typing import Any

__all__ = [
    "is_equal",
    "argin",
    "argin_single",
    "argin_batch",
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


def argin(
    element: Array,
    test_elements: Array,
    batch_ndim: int = 0,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> jnp.ndarray:
    """Get the index/indices of the first element(s) in `test_elements` that match the element(s) in `element`.

    Parameters
    ----------
    element
        Element(s) to match against.
    test_elements
        Array of elements to test against.
    batch_ndim
        Number of leading batch dimensions in `element`.
    rtol
        Relative tolerance for floating-point comparison.
    atol
        Absolute tolerance for floating-point comparison.
    equal_nan
        Treat NaNs in `element` and `test_elements` as equal.
    """
    if not isinstance(element, jax.Array):
        element = np.asarray(element)
    if not isinstance(test_elements, jax.Array):
        test_elements = np.asarray(test_elements)
    if np.issubdtype(test_elements.dtype, np.str_):
        max_len = max([np.max(np.strings.str_len(a)) for a in (element, test_elements)])
        element = str_to_int_array(element, max_len=max_len)
        test_elements = str_to_int_array(test_elements, max_len=max_len)
    if batch_ndim == 0:
        return argin_single(element, test_elements, rtol, atol, equal_nan)
    if batch_ndim == 1:
        return argin_batch(element, test_elements, rtol, atol, equal_nan)
    element_reshaped = element.reshape(-1, *element.shape[batch_ndim:])
    result = _argin_batch(element_reshaped, test_elements, rtol, atol, equal_nan)
    return result.reshape(*element.shape, -1).squeeze(axis=-1)


@jax.jit
def argin_single(
    element: Array,
    test_elements: Array,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> jnp.ndarray:
    """Get the index of the first element in `test_elements` that matches `element`.

    Parameters
    ----------
    element
        Element to match against.
        This can be a tensor of any shape.
    test_elements
        Array of elements to test against.
        This array must have at least one more dimension than `element`,
        and its trailing dimensions must match the shape of `element`.
    rtol
        Relative tolerance for floating-point comparison.
    atol
        Absolute tolerance for floating-point comparison.
    equal_nan
        Treat NaNs in `element` and `test_elements` as equal.

    Returns
    -------
    An integer array of shape `(test_elements.ndim - element.ndim,)`
    (or a 0D array if `test_elements` is 1D)
    representing the index of the matching element in `test_elements`.
    If no match is found, returns `-1` in all index components.

    Example
    -------
    >>> from arrayer.tensor import argin_single
    >>> argin_single(11, [10, 11, 12])
    Array(1, dtype=int32)
    >>> argin_single(11, [[10], [11], [12]])
    Array([1, 0], dtype=int32)
    >>> argin_single([1, 2], [[1, 2], [3, 4], [5, 6]])
    Array(0, dtype=int32)
    >>> argin_single([1, 2], [[3, 4], [5, 6]])
    Array(-1, dtype=int32)
    """
    element = jnp.asarray(element)
    test_elements = jnp.asarray(test_elements)
    n_batch_dims = test_elements.ndim - element.ndim
    if n_batch_dims < 1:
        raise exception.InputError(
            name="element",
            value=element,
            problem=f"`element` must have fewer dimensions than `test_elements`, "
                    f"but got {element.ndim}D `element` and {test_elements.ndim}D `test_elements`."
        )
    batch_shape = test_elements.shape[:n_batch_dims]
    flat_refs = test_elements.reshape(-1, *element.shape)
    condition = jnp.isclose(
        flat_refs,
        element,
        rtol=rtol,
        atol=atol,
        equal_nan=equal_nan,
    ) if jnp.issubdtype(test_elements.dtype, jnp.floating) else flat_refs == element
    mask = jnp.all(condition, axis=tuple(range(1, flat_refs.ndim)))
    flat_idx = jnp.argmax(mask)
    found = mask[flat_idx]
    unraveled = jnp.stack(jnp.unravel_index(flat_idx, batch_shape)).squeeze()
    return jax.lax.select(found, unraveled, -jnp.ones_like(unraveled))


@jax.jit
def argin_batch(
    element: Array,
    test_elements: Array,
    rtol: float = 1e-6,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> jnp.ndarray:
    """Get the indices of the first elements in `test_elements` that match the elements in `element`.

    Parameters
    ----------
    element
        Elements to match against,
        as an array of shape `(n_elements, *element_shape)`.
    test_elements
        Array of elements to test against.
        This array must have at least the same number of dimensions as `element`,
        and its trailing dimensions must match `element_shape`.
    rtol
        Relative tolerance for floating-point comparison.
    atol
        Absolute tolerance for floating-point comparison.
    equal_nan
        Treat NaNs in `element` and `test_elements` as equal.

    Returns
    -------
    An integer array of shape `(n_elements, test_elements.ndim - element.ndim + 1,)`
    (or a 1D array of size `n_elements` if `test_elements` is 1D)
    representing the indices of the matching elements in `test_elements`.
    If no match is found, returns `-1` in all index components.
    """
    return _argin_batch(element, test_elements, rtol, atol, equal_nan)


def str_to_int_array(str_array, max_len: int | None = None):
    """Convert an array of strings to an array of integers."""
    input_is_string = isinstance(str_array, str)
    if input_is_string:
        str_array = [str_array]
    arr = np.array(str_array, dtype=f"<U{max_len}" if max_len else None)
    int_array = arr[..., None].view(dtype=(str, 1)).view(dtype=np.uint32)
    if input_is_string:
        return int_array.squeeze(axis=0)
    return int_array


_argin_batch = jax.vmap(argin_single, in_axes=(0, None, None, None, None))
