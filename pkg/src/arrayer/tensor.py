"""Tensor operations and properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax
import jax.numpy as jnp

from arrayer.typing import Array
from arrayer import exception

if TYPE_CHECKING:
    from typing import Any, Sequence

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
    t1_is_array = isinstance(t1, np.ndarray | jax.Array)
    t2_is_array = isinstance(t2, np.ndarray | jax.Array)
    if t1_is_array and t2_is_array:
        return np.array_equal(t1, t2)
    if any((t1_is_array, t2_is_array)):
        # If one is an array and the other is not, they cannot be equal
        return False
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


def ensure_padding(
    arr: Array,
    padding: int | Sequence[int | Sequence[int]],
    spatial_axes: Sequence[int]
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    """En

    Ensure there are exactly the requested falsey planes on each side of each
    spatial axis, cropping or padding with 0/False as needed.

    Parameters
    ----------
    arr : np.ndarray
        N-D boolean or integer array.
    padding : int or sequence
        If an int, that padding is used on both low/high sides of every spatial axis.
        If a sequence of length len(spatial_axes), each element must be either:
          - an int (same padding on low/high for that axis), or
          - a sequence of two ints (low, high padding for that axis).
    spatial_axes : Sequence[int]
        Indices of the axes in `arr` which are spatial. Must be unique
        and valid in [-arr.ndim, arr.ndim).

    Returns
    -------
    out : np.ndarray
        Cropped/padded array, same dtype as `arr`.
    deltas : Dict[int, Tuple[int, int]]
        Mapping each spatial axis → (low_delta, high_delta),
        where positive = planes added, negative = planes removed.

    Raises
    ------
    ValueError
        If padding is invalid, or spatial_axes are invalid/duplicated,
        or padding sequence length mismatches spatial_axes.
    """
    # --- Validate spatial_axes ---
    ndim = arr.ndim
    axes = []
    for ax in spatial_axes:
        ax_norm = ax if ax >= 0 else ndim + ax
        if ax_norm < 0 or ax_norm >= ndim:
            raise ValueError(f"Spatial axis {ax!r} out of range for array with ndim={ndim}")
        axes.append(ax_norm)
    if len(set(axes)) != len(axes):
        raise ValueError(f"Duplicate axes in spatial_axes: {spatial_axes!r}")

    k = len(axes)
    if k == 0:
        raise ValueError("Must specify at least one spatial axis.")

    # --- Parse padding argument into per-axis (low, high) tuples ---
    def make_pad_pair(x) -> Tuple[int,int]:
        if isinstance(x, int):
            return x, x
        if (isinstance(x, Sequence) and len(x) == 2
            and all(isinstance(i, int) for i in x)):
            return x[0], x[1]
        raise ValueError(
            "Each padding element must be int or 2-tuple of ints, got: "
            f"{x!r}"
        )

    if isinstance(padding, int):
        pad_pairs = [(padding, padding)] * k
    else:
        pad_list = list(padding)
        if len(pad_list) != k:
            raise ValueError(
                f"Padding sequence length {len(pad_list)} does not match "
                f"number of spatial axes {k}"
            )
        pad_pairs = [make_pad_pair(x) for x in pad_list]

    # --- Build the k-D truth mask by collapsing non-spatial dims ---
    non_axes = tuple(sorted(set(range(ndim)) - set(axes)))
    truth_mask = arr.astype(bool).any(axis=non_axes)

    # --- Helper to find first/last True along each mask axis ---
    def bounds(mask: np.ndarray, axis: int) -> Tuple[int, int]:
        proj = mask.any(axis=tuple(i for i in range(mask.ndim) if i != axis))
        idx = np.nonzero(proj)[0]
        return (int(idx[0]), int(idx[-1])) if idx.size else (0, -1)

    # --- Compute mins, maxs, and original sizes for each spatial dim ---
    mins, maxs, orig = [], [], list(truth_mask.shape)
    for i in range(k):
        lo, hi = bounds(truth_mask, i)
        mins.append(lo)
        maxs.append(hi)

    # --- Determine actual crop vs pad amounts per axis side ---
    low_crop = [0]*k
    high_crop = [0]*k
    low_pad  = [0]*k
    high_pad = [0]*k

    for i in range(k):
        size = orig[i]
        lo_pad_target, hi_pad_target = pad_pairs[i]
        want_lo = mins[i] - lo_pad_target
        want_hi_excl = maxs[i] + 1 + hi_pad_target

        # crop if want_lo >= 0, else pad
        if want_lo >= 0:
            low_crop[i] = want_lo
        else:
            low_pad[i] = -want_lo

        # crop or pad on high end
        if want_hi_excl <= size:
            high_crop[i] = size - want_hi_excl
        else:
            high_pad[i] = want_hi_excl - size

    # --- Build deltas mapping back to original array axes ---
    deltas: Dict[int, Tuple[int,int]] = {}
    for idx, arr_ax in enumerate(axes):
        delta_lo = low_pad[idx]  - low_crop[idx]
        delta_hi = high_pad[idx] - high_crop[idx]
        deltas[arr_ax] = (delta_lo, delta_hi)

    # --- Crop the array ---
    slicer = []
    for dim in range(ndim):
        if dim in axes:
            i = axes.index(dim)
            start = low_crop[i]
            end   = orig[i] - high_crop[i]
            slicer.append(slice(start, end if end != 0 else None))
        else:
            slicer.append(slice(None))
    cropped = arr[tuple(slicer)]

    # --- Pad the cropped array ---
    pad_width = []
    for dim in range(ndim):
        if dim in axes:
            i = axes.index(dim)
            pad_width.append((low_pad[i], high_pad[i]))
        else:
            pad_width.append((0, 0))

    out = np.pad(cropped, pad_width, mode="constant", constant_values=0)
    return out, deltas


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
