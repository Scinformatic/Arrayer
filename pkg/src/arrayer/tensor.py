"""Tensor operations and properties."""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from arrayer.typing import Array
from arrayer import exception

if TYPE_CHECKING:
    from typing import Any, Sequence, Literal
    from arrayer.typing import JAXArray

__all__ = [
    "is_equal",
    "argin",
    "argin_single",
    "argin_batch",
]

# @jax.jit(static_argnums=(1, 2, 3, 5))
def indices_sorted_by_value(
    tensor: Array,
    first: Literal["min", "max"] = "min",
    threshold: float | None = None,
    include_equal: bool = True,
    mask: Array | None = None,
    max_elements: int | None = None,
) -> JAXArray:
    """Get the indices of tensor elements sorted by their value.

    Parameters
    ----------
    tensor
        Input tensor.
    first
        - "min": Sort indices from smallest to largest values.
        - "max": Sort indices from largest to smallest values.
    threshold
        Optional threshold value; if provided,
        - for "min", only indices of elements equal to or below this value are returned.
        - for "max", only indices of elements equal to or above this value are returned.
    include_equal
        Only applicable if `threshold` is provided:
        - If `True`, the returned indices will include those where the tensor value
          is equal to the threshold.
        - If `False`, only indices strictly below (for "min") or above (for "max")
          the threshold are returned.
    mask
        Optional boolean mask to apply to the tensor.
        If provided, only indices of elements where `mask` is `True` are returned.
        The mask must have the same shape as the input `tensor`.
    max_elements
        Optional maximum number of indices to return.
        If provided, the function will return at most this many indices.

    Returns
    -------
    An array of shape `(n_elements, tensor.ndim)`
    containing the sorted indices of the elements in the tensor.
    """
    tensor = jnp.asarray(tensor)

    # Validate inputs
    if mask is not None:
        mask = jnp.asarray(mask).astype(bool)
        if mask.shape != tensor.shape:
            raise ValueError(f"mask shape {mask.shape} does not match tensor shape {tensor.shape}")
    if first not in ("min", "max"):
        raise ValueError(f"Invalid value for `first`: {first}. Must be 'min' or 'max'.")
    if max_elements is not None and max_elements < 0:
        raise ValueError(f"max_elements must be non-negative, got {max_elements}.")

    # Flatten tensor and (optional) mask
    tensor_flat = tensor.ravel()
    n_elements = tensor_flat.size
    mask_flat = jnp.ones(n_elements, dtype=bool) if mask is None else mask.ravel()

    # Add threshold filter to mask if provided
    if threshold is not None:
        if first == "min":
            mask_threshold = (
                (tensor_flat <= threshold)
                if include_equal else
                (tensor_flat < threshold)
            )
        else:
            mask_threshold = (
                (tensor_flat >= threshold)
                if include_equal else
                (tensor_flat > threshold)
            )
        mask_flat = jnp.logical_and(mask_flat, mask_threshold)

    # Sort by value
    valid_values = tensor_flat[mask_flat]
    n_valid_elements = valid_values.size
    values_to_sort = valid_values if first == "min" else -valid_values
    if max_elements is None or max_elements >= n_elements:
        order = jnp.argsort(values_to_sort)
    else:
        part = jnp.argpartition(values_to_sort, max_elements)
        topk = part[:max_elements]
        order = topk[jnp.argsort(values_to_sort[topk])]

    valid_indices = jnp.arange(n_elements)[mask_flat]
    sorted_indices = valid_indices[order]

    # Convert flat indices back to multi-dimensional indices
    return jnp.stack(jnp.unravel_index(sorted_indices, tensor.shape), axis=-1)


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


def pairwise_allclose(
    array: np.ndarray,
    tolerance: float | None = None,
    epsilon_factor: float = 1,
) -> bool:
    """Check whether all elements of `array` lie within a given tolerance of each other.

    In other words, check if `max(array) - min(array) <= tolerance`.
    If `tolerance` is not provided:
    - For integer arrays, requires exact equality (i.e. `tolerance = 0`).
    - For floating-point arrays, calculates the tolerance as
      `epsilon_factor * epsilon * max(|max(array)|, |min(array)|, 1.0)`
      where `epsilon` is the machine-precision epsilon for `array.dtype`.

    Parameters
    ----------
    array
        Input array to check.
    tolerance
        Optional absolute tolerance.
        If not provided, the tolerance is calculated
        based on the array's dtype.
    epsilon_factor
        Factor to multiply the machine epsilon by
        when calculating the tolerance for floating-point arrays.

    Returns
    -------
    Boolean indicating whether all elements
    are within the specified tolerance of each other.
    """
    array = np.asarray(array)
    if array.size == 0:
        return True
    max_val = array.max()
    min_val = array.min()
    span = max_val - min_val
    if tolerance is None:
        # no user tolerance → auto for floats, exact for ints
        if not issubclass(array.dtype.type, np.inexact):
            # integer (or exact) types must match exactly
            return bool(span == 0)
        eps = np.finfo(array.dtype).eps
        scale = max(abs(max_val), abs(min_val), 1.0)
        tolerance = epsilon_factor * eps * scale
    return span <= tolerance


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
    tensor: Array,
    axes: Sequence[int],
    padding: int | Sequence[int | Sequence[int]],
    pad_value: Any = 0
) -> tuple[JAXArray, list[tuple[int, int]]]:
    """Ensure padding around the minimal bounding box of non-pad values in array.

    This function ensures exactly the requested number of planes of `pad_value`
    around the minimum axis-aligned bounding box of all values that are not `pad_value`
    along each given axis, cropping or padding with `pad_value` as needed.

    Parameters
    ----------
    arr
        N-D input array to pad.
    axes
        Index of axes to pad.
        Indices must be unique and in the range `[-arr.ndim, arr.ndim]`.
    padding
        Number of required padding planes.
        If a single integer, it applies to all axes/sides.
        If a sequence, it must match the length of `axes`,
        where each element specifies the padding for the corresponding axis.
        Each element can be an integer (same padding on both directions along that axis)
        or a 2-tuple of integers (low, high) specifying the padding
        on the low and high sides of that axis.
    pad_value
        Fill value for all newly added planes (and treated as background).

    Returns
    -------
    out
        Cropped/padded array (same dtype as `arr`, upcast if needed).
    deltas
        Number of added (positive) or removed (negative) planes
        along each side (low, high) of each axis in `axes`.

    Examples
    --------
    >>> import numpy as np
    >>> from arrayer.tensor import ensure_padding
    >>> arr = np.array([[1, 2, 3], [4, 5, 6]])
    >>> padded, deltas = ensure_padding(arr, axes=[0, 1], padding=1, pad_value=0)
    >>> print(padded)
    [[0 0 0 0 0]
     [0 1 2 3 0]
     [0 4 5 6 0]
     [0 0 0 0 0]]
    >>> print(deltas)
    [(1, 1), (1, 1)]
    """
    def process_input_axes():
        """Normalize and validate input axes."""
        norm_axes = []
        for axis in axes:
            norm_axis = axis if axis >= 0 else ndim + axis
            if norm_axis < 0 or norm_axis >= ndim:
                raise ValueError(f"Axis {axis!r} out of range for ndim={ndim}")
            norm_axes.append(norm_axis)
        if len(set(norm_axes)) != len(norm_axes):
            raise ValueError(f"Duplicate axes: {axes!r}")
        if not norm_axes:
            raise ValueError("Must specify at least one spatial axis.")
        return norm_axes

    def process_input_padding():
        def to_pair(x):
            """Convert a padding element to a (low, high) pair."""
            if isinstance(x, int):
                return x, x
            if (
                isinstance(x, Sequence) and len(x) == 2
                and all(isinstance(i, int) for i in x)
            ):
                return x[0], x[1]
            raise ValueError(f"Invalid padding element {x!r}")

        if isinstance(padding, int):
            pads = [(padding, padding)] * n_axes
        else:
            p_list = list(padding)
            if len(p_list) != n_axes:
                raise ValueError(f"Padding length {len(p_list)} != #axes {n_axes}")
            pads = [to_pair(x) for x in p_list]
        return pads

    def bounds(mask: np.ndarray, axis: int) -> tuple[int, int]:
        """Find the bounds of non-zero elements in mask along a given axis."""
        proj = mask.any(axis=tuple(i for i in range(mask.ndim) if i != axis))
        idx = np.nonzero(proj)[0]
        return (int(idx[0]), int(idx[-1])) if idx.size else (0, -1)

    ndim = tensor.ndim
    axes = process_input_axes()
    n_axes = len(axes)
    pads = process_input_padding()

    # Build truth mask, treating pad_value as background
    permuted = np.moveaxis(tensor, axes, range(tensor.ndim - n_axes, tensor.ndim))
    non_axes = tuple(range(tensor.ndim - n_axes))
    truth = (permuted != pad_value).any(axis=non_axes)
    orig = list(truth.shape)
    mins, maxs = zip(*(bounds(truth, i) for i in range(n_axes)))

    # Compute how much to crop vs pad on each side
    low_crop = [0] * n_axes
    high_crop = [0] * n_axes
    low_pad = [0] * n_axes
    high_pad = [0] * n_axes

    for i in range(n_axes):
        size = orig[i]
        lo_t, hi_t = pads[i]
        want_lo = mins[i] - lo_t
        want_hi = maxs[i] + 1 + hi_t
        if want_lo >= 0:
            low_crop[i] = want_lo
        else:
            low_pad[i] = -want_lo
        if want_hi <= size:
            high_crop[i] = size - want_hi
        else:
            high_pad[i] = want_hi - size

    # Build deltas per original axis
    deltas = [
        (low_pad[idx] - low_crop[idx], high_pad[idx] - high_crop[idx])
        for idx in range(n_axes)
    ]

    # Crop
    slicer = []
    for dim in range(ndim):
        if dim in axes:
            i = axes.index(dim)
            start = low_crop[i]
            stop  = orig[i] - high_crop[i]
            slicer.append(slice(start, stop))
        else:
            slicer.append(slice(None))
    cropped = tensor[tuple(slicer)]

    # Pad with pad_value
    pad_width = [(0,0)] * ndim
    for idx, ax in enumerate(axes):
        pad_width[ax] = (low_pad[idx], high_pad[idx])
    out = jnp.pad(
        cropped,
        pad_width,
        mode="constant",
        constant_values=pad_value
    )
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


def make_batches(
    tensor: Array,
    axis: int = 0,
    min_size: int = 50,
    max_size: int = 2000,
    grow_factor: float = 2.0,
) -> list[Array]:
    """Split a tensor into batches along a specified axis.

    This function is useful for processing large tensors
    in smaller chunks to avoid memory issues,
    e.g., for training models or
    performing computations that can be parallelized.

    Parameters
    ----------
    tensor
        Input tensor to be split into batches.
    axis
        Index of the axis along which to split the tensor into batches.
    min_size
        Minimum batch size.
    max_size
        Maximum batch size; batches will not exceed this many elements.
        Too small values result in too many Python loops,
        which can slow down the process,
        while too large values can lead to large memory allocations,
        which may also slow down or crash the process.
    grow_factor
        Factor by which the batch size grows.
        The first batch will be of size `min_size`,
        and subsequent batches will grow by this factor
        (i.e. each batch will be ca. `grow_factor` times larger than the previous one)
        until they reach `max_size`.

    Returns
    -------
    Tensor batches along the specified axis.
    """
    if min_size is None:
        return [tensor]
    n_elements = tensor.shape[axis]
    if n_elements <= min_size:
        return [tensor]
    split_indices = [min_size]
    split_size = min_size
    while n_elements - split_indices[-1] > max_size:
        split_size = min(int(np.rint(split_size * grow_factor)), max_size)
        split_indices.append(split_indices[-1] + split_size)
    return np.array_split(tensor, indices_or_sections=split_indices, axis=axis)


_argin_batch = jax.vmap(argin_single, in_axes=(0, None, None, None, None))




# Not ready functions, to be implemented later

def _filter_array(
    array: np.ndarray,
    minmax_vals: tuple[float, float],
    reduction_op: str,
    reduction_axis: int = -1,
    invert: bool = False,
) -> np.ndarray:
    """
    Filter points by their distances to their nearest atoms.

    Parameters
    ----------
    array
    dist_range : tuple[float, float]
        Lower and upper bounds of distance to the nearest atom.
    invert : bool, optional, default: False
        If False (default), the points whose nearest atoms lie within the distance range are
        returned. If True, the points whose nearest atoms lie outside the range are returned.

    Returns
    -------
    boolean_mask : ndarray
    """
    red_funcs_pre = {"max": np.max, "min": np.min, "avg": np.mean, "sum": np.sum}
    red_funcs_post = {"all": np.all, "any": np.any, "one": np.logical_xor.reduce}

    if reduction_op in red_funcs_pre:
        array_values = red_funcs_pre[reduction_op](array, axis=reduction_axis)
        array_mask = filter_array_by_range(
            array=array_values, minmax_vals=minmax_vals, invert=invert
        )
        return array_mask, (array_values.min(), array_values.max())
    if reduction_op in red_funcs_post:
        mask_array = filter_array_by_range(array=array, minmax_vals=minmax_vals, invert=invert)
        reduced_mask_array = red_funcs_post[reduction_op](mask_array, axis=reduction_axis)
        return reduced_mask_array, (array.min(), array().max())
    raise ValueError("Reduction operation not recognized.")


def _filter_array_by_range(
    array: np.ndarray, minmax_vals: tuple[float, float], invert: bool = False
):
    op_lower, op_upper = (operator.le, operator.ge) if invert else (operator.ge, operator.le)
    op_combine = np.logical_or if invert else np.logical_and
    mask = op_combine(
        op_lower(array, minmax_vals[0]),
        op_upper(array, minmax_vals[1]),
    )
    return mask
