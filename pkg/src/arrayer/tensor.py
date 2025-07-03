"""Tensor operations and properties."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
import jax
import jax.numpy as jnp

from arrayer.typing import Array
from arrayer import exception

if TYPE_CHECKING:
    from typing import Any, Sequence
    from arrayer.typing import JAXArray

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
