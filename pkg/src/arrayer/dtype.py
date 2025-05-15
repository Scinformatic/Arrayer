"""Functionalities to determine suitable NumPy dtypes."""

import numpy as np


def smallest_integer(minimum: int | float, maximum: int | float) -> type[np.integer]:
    """Get the smallest NumPy integer data type that can hold the given range.

    Parameters
    ----------
    minimum
        Minimum value of the range.
    maximum
        Maximum value of the range.
    """
    unsigned_types = [np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong]
    unsigned_maxes = [np.iinfo(dtype).max for dtype in unsigned_types]

    signed_types = [np.byte, np.short, np.intc, np.int_, np.longlong]
    signed_mins = [np.iinfo(dtype).min for dtype in signed_types]
    signed_maxes = [np.iinfo(dtype).max for dtype in signed_types]

    if not np.issubdtype(type(minimum), np.integer) or not np.issubdtype(type(maximum), np.integer):
        raise ValueError(
            "Parameters `min_val` and `max_val` expect integer values. "
            f"Input was {minimum, maximum}"
        )
    if minimum > maximum:
        raise ValueError(
            f"`min_val` must be smaller than `max_val`. Input was: {minimum}, {maximum}"
        )

    if minimum >= 0:
        for idx_dtype, dtype in enumerate(unsigned_types):
            if maximum <= unsigned_maxes[idx_dtype]:
                return dtype
        raise ValueError(
            f"Bit overflow. Bounds for largest type ({[unsigned_types[-1]]}) is "
            f"[0, {unsigned_maxes[-1]}]. Given interval was [{minimum}, {maximum}]."
        )
    for idx_dtype, dtype in enumerate(signed_types):
        if minimum >= signed_mins[idx_dtype] and maximum <= signed_maxes[idx_dtype]:
            return dtype
    raise ValueError(
        f"Bit overflow. Bounds for largest type ({[signed_types[-1]]}) is "
        f"[{signed_mins[-1]}, {signed_maxes[-1]}]. Given interval was [{minimum}, {maximum}]."
    )
