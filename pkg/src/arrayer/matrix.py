"""Matrix operations and properties."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from arrayer import exception

if TYPE_CHECKING:
    from typing import Callable, Any
    from numpy.typing import ArrayLike

__all__ = [
    "is_rotation",
    "is_orthogonal",
    "has_unit_determinant",
    "is_rotation_single",
    "is_orthogonal_single",
    "has_unit_determinant_single",
]


def is_rotation(matrix: ArrayLike, tol: float = 1e-8) -> bool | jnp.ndarray:
    """Check whether the input represents a pure rotation matrix (or batch thereof).

    This is done by checking whether the matrix is
    both orthogonal and has a determinant of +1
    within a numerical tolerance.

    Parameters
    ----------
    matrix
        One of:
        - Single matrix of shape `(n_dims, n_dims)`.
        - Batch of matrices with shape `(n_batches, n_dims, n_dims)`.

        Each matrix is expected to be real-valued and square.
    tol
        Absolute tolerance used for both orthogonality and determinant tests.
        This threshold defines the allowed numerical deviation
        from perfect rotation properties.
        This should be a small positive float, e.g., 1e-8.

    Returns
    -------
    - For a single matrix input: a Python `bool` indicating whether the matrix is a pure rotation matrix.
    - For a batch of matrices: a JAX boolean array of shape `(n_batches,)`
      where each element indicates whether the corresponding matrix is a pure rotation matrix.

    Raises
    ------
    arrayer.exception.InputError
        If the input array is not 2D/3D,
        or if the matrices are not square.

    Notes
    -----
    A rotation matrix is a square matrix that represents a rigid-body rotation
    in Euclidean space, preserving the length of vectors and angles between them
    (i.e., no scaling, shearing, or reflection).
    A matrix is a rotation matrix if it is orthogonal
    ($R^\\top R \\approx I$) and has determinant +1,
    meaning it preserves both length/angle and orientation.

    References
    ----------
    - [Rotation matrix - Wikipedia](https://en.wikipedia.org/wiki/Rotation_matrix)
    """
    return _is_rotation(matrix, tol)


def is_orthogonal(matrix: ArrayLike, tol: float = 1e-8) -> bool | jnp.ndarray:
    """Check whether the input represents an orthogonal matrix (or batch thereof).

    This is done by checking whether the transpose of the matrix
    multiplied by the matrix itself yields the identity matrix
    within a numerical tolerance, i.e., $R^\\top R \\approx I$.

    Parameters
    ----------
    matrix
        One of:
        - Single matrix of shape `(n_dims, n_dims)`.
        - Batch of matrices with shape `(n_batches, n_dims, n_dims)`.

        Each matrix is expected to be real-valued and square.
    tol
        Absolute tolerance for comparison against the identity matrix.
        This should be a small positive float, e.g., 1e-8.

    Returns
    -------
    - For a single matrix input: a Python `bool` indicating whether the matrix is orthogonal.
    - For a batch of matrices: a JAX boolean array of shape `(n_batches,)`
      where each element indicates whether the corresponding matrix is orthogonal.

    Raises
    ------
    arrayer.exception.InputError
        If the input array is not 2D/3D,
        or if the matrices are not square.
    """
    return _is_orthogonal(matrix, tol)


def has_unit_determinant(matrix: ArrayLike, tol: float = 1e-8) -> bool | jnp.ndarray:
    """Check whether the input represents a matrix (or batch thereof) with determinant approximately +1, i.e., $\\det(R) \\approx 1$.

    This can be used to test if a transformation matrix
    preserves orientation and volume,
    as required for a proper rotation.

    Parameters
    ----------
    matrix
        One of:
        - Single matrix of shape `(n_dims, n_dims)`.
        - Batch of matrices with shape `(n_batches, n_dims, n_dims)`.

        Each matrix is expected to be real-valued and square.
    tol
        Absolute tolerance for determinant deviation from +1.
        This should be a small positive float, e.g., 1e-8.

    Returns
    -------
    - For a single matrix input: a Python `bool` indicating whether the matrix has unit determinant.
    - For a batch of matrices: a JAX boolean array of shape `(n_batches,)`
      where each element indicates whether the corresponding matrix has unit determinant.

    Raises
    ------
    arrayer.exception.InputError
        If the input array is not 2D/3D,
        or if the matrices are not square.
    """
    return _has_unit_determinant(matrix, tol)


@jax.jit
def is_rotation_single(matrix: jnp.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a matrix is a pure rotation matrix."""
    return is_orthogonal_single(matrix, tol=tol) & has_unit_determinant_single(matrix, tol=tol)


@jax.jit
def is_orthogonal_single(matrix: jnp.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a matrix is orthogonal."""
    identity = jnp.eye(matrix.shape[0], dtype=matrix.dtype)
    deviation = jnp.abs(matrix.T @ matrix - identity)
    return jnp.all(deviation <= tol)


@jax.jit
def has_unit_determinant_single(matrix: jnp.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a matrix has determinant approximately +1."""
    return jnp.abs(jnp.linalg.det(matrix) - 1.0) <= tol


def _vmap_if_batch(
    fn_single: Callable[..., Any],
    *,
    in_axes: int | tuple[int, ...] = (0, None),
    arg_name: str = "matrix",
) -> Callable[..., Any]:
    """Decorator to create a function that handles both single and batched matrix inputs.

    Parameters
    ----------
    fn_single
        A function that operates on a single matrix.
    in_axes
        The axes to map over for the batched version.
    arg_name
        Used for exception messages.

    Returns
    -------
    callable
        A function that dispatches to the jitted single or batched (vmapped + jitted) version.
    """
    # Create statically-compiled vmap+jit version at decoration time
    fn_batch = jax.jit(jax.vmap(fn_single, in_axes=in_axes))

    @wraps(fn_single)
    def wrapper(matrix: jnp.ndarray, *args, **kwargs) -> bool | jnp.ndarray:
        matrix = jnp.asarray(matrix)
        if matrix.shape[-1] != matrix.shape[-2]:
            raise exception.InputError(
                name=arg_name, value=matrix, problem=f"Matrices must be square, but have shape {matrix.shape[-2:]}."
            )
        if matrix.ndim == 2:
            return fn_single(matrix, *args, **kwargs).item()  # convert from DeviceArray(bool) to Python bool
        if matrix.ndim == 3:
            return fn_batch(matrix, *args, **kwargs)
        raise exception.InputError(
            name=arg_name,
            value=matrix,
            problem=f"Expected 2D or 3D input, but got shape {matrix.shape}"
        )
    return wrapper


_is_rotation = _vmap_if_batch(is_rotation_single)
_is_orthogonal = _vmap_if_batch(is_orthogonal_single)
_has_unit_determinant = _vmap_if_batch(has_unit_determinant_single)
