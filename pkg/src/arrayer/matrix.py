"""Matrix operations and properties."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp
import scipy as sp

from arrayer import exception

if TYPE_CHECKING:
    from typing import Literal


def matrix_is_rotational(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Determine if a matrix is a pure rotation matrix.

    A rotation matrix is a square matrix that represents a rotation
    in Euclidean space, preserving the length of vectors and angles between them
    (i.e., no scaling, shearing, or reflection).
    A matrix is a rotation matrix if it is orthogonal
    ($R^\\top R \\approx I$) and has determinant +1,
    meaning it preserves both length/angle and orientation.

    Parameters
    ----------
    matrix
        Square matrix of shape (n, n).
        Assumed to be real-valued.
        The matrix should represent a
        linear transformation in ℝⁿ.
    tol
        Absolute tolerance used for both orthogonality and determinant tests.

    Returns
    -------
    True if R is a pure rotation matrix; False otherwise.

    Raises
    ------
    scids.exception.InputError
        If R is not a 2D square matrix.

    References
    ----------
    - [Rotation matrix - Wikipedia](https://en.wikipedia.org/wiki/Rotation_matrix)
    """
    return matrix_is_orthogonal(matrix, tol=tol) and matrix_has_unit_determinant(matrix, tol=tol)


def matrix_is_orthogonal(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a matrix is orthogonal.

    Tests whether the transpose of the matrix multiplied by the matrix
    itself yields the identity matrix within a numerical tolerance.
    Matrix must be square and real-valued.

    Parameters
    ----------
    matrix
        Square matrix of shape (n, n).
        The matrix is expected to be real-valued (not complex),
        representing a linear transformation in ℝⁿ.
    tol
        Absolute tolerance for comparison against the identity matrix.
        Should be a small positive float, e.g., 1e-8.

    Returns
    -------
    True if $R^\\top R \\approx I$ within the given tolerance; False otherwise.

    Raises
    ------
    scids.exception.InputError
        If R is not a 2D square matrix.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise exception.InputError("matrix", f"Matrix must be 2D, but is {matrix.ndim}D: {matrix}.")
    if matrix.shape[0] != matrix.shape[1]:
        raise exception.InputError("matrix", f"Matrix must be square, but has shape {matrix.shape}.")
    identity = np.eye(matrix.shape[0], dtype=matrix.dtype)
    return np.allclose(matrix.T @ matrix, identity, atol=tol)


def matrix_has_unit_determinant(matrix: np.ndarray, tol: float = 1e-8) -> bool:
    """Check whether a matrix has determinant approximately +1.

    Used to test if a transformation matrix preserves orientation
    and volume, as required for a proper rotation.

    Parameters
    ----------
    matrix
        Square matrix of shape (n, n).
        Matrix is assumed to be real-valued.
    tol
        Absolute tolerance for deviation from +1.

    Returns
    -------
    True if $\\det(R) \\approx 1$ within the given tolerance; False otherwise.

    Raises
    ------
    scids.exception.InputError
        If R is not a 2D square matrix.
    """
    matrix = np.asarray(matrix)
    if matrix.ndim != 2:
        raise exception.InputError("matrix", f"Matrix must be 2D, but is {matrix.ndim}D: {matrix}.")
    if matrix.shape[0] != matrix.shape[1]:
        raise exception.InputError("matrix", f"Matrix must be square, but has shape {matrix.shape}.")
    return abs(np.linalg.det(matrix) - 1.0) <= tol
