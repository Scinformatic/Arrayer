"""Matrix operations and properties.

This module provides functionalities for performing various checks and operations
on matrices of shape `(n_rows, n_columns)`,
or batches thereof with shape `(*n_batches, n_samples, n_features)`,
where `*n_batches` can be any number of leading batch dimensions.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from arrayer.typing import atypecheck, Array, JAXArray, Num, Bool, Float


__all__ = [
    "is_rotation",
    "is_orthogonal",
    "has_unit_determinant",
]


@jax.jit
@atypecheck
def is_rotation(
    matrix: Num[Array, "*n_batches n_rows n_rows"],
    tol: float | Float[Array, ""] = 1e-6,
) -> Bool[JAXArray, "*n_batches"]:
    """Check whether the input represents pure [rotation matrices](https://en.wikipedia.org/wiki/Rotation_matrix).

    This is done by verifying that each matrix is
    orthogonal and has a determinant of +1
    within a numerical tolerance.

    Parameters
    ----------
    matrix
        Square matrices (`n_rows` = `n_columns`)
        as an array of shape `(*n_batches, n_rows, n_columns)`,
        where `*n_batches` is zero or more batch dimensions.
    tol
        Absolute tolerance used for both orthogonality and determinant tests.
        This threshold defines the allowed numerical deviation
        from perfect rotation properties.
        It should be a small positive number.

    Returns
    -------
    A JAX boolean array of shape `(*n_batches,)`,
    where each element indicates whether the corresponding matrix is a rotation matrix.

    Notes
    -----
    A rotation matrix is a square matrix that represents a rigid-body rotation
    in Euclidean space, preserving the length of vectors and angles between them
    (i.e., no scaling, shearing, or reflection).
    A matrix is a rotation matrix if it is orthogonal
    ($R^\\top R \\approx I$) and has determinant +1,
    meaning it preserves both length/angle and orientation.
    """
    return is_orthogonal(matrix, tol=tol) & has_unit_determinant(matrix, tol=tol)


@jax.jit
@atypecheck
def is_orthogonal(
    matrix: Num[Array, "*n_batches n_rows n_rows"],
    tol: float | Float[Array, ""] = 1e-6,
) -> Bool[JAXArray, "*n_batches"]:
    """Check whether the input represents orthogonal matrices.

    This is done by checking whether the transpose of the matrix
    multiplied by the matrix itself yields the identity matrix
    within a numerical tolerance, i.e., $R^\\top R \\approx I$.

    Parameters
    ----------
    matrix
        Square matrices (`n_rows` = `n_columns`)
        as an array of shape `(*n_batches, n_rows, n_columns)`,
        where `*n_batches` is zero or more batch dimensions.
    tol
        Absolute tolerance for comparison against the identity matrix.
        This should be a small positive float, e.g., 1e-8.

    Returns
    -------
    A JAX boolean array of shape `(*n_batches,)`,
    where each element indicates whether the corresponding matrix is orthogonal.
    """
    transposed_matrix = jnp.swapaxes(matrix, -1, -2)
    gram_matrix = transposed_matrix @ matrix
    identity_matrix = jnp.eye(matrix.shape[-1], dtype=matrix.dtype)
    deviation = jnp.abs(gram_matrix - identity_matrix)
    return jnp.all(deviation <= tol, axis=(-2, -1))


@jax.jit
@atypecheck
def has_unit_determinant(
    matrix: Num[Array, "*n_batches n_rows n_rows"],
    tol: float | Float[Array, ""] = 1e-6,
) -> Bool[JAXArray, "*n_batches"]:
    """Check whether the input represents matrices with determinant approximately +1.

    This can be used, e.g., to test if a transformation matrix
    preserves orientation and volume, as required for a proper rotation.

    Parameters
    ----------
    matrix
        Square matrices (`n_rows` = `n_columns`)
        as an array of shape `(*n_batches, n_rows, n_columns)`,
        where `*n_batches` is zero or more batch dimensions.
    tol
        Absolute tolerance for determinant deviation from +1.
        This should be a small positive float, e.g., 1e-8.

    Returns
    -------
    A JAX boolean array of shape `(*n_batches,)`,
    where each element indicates whether the corresponding matrix has unit determinant.

    Notes
    -----
    - Determinants are computed using
      [`jax.scipy.linalg.det`](https://docs.jax.dev/en/latest/_autosummary/jax.scipy.linalg.det.html).
    """
    return jnp.abs(jsp.linalg.det(matrix) - 1.0) <= tol
