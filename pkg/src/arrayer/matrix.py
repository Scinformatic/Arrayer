"""Matrix operations and properties.

This module provides functionalities for performing various checks and operations
on matrices of shape `(n_rows, n_columns)`,
or batches thereof with shape `(*n_batches, n_samples, n_features)`,
where `*n_batches` can be any number of leading batch dimensions.
"""

import jax
import jax.numpy as jnp
import jax.scipy as jsp

from arrayer.typing import atypecheck, Array, JAXArray, Num, Bool, Float, Int


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
    transposed_matrix = jnp.swapaxes(matrix, -2, -1)
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


@atypecheck
def linearly_dependent_pairs(
    matrix: Num[Array, "n_rows n_columns"],
    rtol: float = 1e-6,
    atol: float = 1e-8,
    equal_nan: bool = False,
) -> Int[JAXArray, "n_pairs 2"]:
    """Find index pairs of row vectors that are linearly dependent.

    Two vectors are considered linearly dependent if one is a scalar multiple of the other.
    This includes both parallel and anti-parallel vectors.

    Parameters
    ----------
    matrix
        2D matrix of shape `(n_rows, n_columns)`,
        where each row is a `n_columns`-dimensional vector.
    rtol
        Relative tolerance for floating-point comparison.
    atol
        Absolute tolerance for floating-point comparison.
    equal_nan
        Treat NaNs as equal.

    Returns
    -------
    Index pairs (i, j) with i < j such that `matrix[i]` and `matrix[j]` are linearly dependent.
    """
    # Normalize vectors
    norms = jnp.linalg.norm(matrix, axis=1)
    unit = matrix / norms[:, None]
    # Compute cosine similarity matrix (absolute value to catch both parallel and anti-parallel)
    cos_sim = unit @ unit.T
    abs_cos_sim = jnp.abs(cos_sim)
    # Ignore diagonal
    abs_cos_sim = abs_cos_sim - jnp.eye(abs_cos_sim.shape[0])
    # Get upper triangle mask (i < j)
    mask = jnp.triu(jnp.ones_like(abs_cos_sim, dtype=bool), k=1)
    # Condition for linear dependence: |cos_sim| ≈ 1.0
    linear_dep = jnp.isclose(abs_cos_sim, 1.0, rtol=rtol, atol=atol, equal_nan=equal_nan)
    # Apply mask to get (i, j) indices
    valid = jnp.logical_and(mask, linear_dep)
    i, j = jnp.where(valid)
    return jnp.stack((i, j), axis=1)
