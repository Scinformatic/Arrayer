from __future__ import annotations

from typing import TYPE_CHECKING
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp

from arrayer import exception

if TYPE_CHECKING:
    from typing import Literal
    from numpy.typing import ArrayLike


__all__ = ["pca", "pca_single", "pca_batch"]


def pca(
    points: ArrayLike,
    variance_type: Literal["raw", "ratio", "biased", "unbiased"] = "unbiased"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform Principal Component Analysis (PCA) on a set of points.

    Parameters
    ----------
    points
        Input data of shape `(n_samples, n_features)`
        or `(n_batches, n_samples, n_features)`.
    variance_type
        Which explained variance values to return:
        - "raw": Raw variance magnitudes (i.e., PCA energies).
          These are the squares of the singular values.
        - "ratio": Variance ratios.
          These are the raw variances normalized to sum to 1.
        - "biased": Biased variance.
          This is the raw variances divided by `n_samples`.
        - "unbiased": Unbiased variance.
          This is the raw variances divided by `n_samples - 1`,
          i.e., applying Bessel's correction.
          This is the same as the eigenvalues of the covariance matrix.

    Returns
    -------
    A 4-tuple containing
    (note that the given shapes here are for the case of 2D input data;
    for 3D input data, the batch dimension is added as the first axis):

    1. Principal component matrix `P`.
       This is a matrix of shape `(n_features, n_features)`,
       where each row is a principal component,
       i.e., an eigenvector of the covariance matrix
       (sorted from largest to smallest variance).
       This matrix can act as a rotation matrix
       to align points with the principal axes.
       For example, to rotate points in an array `a`
       where the last axis is the feature axis,
       i.e., any array of shape `(..., n_features)`,
       use `a @ P.T`, or the equivalent `np.matmul(a, P.T)`.
    2. Variance explained by each principal component.
       This is a 1D array of shape `(n_features,)`
       containing the variance explained by each principal component.
       The type of variance is determined by the `variance_type` parameter.
    3. Translation vector `t`.
       This is a 1D array of shape `(n_features,)`
       representing the translation vector used to center the points,
       i.e., `points_centered = points + t`.
    4. Transformed points `points_transformed`.
       This is a 2D array of shape `(n_samples, n_features)`,
       where each row is a corresponding point from `points` in the PCA space.
       The points are centered and rotated to align with the principal axes,
       i.e., `points_transformed = points_centered @ P.T`.

    To reduce the dimensionality of the data,
    multiply the points with the first `k` principal components.
    For example, to eliminate the last dimension,
    you can use `points_reduced = points @ P[:-1].T`,
    or `points_centered_and_reduced = (points + t) @ P[:-1].T`.

    Notes
    -----
    PCA is performed using Singular Value Decomposition (SVD).
    This function enforces a pure rotation matrix (i.e., no reflection)
    and a deterministic output.
    This is done to ensure that the transformation
    can be applied to chiral data (e.g., atomic coordinates in a molecule),
    and that the principal components are consistent across different runs.
    To do so, principal axes are first adjusted such that
    the loadings in the axes that are largest
    in absolute value are positive.
    Subsequently, if the determinant of the resulting principal component matrix
    is negative, the sign of the last principal axis is flipped.

    References
    ----------
    - [Scikit-learn PCA implementation](https://github.com/scikit-learn/scikit-learn/blob/aa21650bcfbebeb4dd346307931dd1ed14a6f434/sklearn/decomposition/_pca.py#L113)
    - [Scikit-learn PCA documentation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)
    """
    points = jnp.asarray(points)
    if points.shape[-2] < 2:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"At least 2 points are required, but got {points.shape[0]}."
        )
    if points.shape[-1] < 2:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"At least 2 features are required, but got {points.shape[1]}."
        )
    if points.ndim == 2:
        return pca_single(points, variance_type)
    if points.ndim == 3:
        return pca_batch(points, variance_type)
    raise exception.InputError(
        name="points",
        value=points,
        problem=f"Points must be a 2D or 3D array, but is {points.ndim}D."
    )


@partial(jax.jit, static_argnames=("variance_type",))
def pca_single(
    points: jnp.ndarray,
    variance_type: Literal["raw", "ratio", "biased", "unbiased"] = "unbiased"
):
    """Perform PCA on a single set of points with shape `(n_samples, n_features)`."""
    # Center points
    center = jnp.mean(points, axis=0)
    translation_vector = -center
    points_centered = points + translation_vector

    # SVD decomposition
    u, s, vt = jnp.linalg.svd(points_centered, full_matrices=False)

    # Flip eigenvectors' signs to enforce deterministic output
    # Ref: https://github.com/scikit-learn/scikit-learn/blob/aa21650bcfbebeb4dd346307931dd1ed14a6f434/sklearn/utils/extmath.py#L895
    max_abs_v_rows = jnp.argmax(jnp.abs(vt), axis=1)
    shift = jnp.arange(vt.shape[0])
    signs = jnp.sign(vt[shift, max_abs_v_rows])
    u = u * signs[None, :]
    vt = vt * signs[:, None]

    # Enforce right-handed coordinate system,
    # i.e., no reflections (determinant must be +1 and not -1)
    det_vt = jnp.linalg.det(vt)
    flip_factor = jnp.where(det_vt < 0, -1.0, 1.0)
    # Flip last row of vt and last column of u (if needed)
    vt = vt.at[-1].multiply(flip_factor)  # flip last principal component
    u = u.at[:, -1].multiply(flip_factor)  # adjust projected data to match

    # Transformed points (projection)
    points_transformed = u * s  # equal to `points_centered @ vt.T`

    # Note that the same can be achieved by eigen decomposition of the covariance matrix:
    #     covariance_matrix = np.cov(points_centered, rowvar=False)
    #     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    #     sorted_indices = np.argsort(eigenvalues)[::-1]
    #     variance = eigenvalues[sorted_indices]
    #     principal_components = eigenvectors[:, sorted_indices].T
    #     points_transformed = points @ principal_components

    # Calculate explained variance
    variance = s ** 2
    variance = jax.lax.switch(
        {"raw": 0, "ratio": 1, "biased": 2, "unbiased": 3}[variance_type],
        [
            lambda v: v,
            lambda v: v / v.sum(),
            lambda v: v / points.shape[0],
            lambda v: v / (points.shape[0] - 1)
        ],
        variance
    )
    return vt, variance, translation_vector, points_transformed


pca_batch = jax.vmap(pca_single, in_axes=(0, None))
"""Perform PCA on a batch of points with shape `(n_batches, n_samples, n_features)`."""
