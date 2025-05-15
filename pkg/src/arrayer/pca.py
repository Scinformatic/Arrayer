"""Principal Component Analysis (PCA)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from functools import partial

import jax
import numpy as np
import jax.numpy as jnp

from arrayer import exception

if TYPE_CHECKING:
    from numpy.typing import ArrayLike


__all__ = [
    "pca",
    "pca_single",
    "pca_batch",
    "PCAOutput",
]


@dataclass
class PCAOutput:
    """Principal Component Analysis (PCA) output.

    Note that all given shapes here
    are for the case of 2D input data (i.e., shape `(n_samples, n_features)`).
    For 3D input data (i.e., shape `(n_batches, n_samples, n_features)`),
    the batch dimension is added as the first axis.

    Attributes
    ----------
    points
        Transformed points in PCA space.
        This has the same shape as the input data,
        but each point is centered and rotated
        to align with the principal axes.
    components
        Principal component matrix.
        This is a matrix of shape `(n_features, n_features)`,
        where each row is a principal component,
        i.e., an eigenvector of the covariance matrix
        (sorted from largest to smallest variance).
        This matrix can act as a rotation matrix
        to align points with the principal axes.
        For example, to rotate points in an array `a`
        where the last axis is the feature axis,
        i.e., any array of shape `(..., n_features)`,
        use `a @ self.components.T`, or the equivalent `np.matmul(a, self.components.T)`.
    singular_values
        Singular values from SVD.
    translation
        Translation vector used to center the input data,
        as a 1D array of shape `(n_features,)`.

    Notes
    -----
    - With input data `input_points`, `self.points` is equal to
      `(input_points + self.translation) @ self.components.T`.
    - To reduce the dimensionality of the data from `n_features` to `k` (`k < n_features`),
      you can simply take the first `k` axes of each point in `self.points`, i.e.,
      `points_reduced = self.points[..., :k]`, which is equivalent to
      `(input_points + self.translation) @ self.components[:k].T`.
    """
    points: jnp.ndarray
    components: jnp.ndarray
    singular_values: jnp.ndarray
    translation: jnp.ndarray

    def __post_init__(self):
        self._variance_magnitude = None
        self._variance_ratio = None
        self._variance_biased = None
        self._variance_unbiased = None
        return

    @property
    def variance_magnitude(self) -> jnp.ndarray:
        """Raw variance magnitudes (i.e., PCA energies) explained by the principal components.

        These are the squares of the singular values from SVD.
        This is a 1D array of shape `(n_features,)`.
        """
        if self._variance_magnitude is None:
            self._variance_magnitude = self.singular_values ** 2
        return self._variance_magnitude

    @property
    def variance_ratio(self) -> jnp.ndarray:
        """Variance ratios explained by the principal components.

        These are the raw variance magnitudes `self.variance_magnitude`
        normalized to sum to 1 for each batch.
        """
        if self._variance_ratio is None:
            self._variance_ratio = self.variance_magnitude / self.variance_magnitude.sum(axis=-1, keepdims=True)
        return self._variance_ratio

    @property
    def variance_biased(self) -> jnp.ndarray:
        """Biased variances explained by the principal components.

        These are the raw variances `self.variance_magnitude`
        divided by the number of samples `n_samples`.
        """
        if self._variance_biased is None:
            num_samples = self.points.shape[-2]
            self._variance_biased = self.variance_magnitude / num_samples
        return self._variance_biased

    @property
    def variance_unbiased(self) -> jnp.ndarray:
        """Unbiased variances explained by the principal components.

        These are the raw variances `self.variance_magnitude`
        divided by `n_samples - 1`, i.e., applying Bessel's correction.
        These are the same as the eigenvalues of the covariance matrix.
        """
        if self._variance_unbiased is None:
            num_samples = self.points.shape[-2]
            self._variance_unbiased = self.variance_magnitude / (num_samples - 1)
        return self._variance_unbiased


def pca(points: ArrayLike) -> PCAOutput:
    """Perform Principal Component Analysis (PCA) on a set of points.

    Parameters
    ----------
    points
        Input data of shape `(n_samples, n_features)`
        or `(n_batches, n_samples, n_features)`.

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
        func = pca_single
    elif points.ndim == 3:
        func = pca_batch
    else:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"Points must be a 2D or 3D array, but is {points.ndim}D."
        )
    return PCAOutput(*func(points))


@jax.jit
def pca_single(
    points: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Perform PCA on a single set of points with shape `(n_samples, n_features)`."""
    # Center points
    center = jnp.mean(points, axis=0)
    translation_vector = -center
    points_centered = points + translation_vector

    # SVD decomposition
    u, singular_values, vt = jnp.linalg.svd(points_centered, full_matrices=False)

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
    components = vt.at[-1].multiply(flip_factor)  # flip last principal component
    u = u.at[:, -1].multiply(flip_factor)  # adjust projected data to match

    # Transformed points (projection)
    points_transformed = u * singular_values  # equal to `points_centered @ vt.T`

    # Note that the same can be achieved by eigen decomposition of the covariance matrix:
    #     covariance_matrix = np.cov(points_centered, rowvar=False)
    #     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    #     sorted_indices = np.argsort(eigenvalues)[::-1]
    #     variance = eigenvalues[sorted_indices]
    #     principal_components = eigenvectors[:, sorted_indices].T
    #     points_transformed = points @ principal_components

    return points_transformed, components, singular_values, translation_vector


pca_batch = jax.vmap(pca_single)
"""Perform PCA on a batch of points with shape `(n_batches, n_samples, n_features)`."""
