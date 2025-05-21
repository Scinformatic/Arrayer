"""Principal Component Analysis ([PCA](https://en.wikipedia.org/wiki/Principal_component_analysis)).

This module provides functionalities for performing PCA
on real or complex-valued point clouds
using Singular Value Decomposition ([SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)).
Input data can be arrays of shape `(n_samples, n_features)`
representing `n_samples` points in `n_features` dimensions,
or batches of point clouds with shape `(*n_batches, n_samples, n_features)`,
where `*n_batches` can be any number of leading batch dimensions.

End users should call the `pca` function,
which automatically handles different input data shapes,
performs the necessary checks,
and returns a `PCAOutput` object containing the results.
The `pca_single` and `pca_batch` functions
are lower-level [JAX-jitted](https://docs.jax.dev/en/latest/jit-compilation.html) functions
that perform PCA on a single point cloud
or a batch of point clouds, respectively,
and can be used for integration into other JAX-based workflows.

See Also
--------
sklearn.decomposition.PCA : [Scikit-learn PCA implementation](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html)

Notes
-----
- In contrast to `sklearn.decomposition.PCA`,
  this implementation **supports complex-valued data**.
- In contrast to `sklearn.decomposition.PCA`,
  this implementation **enforces a pure rotation matrix** (i.e., no reflection)
  for real-valued data to ensure that the transformation can be safely applied
  to chiral point clouds (e.g., atomic coordinates in a molecule).
- Similar to `sklearn.decomposition.PCA`,
  this implementation **enforces a deterministic output**
  for real-valued data to ensure that the results are consistent across different runs.
  To do so, principal axes are first adjusted such that
  the loadings in the axes that are largest in absolute value are positive.
  Subsequently, if the determinant of the resulting principal component matrix
  is negative, the sign of the last principal axis is flipped.

References
----------
- [Greenacre, M., Groenen, P.J.F., Hastie, T. et al. Principal component analysis.
  *Nat Rev Methods Primers* **2**, 100 (2022)](https://doi.org/10.1038/s43586-022-00184-w)
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from arrayer import exception
from arrayer.typing import atypecheck, Array, JAXArray, Num


__all__ = [
    "pca",
    "pca_single",
    "pca_batch",
    "PCAOutput",
]

@atypecheck
@dataclass
class PCAOutput:
    """PCA results returned by the `pca` function.

    Attributes
    ----------
    points
        Point cloud projected onto the principal components.
        This has the same shape as the input data,
        but each point is centered and rotated
        to align with the principal axes.
    components
        Principal component matrix,
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
        Singular values from SVD corresponding to the principal components.
    translation
        Translation vector used to center the input data.

    Notes
    -----
    - With input data `input_points`, `self.points` is equal to
      `(input_points + self.translation) @ self.components.T`.
    - To reduce the dimensionality of the data from `n_features` to `k` (`k < n_features`),
      you can simply take the first `k` axes of each point in `self.points`, i.e.,
      `points_reduced = self.points[..., :k]`, which is equivalent to
      `(input_points + self.translation) @ self.components[:k].T`.
    """
    points: Num[JAXArray, "*n_batches n_samples n_features"]
    components: Num[JAXArray, "*n_batches n_features n_features"]
    singular_values: Num[JAXArray, "*n_batches n_features"]
    translation: Num[JAXArray, "*n_batches n_features"]

    def __post_init__(self):
        self._variance_magnitude = None
        self._variance_ratio = None
        self._variance_biased = None
        self._variance_unbiased = None
        return

    @property
    def variance_magnitude(self) -> Num[JAXArray, "*n_batches n_features"]:
        """Raw variance magnitudes (i.e., PCA energies) explained by the principal components.

        These are the squares of the singular values from SVD.
        """
        if self._variance_magnitude is None:
            self._variance_magnitude = self.singular_values ** 2
        return self._variance_magnitude

    @property
    def variance_ratio(self) -> Num[JAXArray, "*n_batches n_features"]:
        """Variance ratios explained by the principal components.

        These are the raw variance magnitudes `self.variance_magnitude`
        normalized to sum to 1 for each batch.
        """
        if self._variance_ratio is None:
            self._variance_ratio = self.variance_magnitude / self.variance_magnitude.sum(axis=-1, keepdims=True)
        return self._variance_ratio

    @property
    def variance_biased(self) -> Num[JAXArray, "*n_batches n_features"]:
        """Biased variances explained by the principal components.

        These are the raw variances `self.variance_magnitude`
        divided by the number of samples `n_samples`.
        """
        if self._variance_biased is None:
            num_samples = self.points.shape[-2]
            self._variance_biased = self.variance_magnitude / num_samples
        return self._variance_biased

    @property
    def variance_unbiased(self) -> Num[JAXArray, "*n_batches n_features"]:
        """Unbiased variances explained by the principal components.

        These are the raw variances `self.variance_magnitude`
        divided by `n_samples - 1`, i.e., applying Bessel's correction.
        These are the same as the eigenvalues of the covariance matrix.
        """
        if self._variance_unbiased is None:
            num_samples = self.points.shape[-2]
            self._variance_unbiased = self.variance_magnitude / (num_samples - 1)
        return self._variance_unbiased


@atypecheck
def pca(points: Num[Array, "*n_batches n_samples n_features"]) -> PCAOutput:
    """Perform PCA on one or several point clouds.

    Parameters
    ----------
    points
        Point cloud(s) as a real or complex-valued array of
        shape `(*n_batches, n_samples, n_features)`,
        where `*n_batches` is zero or more batch dimensions,
        holding point clouds with `n_samples` points in `n_features` dimensions.
        Note that both `n_samples` and `n_features` must be at least 2.

    References
    ----------
    - [Scikit-learn PCA implementation](https://github.com/scikit-learn/scikit-learn/blob/aa21650bcfbebeb4dd346307931dd1ed14a6f434/sklearn/decomposition/_pca.py#L113)
    """
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
        return PCAOutput(*pca_single(points))
    elif points.ndim == 3:
        return PCAOutput(*pca_batch(points))
    points_reshaped = points.reshape(-1, *points.shape[-2:])
    points_transformed, components, singular_values, translation_vector = pca_batch(points_reshaped)
    batch_shape = points.shape[:-2]
    return PCAOutput(
        points=points_transformed.reshape(points.shape),
        components=components.reshape(*batch_shape, components.shape[-2:]),
        singular_values=singular_values.reshape(*batch_shape, singular_values.shape[-1]),
        translation=translation_vector.reshape(*batch_shape, translation_vector.shape[-1]),
    )


@jax.jit
@atypecheck
def pca_single(
    points: Num[Array, "n_samples n_features"]
) -> tuple[
    Num[JAXArray, "n_samples n_features"],
    Num[JAXArray, "n_features n_features"],
    Num[JAXArray, "n_features"],
    Num[JAXArray, "n_features"],
]:
    """Perform PCA on a single point cloud.

    Parameters
    ----------
    points
        Point cloud as a real or complex-valued
        2D array of shape `(n_samples, n_features)`,
        representing `n_samples` points in `n_features` dimensions.
        Note that both `n_samples` and `n_features` must be at least 2.

    Returns
    -------
    A 4-tuple corresponding to the input arguments of `arrayer.pca.PCAOutput`.
    """
    points = jnp.asarray(points)

    # Center points
    center = jnp.mean(points, axis=0)
    translation_vector = -center
    points_centered = points + translation_vector

    # SVD decomposition
    # (u: left singular vectors, s: singular values, vh: conjugate-transposed right singular vectors)
    u, s, vh = jnp.linalg.svd(points_centered, full_matrices=False)

    if jnp.iscomplexobj(points):
        # For complex eigenvectors, the phase is arbitrary, and so
        # flipping signs and handedness are undefined and unnecessary.
        # We thus skip the sign flipping and handedness corrections.
        # On the other hand, Vh is now the conjugate transpose of V.
        points_transformed = points_centered @ vh.conj().T
    else:
        # Flip eigenvectors' signs to enforce deterministic output
        # Ref: https://github.com/scikit-learn/scikit-learn/blob/aa21650bcfbebeb4dd346307931dd1ed14a6f434/sklearn/utils/extmath.py#L895
        max_abs_v_rows = jnp.argmax(jnp.abs(vh), axis=1)
        shift = jnp.arange(vh.shape[0])
        signs = jnp.sign(vh[shift, max_abs_v_rows])
        u *= signs[None, :]
        vh *= signs[:, None]

        # Enforce right-handed coordinate system,
        # i.e., no reflections (determinant must be +1 and not -1)
        det_vt = jnp.linalg.det(vh)
        flip_factor = jnp.where(det_vt < 0, -1.0, 1.0)
        # Flip last row of vt and last column of u (if needed)
        vh = vh.at[-1].multiply(flip_factor)  # flip last principal component
        u = u.at[:, -1].multiply(flip_factor)  # adjust projected data to match

        # Transformed points (projection)
        # For real data, `u * s` is equal to `points_centered @ vh.conj().T`.
        points_transformed = u * s

    # Note that the same can be achieved by eigen decomposition of the covariance matrix:
    #     covariance_matrix = np.cov(points_centered, rowvar=False)
    #     eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    #     sorted_indices = np.argsort(eigenvalues)[::-1]
    #     variance = eigenvalues[sorted_indices]
    #     principal_components = eigenvectors[:, sorted_indices].T
    #     points_transformed = points @ principal_components

    return points_transformed, vh, s, translation_vector


@jax.jit
@atypecheck
def pca_batch(
    points: Num[Array, "n_batches n_samples n_features"]
) -> tuple[
    Num[JAXArray, "n_batches n_samples n_features"],
    Num[JAXArray, "n_batches n_features n_features"],
    Num[JAXArray, "n_batches n_features"],
    Num[JAXArray, "n_batches n_features"],
]:
    """Perform PCA on a batch of point clouds.

    Parameters
    ----------
    points
        Batch of point clouds as a real or complex-valued
        3D array of shape `(n_batches, n_samples, n_features)`,
        representing `n_batches` point clouds each with
        `n_samples` points in `n_features` dimensions.
        Note that both `n_samples` and `n_features` must be at least 2.

    Returns
    -------
    A 4-tuple corresponding to the input arguments of `arrayer.pca.PCAOutput`.
    """
    points = jnp.asarray(points)
    return _pca_batch(points)


_pca_batch = jax.vmap(pca_single)
