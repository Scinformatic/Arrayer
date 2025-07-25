"""Kabsch Alignment ([Kabsch-Umeyama algorithm](https://en.wikipedia.org/wiki/Kabsch_algorithm)).

This module provides functionalities for performing
the Kabsch alignment algorithm on point clouds
using Singular Value Decomposition ([SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition)).
The Kabsch algorithm is used to find the optimal rigid transformations (i.e. rotation and translation)
to minimize the root mean square deviation (RMSD) between two sets of points.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from arrayer import exception
from arrayer.typing import atypecheck, Array, JAXArray, Num


__all__ = [
    "KabschOutput",
    "kabsch",
]


@atypecheck
@dataclass
class KabschOutput:
    """Kabsch alignment results returned by `kabsch` function.

    Attributes
    ----------
    """
    pass




@jax.jit
def kabsch_unweighted(p0: jax.Array, p1: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """
    Align two point clouds by a rigid transformation (i.e. only rotation and translation),
    using singular value decomposition (SVD) according to the Kabsch algorithm.

    Parameters
    ----------
    p0 : jax.Array, shape: (n_samples, n_features)
        The reference point cloud.
    p1 : jax.Array, shape: (n_samples, n_features)
        The point cloud to align.

    Returns
    -------
    r, t, rmsd : 3-tuple of jax.Array
        r : jax.Array, shape: (n_features, n_features)
            Rotation matrix to align p1 onto p0.
        t : jax.Array, shape: (n_features, )
            Translation vector to align p1 onto p0.
        rmsd : jax.Array, shape: ()
            RMSD value between p0 and p1 after alignment.

        The coordinates of the aligned point cloud can be obtained as:
        `p1_aligned = p1 @ r + t`
    """
    c_p0 = jnp.mean(p0, axis=0)  # geometric center of p0
    c_p1 = jnp.mean(p1, axis=0)  # geometric center of p1
    p0c = p0 - c_p0  # p0 centered
    p1c = p1 - c_p1  # p1 centered
    # Calculate SVD of the cross-covariance matrix
    h = jnp.einsum("ji,jk->ik", p0c, p1c)  # equivalent to `p0c.T @ p1c`
    u, s, vt = jnp.linalg.svd(h)
    # Calculate rotation matrix
    rot = jnp.einsum("ij,jk->ki", u, vt)  # equivalent to `(u @ vt).T`
    det_sign = jnp.sign(jnp.linalg.det(rot))  # Sign of the determinant of the rotation matrix
    # Note: since u @ vt is a rotation matrix, it's determinant is either 1, or -1 in case of
    #  an improper rotation. In the later case, the improper rotation is corrected to a proper rotation
    #  by manipulating the s and u values:
    s = s.at[-1].multiply(det_sign)
    u = u.at[:, -1].multiply(det_sign)
    rot = jnp.einsum("ij,jk->ki", u, vt)  # final rotation matrix
    # Notice that the above code could have be written as a conditional as well:
    #     rot = jnp.einsum('ij,jk->ki', u, vt)
    #     if jnp.linalg.det(rot) < 0:
    #         s[-1] *= -1
    #         u[:, -1] *= -1
    #         rot = np.einsum('ij,jk->ki', u, vt)
    #  However, then the function could not have been jitted.
    trans = c_p0 - jnp.dot(c_p1, rot)  # translation vector
    e0 = jnp.sum(p0c**2 + p1c**2)  # initial residual
    rmsd = jnp.sqrt(jnp.maximum((e0 - (2 * jnp.sum(s))) / p0.shape[0], 0))  # rmsd after alignment
    return rot, trans, rmsd


@jax.jit
def kabsch_unweighted_transform(
    p0: jax.Array, p1: jax.Array
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    rot, trans, rmsd = kabsch_unweighted(p0, p1)
    return rot, trans, rmsd, p1 @ rot + trans


@jax.jit
def kabsch_weighted(p0, p1, w):
    denom = jnp.sum(w)
    if w.shape == p0.shape:
        # Full weights were provided, thus sum(w) is the full sum;
        #  it only needs to be divided by n_features.
        denom /= p0.shape[1]
        denom2 = jnp.sum(w, axis=0)
    elif w.shape[0] != p0.shape[0]:
        # w has shape (1, n_feature), thus sum(w) was only for one sample;
        #  it must be multiplied by the n_samples, and then divided by n_features.
        denom *= p0.shape[0] / p0.shape[1]
        denom2 = w[0] * p0.shape[0]
    else:
        denom2 = denom

    c_p0 = jnp.sum(p0 * w, axis=0) / denom2
    c_p1 = jnp.sum(p1 * w, axis=0) / denom2
    p0c = p0 - c_p0
    p1c = p1 - c_p1
    h = jnp.einsum("ji,jk->ik", p0c * w, p1c)
    u, s, vt = jnp.linalg.svd(h)
    det_sign = jnp.sign(jnp.linalg.det(jnp.einsum("ij,jk->ki", u, vt)))
    s = s.at[-1].multiply(det_sign)
    u = u.at[:, -1].multiply(det_sign)
    rot = jnp.einsum("ij,jk->ki", u, vt)
    trans = c_p0 - jnp.dot(c_p1, rot)
    e0 = jnp.sum(w * (p0c**2 + p1c**2))
    rmsd = jnp.sqrt(jnp.maximum((e0 - (2 * jnp.sum(s))) / denom, 0))
    return rot, trans, rmsd


@jax.jit
def affine_map(p: jax.Array, linear_map: jax.Array, translation: jax.Array) -> jax.Array:
    """
    Apply an affine map (linear map and translation) to an n-dimensional array.

    Parameters
    ----------
    p : jax.Array
        Array to be mapped.
    linear_map : jax.Array
        Array representation of the linear map.
    trans : jax.Array
        Translation vector.

    Returns
    -------
    p_transformed : jax.Array
    """
    return p @ linear_map + translation


# Other Kabsh implementations

import numpy as np


def _scipy(p0, p1, weights=None):
    if weights is None:
        weights = np.ones(len(p1))
    else:
        weights = np.asarray(weights)

    h = np.einsum("ji,jk->ik", weights[:, None] * p0, p1)

    u, s, vt = np.linalg.svd(h)

    if np.linalg.det(u @ vt) < 0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    rot = np.dot(u, vt)

    rssd = np.sqrt(max(np.sum(weights * np.sum(p1**2 + p0**2, axis=1)) - 2 * np.sum(s), 0))

    return rot, rssd


def _optAlign(p0, p1):
    """
    OptAlign performs the Kabsch alignment algorithm upon the alpha-carbons of two selections.
    Example:   optAlign MOL1 and i. 20-40, MOL2 and i. 102-122
    Example 2: optAlign 1GGZ and i. 4-146 and n. CA, 1CLL and i. 4-146 and n. CA

    Two RMSDs are returned.  One comes from the Kabsch algorithm and the other from
    PyMol based upon your selections.

    By default, this program will optimally align the ALPHA CARBONS of the selections provided.
    To turn off this feature remove the lines between the commented "REMOVE ALPHA CARBONS" below.

    @param p0: First PyMol selection with N-atoms
    @param p1: Second PyMol selection with N-atoms
    """
    # must alway center the two proteins to avoid
    # affine transformations.  Center the two proteins to their selections.

    # Initial residual, see Kabsch.
    E0 = np.sum(np.sum(p0_centered**2, axis=0), axis=0) + np.sum(
        np.sum(p1_centered**2, axis=0), axis=0
    )

    u, s, vt = np.linalg.svd(np.dot(p1_centered.T, p0_centered))

    reflect = float(str(float(np.linalg.det(u) * np.linalg.det(vt))))

    if reflect == -1.0:
        s[-1] = -s[-1]
        u[:, -1] = -u[:, -1]

    RMSD = E0 - (2.0 * sum(s))
    RMSD = np.sqrt(abs(RMSD / L))

    # U is simply V*Wt
    rot = np.dot(u, vt)

    # rotate and translate the molecule
    p1 = np.dot((mol2 - center_p1), rot)
    # center the molecule
    p0 = mol1 - center_p0
