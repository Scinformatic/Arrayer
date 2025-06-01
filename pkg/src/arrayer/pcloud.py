"""Generate point clouds in different shapes and orientations."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from arrayer import exception
from arrayer.typing import atypecheck, Array, JAXArray, Num


@partial(jax.jit, static_argnames=('per_instance',))
@atypecheck
def bounds(
    points: Num[Array, "*n_batches n_samples n_features"],
    per_instance: bool = True,
) -> tuple[
    Num[JAXArray, "*n_batches n_features"] | Num[JAXArray, "n_features"],
    Num[JAXArray, "*n_batches n_features"] | Num[JAXArray, "n_features"],
]:
    """Calculate lower and upper bounds of point cloud coordinates.

    Compute the minimum and maximum coordinates along the point dimension for
    one or several point clouds. Supports per-instance or global computation.

    Parameters
    ----------
    points
        Point cloud(s) as an array of shape `(*n_batches, n_samples, n_features)`,
        where `*n_batches` is zero or more batch dimensions,
        holding point clouds with `n_samples` points in `n_features` dimensions.
    per_instance
        If True, compute bounds separately for each instance,
        yielding arrays of shape `(*n_batches, n_features)`.
        If False, compute bounds over all instances superposed,
        yielding arrays of shape `(n_features,)`.

    Returns
    -------
    lower_bounds
        Minimum values per dimension.
    upper_bounds
        Maximum values per dimension.
    """
    if points.ndim < 2:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"At least a 2D array is required, but got {points.ndim}D array with shape {points.shape}."
        )
    # If per_instance, reduce over the points axis (-2) only, preserving batch dimensions,
    # otherwise reduce over all axes except the last (dimension axis).
    axis = -2 if per_instance else tuple(range(points.ndim - 1))
    lower_bounds = jnp.min(points, axis=axis)
    upper_bounds = jnp.max(points, axis=axis)
    return lower_bounds, upper_bounds


@partial(jax.jit, static_argnames=('per_instance',))
@atypecheck
def aabb(
    points: Num[Array, "*n_batches n_samples n_features"],
    per_instance: bool = True,
) -> tuple[
    Num[JAXArray, "*n_batches n_features"] | Num[JAXArray, "n_features"],
    Num[JAXArray, "*n_batches n_features"] | Num[JAXArray, "n_features"],
    Num[JAXArray, "*n_batches"] | Num[JAXArray, ""],
    Num[JAXArray, "*n_batches"] | Num[JAXArray, ""],
]:
    """Calculate lower and upper bounds of point cloud coordinates.

    Compute the minimum and maximum coordinates along the point dimension for
    one or several point clouds. Supports per-instance or global computation.

    Parameters
    ----------
    points
        Point cloud(s) as an array of shape `(*n_batches, n_samples, n_features)`,
        where `*n_batches` is zero or more batch dimensions,
        holding point clouds with `n_samples` points in `n_features` dimensions.
    per_instance
        If True, compute bounds separately for each instance,
        yielding arrays of shape `(*n_batches, n_features)`.
        If False, compute bounds over all instances superposed,
        yielding arrays of shape `(n_features,)`.

    Returns
    -------
    lower_bounds
        Minimum values per dimension.
    upper_bounds
        Maximum values per dimension.
    """
    if points.ndim < 2:
        raise exception.InputError(
            name="points",
            value=points,
            problem=f"At least a 2D array is required, but got {points.ndim}D array with shape {points.shape}."
        )
    # If per_instance, reduce over the points axis (-2) only, preserving batch dimensions,
    # otherwise reduce over all axes except the last (dimension axis).
    axis = -2 if per_instance else tuple(range(points.ndim - 1))
    lower_bounds = jnp.min(points, axis=axis)
    upper_bounds = jnp.max(points, axis=axis)
    return lower_bounds, upper_bounds


def make_cylinder(
    radius: float = 0.05,
    n_points: int = 1000,
    start: tuple[float, float, float] = (0, 0, -1),
    end: tuple[float, float, float] = (0, 0, 1)
) -> np.ndarray:
    """Generate 3D points forming a solid cylinder between two arbitrary points in space.

    Parameters
    ----------
    radius
        Radius of the cylinder.
    n_points
        Number of points to generate.
    start
        Starting point (x, y, z) of the cylinder axis.
    end
        Ending point (x, y, z) of the cylinder axis.

    Returns
    -------
    Array of shape `(n_points, 3)` with the generated points.
    """
    # Generate cylinder points aligned with Z-axis (unit cylinder from 0 to 1 in Z)
    theta = np.random.uniform(0, 2 * np.pi, n_points)
    r = np.sqrt(np.random.uniform(0, 1, n_points)) * radius
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.random.uniform(0, 1, n_points)

    points = np.column_stack((x, y, z))

    # Compute transformation from (0,0,0)-(0,0,1) to (start)-(end)
    axis_vector = np.array(end) - np.array(start)
    cyl_length = np.linalg.norm(axis_vector)
    if cyl_length == 0:
        raise ValueError("Start and end points must be different to define a cylinder axis.")

    # Normalize direction
    direction = axis_vector / cyl_length

    # Build rotation matrix: align (0,0,1) with desired direction using Rodrigues' rotation formula
    z_axis = np.array([0, 0, 1])
    v = np.cross(z_axis, direction)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, direction)

    if s < 1e-8:
        # No rotation needed (aligned already)
        R = np.eye(3) if c > 0 else -np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (s ** 2))

    # Scale cylinder along its axis
    points[:, 2] *= cyl_length

    # Rotate and translate points
    rotated_points = points @ R.T + np.array(start)

    return rotated_points
