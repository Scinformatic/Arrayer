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


def make_spherical_surface(
    center: np.ndarray,
    radius: float,
    distance: float,
) -> np.ndarray:
    """Generate quasi-uniform points on the surface of a 3D sphere.

    This function uses the Fibonacci lattice method
    to generate coordinates for nearly uniform points on the
    surface of a sphere with given `center` and `radius`.
    The number of points is chosen based on the surface area
    of the sphere and the requested point spacing `distance`.

    Parameters
    ----------
    center
        An array of shape (..., 3) giving the coordinates of the center point(s).
        Broadcasting is supported: multiple centers will produce one set of points per center.
    radius
        The radius of the sphere(s).
    distance
        Approximate geodesic spacing between sampled points on
        the sphere surface.

    Returns
    -------
    An array of shape (..., M, 3) giving the coordinates of the sampled points,
    where M is the number of points sampled on the sphere's surface.

    Notes
    -----
    - This uses a quasi-uniform Fibonacci sphere distribution,
      which avoids clustering at the poles compared to spherical
      grids.
    - The actual spacing may not be exactly `distance`, but is
      close on average across the sphere.
    - Example visualization of point distributions:
    ```python

    import numpy as np
    from matplotlib import pyplot as plt


    points = sample_spherical_surface(center=[0, 0, 0], radius=3.0, distance=2.0)

    # Extract coordinates directly
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Plot
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, s=30, color="royalblue")

    # Sphere wireframe for reference
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 15)
    xs = 3 * np.outer(np.cos(u), np.sin(v))
    ys = 3 * np.outer(np.sin(u), np.sin(v))
    zs = 3 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", alpha=0.3)

    ax.set_box_aspect([1, 1, 1])
    ax.set_title("Sampled points on sphere surface (r=3, d=2)")
    plt.show()
    """
    center = np.asarray(center, dtype=float)

    if center.shape[-1] != 3:
        raise ValueError("`center` must have shape (..., 3)")

    # --- Estimate number of points ---
    # Surface area of sphere: 4πr²
    surface_area = 4.0 * np.pi * radius**2
    # Area per point ~ distance², so N ~ surface_area / distance²
    n_points = max(1, int(np.round(surface_area / (distance**2))))

    # --- Fibonacci sphere algorithm ---
    # Golden angle increment
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))  # ~2.4

    # Indices
    idx = np.arange(n_points)

    # y-coordinates uniformly spaced in [-1, 1]
    y = 1.0 - 2.0 * (idx + 0.5) / n_points
    r_xy = np.sqrt(1.0 - y**2)

    theta = golden_angle * idx
    x = r_xy * np.cos(theta)
    z = r_xy * np.sin(theta)

    # Unit vectors on sphere surface
    unit_points = np.stack((x, y, z), axis=-1)

    # Scale to radius
    sphere_points = radius * unit_points  # (M, 3)

    # Broadcast to each center: (..., M, 3)
    # Add newaxis at -2 to match sphere_points
    result = center[..., None, :] + sphere_points
    return result
