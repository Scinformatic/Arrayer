"""Generate point clouds in different shapes and orientations."""

import numpy as np


def cylinder(
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
