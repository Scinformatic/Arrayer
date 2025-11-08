"""Utility functions for triangular meshes."""

import numpy as np


def vertex_normal(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Calculate per-vertex unit normals from area-weighted face normals."""
    v = vertices.astype(np.float64, copy=False)
    f = faces.astype(np.int64, copy=False)

    n = np.zeros_like(v, dtype=np.float64)
    p0, p1, p2 = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    fn = np.cross(p1 - p0, p2 - p0)  # area-weighted
    for i, tri in enumerate(f):
        n[tri[0]] += fn[i]
        n[tri[1]] += fn[i]
        n[tri[2]] += fn[i]
    lens = np.linalg.norm(n, axis=1)
    lens[lens == 0.0] = 1.0
    n /= lens[:, None]
    return n.astype(np.float32, copy=False)