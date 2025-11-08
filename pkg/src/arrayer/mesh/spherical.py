"""Generate triangular surface meshes from inputs in spherical coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import arrayer.mesh.util as mesh_util


# Types
# ============================================================================

Vec3 = tuple[float, float, float]
NDArrayF = np.ndarray
NDArrayI = np.ndarray


def box(
    r_min: float = 0.0,
    r_max: float = 1.0,
    theta_min: float = 0.0,
    theta_max: float = np.pi,
    phi_min: float = 0.0,
    phi_max: float = 2 * np.pi,
    origin: Vec3 = (0.0, 0.0, 0.0),
    polar_axis: Vec3 = (0.0, 0.0, 1.0),
    azimuthal_axis: Vec3 = (1.0, 0.0, 0.0),
    *,
    n_r: int = 16,
    n_theta: int = 32,
    n_phi: int = 64,
) -> tuple[NDArrayF, NDArrayI]:
    """Generate a triangular mesh for a surface (of a volume) in spherical coordinates.

    This function generates a **watertight** (when geometrically possible),
    **outward-oriented** triangular surface mesh for a volume element
    that is rectangular in spherical coordinates `(r, θ, φ)`,
    but embedded in an arbitrary world coordinate frame
    (arbitrary origin and axis directions).

    It also supports **degenerate spans**, i.e. any of:
    - `r_min == r_max` (no radial thickness),
    - `theta_min == theta_max` (single polar angle),
    - `phi_min == phi_max` (single azimuth).

    In such cases, the output is no longer a closed 3D surface;
    instead, the function emits any **remaining 2D surfaces**
    that still have nonzero area.
    If no 2D surface remains (e.g., `d_r=0` and either `d_theta=0` or `d_phi=0`),
    the function returns empty arrays `(0×3, 0×3, 0×3)`.

    Parameters
    ----------
    r_min, r_max
        Radial bounds, `0 ≤ r_min ≤ r_max`.
    theta_min, theta_max
        Polar bounds, `0 ≤ theta_min ≤ theta_max ≤ π`.
    phi_min, phi_max
        Azimuthal bounds, `0 ≤ phi_min ≤ phi_max ≤ 2π`.
        The azimuthal angle is measured from the azimuthal axis
        toward the direction obtained by the right-hand rule.
    origin
        World-space origin of the spherical system.
    polar_axis
        World-space polar axis direction (need not be unit).
    azimuthal_axis
        World-space azimuthal axis direction (need not be unit).
        If not orthogonal to `polar_axis`,
        it will be orthogonalized by the Gram-Schmidt process,
        i.e., by projecting it onto the plane orthogonal to `polar_axis`.
    n_r, n_theta, n_phi
        Resolution (i.e., interval counts) along r, θ, and φ, respectively (all ≥ 1).

    Returns
    -------
    vertices
        2D float array of shape `(N, 3)`
        containing the coordinates `(x, y, z)` of `N` mesh vertices
        in the original world frame.
    faces
        2D integer array of shape `(M, 3)`
        containing the indices of the triangle vertices
        for `M` mesh faces (in CCW order).
    normals
        2D float array of shape `(N, 3)`
        containing the normals of the mesh vertices
        in the original world frame.

    Raises
    ------
    ValueError
        If inputs are invalid (e.g., degenerate axis, out-of-range angles/radii).

    Notes
    -----
    - The volume $V$ of a rectangular box in spherical coordinates
      is defined by ranges in (r, θ, φ):
      \[
          V = \left\{
              x(r, \theta, \phi) \in \mathbb{R}^3 :
              r \in [r_\text{min}, r_\text{max}],
              \theta \in [\theta_\text{min}, \theta_\text{max}],
              \phi \in [\phi_\text{min}, \phi_\text{max}]
          \right\}
      \]

      with
      - $r \in [0, \infty)$: radial distance from the origin,
      - $\theta \in [0, \pi]$: polar angle from the polar axis,
      - $\phi \in [0, 2\pi]$: azimuthal angle in the equatorial plane.

    - The surface of a spherical “box” consists of up to six patches:
      - two spherical (r-constant) caps at `r = r_min` and `r = r_max`
        (require `d_theta > 0` and `d_phi > 0`),
      - two conical (θ-constant) frustums at `θ = θ_min` and `θ = θ_max`
        (require `d_r > 0`, `d_phi > 0`, and θ must not be exactly a pole, `0` or `π`),
      - two meridional (φ-constant) wedges at `φ = φ_min` and `φ = φ_max`
        (require `d_r > 0`, `d_theta > 0`, and only when the azimuthal span is not a full `2π`)

    - Outward orientation:
        * outer cap (`r=r_max`): normal points +r → default winding;
        * inner cap (`r=r_min`): outward is −r → **flipped** winding;
        * θ=θ_max: outward is +θ → default winding;
        * θ=θ_min: outward is −θ → **flipped** winding;
        * φ=φ_max: outward is +φ → default winding;
        * φ=φ_min: outward is −φ → **flipped** winding.

    - To prevent azimuthal seams/gaps when `φ` spans a full revolution (≈ `2π`),
      azimuthal triangulation is done **periodically**.

    - To avoid “needle/diagonal through the axis” artifacts in wireframes,
      pole rows (θ=0 or π) are **never** included in the rectangular band
      that forms the spherical caps;
      instead, each present pole is triangulated via a **fan**
      (center vertex at the pole plus a fan to the first latitude ring).

    - The implementation is fully vectorized and constructs each surface patch
      on a quadrilateral grid that is triangulated in a consistent, outward-facing
      orientation. Degenerate ranges (zero measure) are automatically elided.
    """
    # ----------------------- permissive validation -----------------------
    vals = (r_min, r_max, theta_min, theta_max, phi_min, phi_max, *origin, *polar_axis)
    if not np.all(np.isfinite(vals)):
        raise ValueError("All scalar inputs must be finite.")
    if not (0.0 <= r_min <= r_max):
        raise ValueError("Require 0 ≤ r_min ≤ r_max.")
    if not (0.0 <= theta_min <= theta_max <= np.pi):
        raise ValueError("Require 0 ≤ theta_min ≤ theta_max ≤ π.")
    if not (0.0 <= phi_min <= phi_max <= 2 * np.pi):
        raise ValueError("Require 0 ≤ phi_min ≤ phi_max ≤ 2π.")
    if n_theta < 1 or n_phi < 1 or n_r < 1:
        raise ValueError("n_theta, n_phi, n_r must be ≥ 1.")

    # ----------------------- local frame definition ----------------------
    e1, e2, e3 = complete_orthonormal_basis(polar_axis=polar_axis, azimuthal_axis=azimuthal_axis)
    R = np.stack([e1, e2, e3], axis=1)
    o = np.asarray(origin, dtype=float)
    frame = Frame(R=R, o=o)

    # ----------------------- span analysis & gating ----------------------
    d_r = r_max - r_min
    d_theta = theta_max - theta_min
    d_phi = phi_max - phi_min

    # φ periodic detection (for meridional planes gating too)
    _, phi_full = phi_samples(phi_min, phi_max, n_phi)  # we only need phi_full here

    # Output accumulators
    vertices_blocks: list[NDArrayF] = []
    faces_blocks: list[NDArrayI] = []
    vtx_offset = 0

    def append_block(V: NDArrayF, F: NDArrayI) -> None:
        nonlocal vtx_offset
        if V.size == 0 or F.size == 0:
            return
        faces_blocks.append(F + vtx_offset)
        vertices_blocks.append(V)
        vtx_offset += V.shape[0]

    # ------------------------------ Caps ---------------------------------
    # Caps exist iff θ and φ both vary (2D parameterization available).
    if d_theta > 0.0 and d_phi > 0.0:
        # Outer cap (r = r_max), outward +r
        cap_Vs, cap_Fs = build_cap_band_and_fans(
            r_val=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
            n_theta=n_theta,
            n_phi=n_phi,
            frame=frame,
            flip_band=False,
            flip_fans=False,
        )
        for V, F in zip(cap_Vs, cap_Fs):
            append_block(V, F)

        # Inner cap (r = r_min) only if strictly inside (non-zero r); outward −r
        if r_min > 0.0 and r_min < r_max:
            cap_Vs, cap_Fs = build_cap_band_and_fans(
                r_val=r_min,
                theta_min=theta_min,
                theta_max=theta_max,
                phi_min=phi_min,
                phi_max=phi_max,
                n_theta=n_theta,
                n_phi=n_phi,
                frame=frame,
                flip_band=True,
                flip_fans=True,
            )
            for V, F in zip(cap_Vs, cap_Fs):
                append_block(V, F)

    # -------------------------- θ-constant cones --------------------------
    # Require r to vary (shell thickness) and φ to vary. Skip exact poles.
    if d_r > 0.0 and d_phi > 0.0:
        # θ = θ_max (outward +θ)
        surf = build_theta_const_surface(
            theta_val=theta_max,
            r_min=r_min,
            r_max=r_max,
            phi_min=phi_min,
            phi_max=phi_max,
            n_r=n_r,
            n_phi=n_phi,
            frame=frame,
            flip=False,
        )
        if surf is not None:
            V, F = surf
            append_block(V, F)

        # θ = θ_min (outward −θ) — may be equal to θ_max; helper handles poles.
        surf = build_theta_const_surface(
            theta_val=theta_min,
            r_min=r_min,
            r_max=r_max,
            phi_min=phi_min,
            phi_max=phi_max,
            n_r=n_r,
            n_phi=n_phi,
            frame=frame,
            flip=True,
        )
        if surf is not None:
            V, F = surf
            append_block(V, F)

    # ------------------------ φ-constant meridional -----------------------
    # Only when φ is not full 2π, and both r and θ vary.
    if not phi_full and d_r > 0.0 and d_theta > 0.0:
        # φ = φ_max (outward +φ)
        surf = build_phi_const_surface(
            phi_val=phi_max,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            n_r=n_r,
            n_theta=n_theta,
            frame=frame,
            flip=False,
        )
        if surf is not None:
            V, F = surf
            append_block(V, F)

        # φ = φ_min (outward −φ)
        surf = build_phi_const_surface(
            phi_val=phi_min,
            r_min=r_min,
            r_max=r_max,
            theta_min=theta_min,
            theta_max=theta_max,
            n_r=n_r,
            n_theta=n_theta,
            frame=frame,
            flip=True,
        )
        if surf is not None:
            V, F = surf
            append_block(V, F)

    # ------------------------------ assemble ------------------------------
    if not vertices_blocks:
        # Nothing 2D remains (e.g., r-only segment, or line/point degeneracy).
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=float)

    V_all = np.concatenate(vertices_blocks, axis=0)
    F_all = np.concatenate(faces_blocks, axis=0).astype(np.int32, copy=False)
    N_all = mesh_util.vertex_normal(V_all, F_all)
    return V_all, F_all, N_all


# ============================================================================
# Patch builders (caps, cones, meridional planes)
# ============================================================================

@dataclass(frozen=True)
class Frame:
    """Affine frame for local->world mapping."""
    R: NDArrayF  # 3x3 rotation matrix
    o: NDArrayF  # origin (3,)


def build_cap_band_and_fans(
    r_val: float,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    n_theta: int,
    n_phi: int,
    frame: Frame,
    *,
    flip_band: bool,
    flip_fans: bool,
) -> tuple[list[NDArrayF], list[NDArrayI]]:
    """Build a spherical cap at constant radius `r_val` as (band + pole fans).

    Parameters
    ----------
    r_val
        Radius of the cap.
    theta_min, theta_max
        Polar bounds (inclusive). Exact poles may be included.
    phi_min, phi_max
        Azimuth bounds.
    n_theta, n_phi
        Interval counts for θ and φ.
    frame
        World transform (rotation and origin).
    flip_band
        Reverse winding for the rectangular band (inner caps need this).
    flip_fans
        Reverse winding for the pole fans (inner caps need this).

    Returns
    -------
    vertices_parts, faces_parts
        Lists of vertex and face blocks for this cap.

    Notes
    -----
    - The rectangular band excludes exact pole rows, which are triangulated
      separately as triangle fans to a single pole vertex.
    - When `φ` spans a full 2π, periodic triangulation is used (no seam).
    """
    R, o = frame.R, frame.o
    vertices_parts: list[NDArrayF] = []
    faces_parts: list[NDArrayI] = []

    # θ and φ sampling
    th_edges = np.linspace(theta_min, theta_max, n_theta + 1)
    band, north, south = theta_band(th_edges)
    ph, phi_full = phi_samples(phi_min, phi_max, n_phi)

    # 1) Rectangular band (no pole rows)
    if band.size >= 2:
        if phi_full:
            TH, PH = np.meshgrid(band, ph, indexing="ij")  # (nu+1, nv)
            P = sph_to_cart_local(np.full_like(TH, r_val), TH, PH).reshape(-1, 3)
            P = to_world(P, R, o)
            F = grid_triangles_periodic_v(TH.shape[0] - 1, TH.shape[1], flip=flip_band)
        else:
            TH, PH = np.meshgrid(band, ph, indexing="ij")  # (nu+1, nv+1)
            P = sph_to_cart_local(np.full_like(TH, r_val), TH, PH).reshape(-1, 3)
            P = to_world(P, R, o)
            F = grid_triangles(TH.shape[0] - 1, TH.shape[1] - 1, flip=flip_band)
        vertices_parts.append(P)
        faces_parts.append(F)

    # 2) Pole fans where needed (north and/or south)
    #    We use the first ring away from the pole from th_edges: th_edges[1] or th_edges[-2].
    if phi_full:
        # periodic φ: ring has length n_phi
        if north and len(th_edges) >= 2:
            th_ring = th_edges[1]
            ring = sph_to_cart_local(np.full_like(ph, r_val), np.full_like(ph, th_ring), ph)
            pole = np.array([[0.0, 0.0, r_val]])
            P = to_world(np.vstack([pole, ring]), R, o)
            a = np.arange(1, 1 + ring.shape[0], dtype=int)
            b = np.roll(a, -1)
            F = np.stack([np.zeros_like(a), a, b], axis=1).astype(np.int32)
            if flip_fans:
                F = F[:, ::-1]
            vertices_parts.append(P)
            faces_parts.append(F)

        if south and len(th_edges) >= 2:
            th_ring = th_edges[-2]
            ring = sph_to_cart_local(np.full_like(ph, r_val), np.full_like(ph, th_ring), ph)
            pole = np.array([[0.0, 0.0, -r_val]])
            P = to_world(np.vstack([ring, pole]), R, o)
            ring_idx = np.arange(0, ring.shape[0], dtype=int)
            nidx = np.roll(ring_idx, -1)
            F = np.stack([ring_idx, np.full_like(ring_idx, ring.shape[0], int), nidx], axis=1).astype(np.int32)
            if flip_fans:
                F = F[:, ::-1]
            vertices_parts.append(P)
            faces_parts.append(F)
    else:
        # open φ: ring has length n_phi+1
        if north and len(th_edges) >= 2:
            th_ring = th_edges[1]
            ring = sph_to_cart_local(np.full_like(ph, r_val), np.full_like(ph, th_ring), ph)
            pole = np.array([[0.0, 0.0, r_val]])
            P = to_world(np.vstack([pole, ring]), R, o)
            a = np.arange(1, P.shape[0] - 1, dtype=int)
            b = a + 1
            F = np.stack([np.zeros_like(a), a, b], axis=1).astype(np.int32)
            if flip_fans:
                F = F[:, ::-1]
            vertices_parts.append(P)
            faces_parts.append(F)

        if south and len(th_edges) >= 2:
            th_ring = th_edges[-2]
            ring = sph_to_cart_local(np.full_like(ph, r_val), np.full_like(ph, th_ring), ph)
            pole = np.array([[0.0, 0.0, -r_val]])
            P = to_world(np.vstack([ring, pole]), R, o)
            a = np.arange(0, ring.shape[0] - 1, dtype=int)
            F = np.stack([a, np.full_like(a, ring.shape[0], int), a + 1], axis=1).astype(np.int32)
            if flip_fans:
                F = F[:, ::-1]
            vertices_parts.append(P)
            faces_parts.append(F)

    return vertices_parts, faces_parts


def build_theta_const_surface(
    theta_val: float,
    r_min: float,
    r_max: float,
    phi_min: float,
    phi_max: float,
    n_r: int,
    n_phi: int,
    frame: Frame,
    *,
    flip: bool,
) -> tuple[NDArrayF, NDArrayI] | None:
    """Build a θ-constant conical surface over `(r, φ)` if it has nonzero area.

    Parameters
    ----------
    theta_val
        The constant polar angle (not at a pole).
    r_min, r_max
        Radial bounds (must satisfy `r_max > r_min` for a 2D surface).
    phi_min, phi_max
        Azimuth bounds (must satisfy `phi_max > phi_min` for a 2D surface).
    n_r, n_phi
        Interval counts along r and φ.
    frame
        World transform (rotation and origin).
    flip
        Reverse winding (used for the θ_min surface to keep outward orientation).

    Returns
    -------
    vertices, faces | None
        The surface vertices and faces, or `None` if the surface is degenerate.

    Notes
    -----
    - If `theta_val` is exactly a pole (0 or π), the surface collapses to the axis.
    - For full 2π azimuth, periodic triangulation is used.
    """
    if np.isclose(theta_val, 0.0, atol=1e-14) or np.isclose(theta_val, np.pi, atol=1e-14):
        return None
    if not (r_max > r_min):
        return None
    d_phi = phi_max - phi_min
    if not (d_phi > 0.0):
        return None

    R, o = frame.R, frame.o
    ph, phi_full = phi_samples(phi_min, phi_max, n_phi)
    r_edges = np.linspace(r_min, r_max, n_r + 1)

    if phi_full:
        Rg, PH = np.meshgrid(r_edges, ph, indexing="ij")   # (n_r+1, n_phi)
        TH = np.full_like(Rg, theta_val)
        P = sph_to_cart_local(Rg, TH, PH).reshape(-1, 3)
        P = to_world(P, R, o)
        F = grid_triangles_periodic_v(n_r, n_phi, flip=flip)
    else:
        Rg, PH = np.meshgrid(r_edges, ph, indexing="ij")   # (n_r+1, n_phi+1)
        TH = np.full_like(Rg, theta_val)
        P = sph_to_cart_local(Rg, TH, PH).reshape(-1, 3)
        P = to_world(P, R, o)
        F = grid_triangles(n_r, n_phi, flip=flip)

    return P, F


def build_phi_const_surface(
    phi_val: float,
    r_min: float,
    r_max: float,
    theta_min: float,
    theta_max: float,
    n_r: int,
    n_theta: int,
    frame: Frame,
    *,
    flip: bool,
) -> tuple[NDArrayF, NDArrayI] | None:
    """Build a φ-constant meridional plane over `(r, θ)` if it has nonzero area.

    Parameters
    ----------
    phi_val
        The constant azimuth value.
    r_min, r_max
        Radial bounds (require `r_max > r_min`).
    theta_min, theta_max
        Polar bounds (require `theta_max > theta_min`).
    n_r, n_theta
        Interval counts along r and θ.
    frame
        World transform (rotation and origin).
    flip
        Reverse winding (the φ_min side is flipped to keep outward orientation).

    Returns
    -------
    vertices, faces | None
        The surface vertices and faces, or `None` if the surface is degenerate.

    Notes
    -----
    This surface is **not** emitted when the azimuth span is a full 2π, since the
    φ-constant planes would be redundant boundaries in that case.
    """
    if not (r_max > r_min):
        return None
    if not (theta_max > theta_min):
        return None

    R, o = frame.R, frame.o
    r_edges = np.linspace(r_min, r_max, n_r + 1)
    th_edges = np.linspace(theta_min, theta_max, n_theta + 1)

    Rg, TH = np.meshgrid(r_edges, th_edges, indexing="ij")  # (n_r+1, n_theta+1)
    PH = np.full_like(Rg, phi_val)
    P = sph_to_cart_local(Rg, TH, PH).reshape(-1, 3)
    P = to_world(P, R, o)
    F = grid_triangles(n_r, n_theta, flip=flip)

    return P, F


# ============================================================================
# Grid triangulation helpers
# ============================================================================

def grid_triangles(nu: int, nv: int, *, flip: bool = False) -> NDArrayI:
    """Triangulate a regular (nu × nv) **non-periodic** quad grid.

    Vertex layout is row-major over a `(nu+1) × (nv+1)` grid; two triangles per
    quad, yielding `2 * nu * nv` faces.

    Parameters
    ----------
    nu
        Number of intervals along the first grid axis (U).
    nv
        Number of intervals along the second grid axis (V).
    flip
        If True, reverse winding (useful to flip surface normals).

    Returns
    -------
    faces
        `(2*nu*nv, 3)` array of triangle vertex indices (0-based).
    """
    U = np.arange(nu)
    V = np.arange(nv)
    u, v = np.meshgrid(U, V, indexing="ij")

    i00 = (u * (nv + 1) + v).ravel()
    i10 = ((u + 1) * (nv + 1) + v).ravel()
    i01 = (u * (nv + 1) + (v + 1)).ravel()
    i11 = ((u + 1) * (nv + 1) + (v + 1)).ravel()

    if not flip:
        t1 = np.stack([i00, i10, i11], axis=1)
        t2 = np.stack([i00, i11, i01], axis=1)
    else:
        t1 = np.stack([i00, i11, i10], axis=1)
        t2 = np.stack([i00, i01, i11], axis=1)

    return np.concatenate([t1, t2], axis=0).astype(np.int32, copy=False)


def grid_triangles_periodic_v(nu: int, nv: int, *, flip: bool = False) -> NDArrayI:
    """Triangulate a regular (nu × nv) grid **periodic in V** (wrap last col to first).

    Vertex layout is row-major over a `(nu+1) × nv` grid (note: *no duplicated*
    last column). V is treated modulo `nv`.

    Parameters
    ----------
    nu
        Number of intervals along U.
    nv
        Number of intervals along periodic V (also number of columns).
    flip
        If True, reverse winding.

    Returns
    -------
    faces
        `(2*nu*nv, 3)` array of triangle indices.
    """
    # helper to index into a (nu+1) × nv grid where v wraps modulo nv
    def idx(u: NDArrayF, v: NDArrayF) -> NDArrayI:
        return (u * nv + (v % nv)).ravel().astype(np.int32, copy=False)

    U = np.arange(nu)[:, None]      # shape (nu, 1)
    V = np.arange(nv)[None, :]      # shape (1, nv)

    u = np.broadcast_to(U, (nu, nv))
    v = np.broadcast_to(V, (nu, nv))

    i00 = idx(u, v)
    i10 = idx(u + 1, v)
    i01 = idx(u, v + 1)
    i11 = idx(u + 1, v + 1)

    if not flip:
        t1 = np.stack([i00, i10, i11], axis=1)
        t2 = np.stack([i00, i11, i01], axis=1)
    else:
        t1 = np.stack([i00, i11, i10], axis=1)
        t2 = np.stack([i00, i01, i11], axis=1)

    return np.concatenate([t1, t2], axis=0).astype(np.int32, copy=False)


# ============================================================================
# Sampling utilities
# ============================================================================

def phi_samples(phi_min: float, phi_max: float, n_phi: int) -> tuple[NDArrayF, bool]:
    """Sample azimuth `φ` either as a closed grid or periodic grid for full 2π.

    Parameters
    ----------
    phi_min, phi_max
        Azimuth bounds in radians with `0 ≤ phi_min ≤ phi_max ≤ 2π`.
    n_phi
        Number of **intervals** along φ.

    Returns
    -------
    ph, phi_full
        - `ph`: 1D array of φ samples.
          * If full 2π (within tight tolerance), length is `n_phi`
            (no duplicated endpoint), meant for **periodic** triangulation.
          * Otherwise, length is `n_phi+1` (both ends included).
        - `phi_full`: `True` if `(phi_max - phi_min)` is effectively `2π`.

    Notes
    -----
    Full 2π is detected modulo 2π to be robust against round-off.
    """
    d_phi = phi_max - phi_min
    two_pi = 2.0 * np.pi
    span_mod = d_phi % two_pi
    span_mod = min(span_mod, two_pi - span_mod)
    phi_full = span_mod <= 1e-12

    if phi_full:
        # n_phi **distinct** columns in [phi_min, phi_min+2π)
        ph = phi_min + (d_phi / n_phi) * np.arange(n_phi, dtype=float)
    else:
        # closed grid with duplicated endpoint
        ph = np.linspace(phi_min, phi_max, n_phi + 1)

    return ph, phi_full


def theta_band(th_edges: NDArrayF) -> tuple[NDArrayF, bool, bool]:
    """Return the θ band for cap rectangular grids and which poles are included.

    The band *excludes* the exact pole rows that are actually present (to avoid
    collapsed rows). The opposite θ boundary remains included to properly join
    with θ-constant surfaces.

    Parameters
    ----------
    th_edges
        1D array of θ samples of length `n_theta+1`.

    Returns
    -------
    band, north, south
        - `band`: θ samples used for the rectangular cap band.
        - `north`: True if θ=0 is included in `th_edges`.
        - `south`: True if θ=π is included in `th_edges`.
    """
    north = np.isclose(th_edges[0], 0.0, atol=1e-14)
    south = np.isclose(th_edges[-1], np.pi, atol=1e-14)

    if north and south:
        band = th_edges[1:-1]
    elif north and not south:
        band = th_edges[1:]   # include θ_max row
    elif south and not north:
        band = th_edges[:-1]  # include θ_min row
    else:
        band = th_edges
    return band, north, south


# ============================================================================
# Linear algebra & coordinate helpers
# ============================================================================

def complete_orthonormal_basis(
    polar_axis: NDArrayF,
    azimuthal_axis: NDArrayF,
) -> tuple[NDArrayF, NDArrayF, NDArrayF]:
    """Construct an orthonormal basis (ONB) in 3D space from polar and azimuthal axes.

    This function generates a right-handed ONB (e1, e2, e3) that satisfies:
    - `e3` is the unit vector aligned with `polar_axis`,
    - `e1` is the projection of `azimuthal_axis` into the equatorial plane orthogonal to `e3`,
    - `e1` and `e2` are unit-length, orthogonal to `e3` and to each other,
    - right-handed orientation: `(e1 × e2) · e3 > 0`.

    Parameters
    ----------
    polar_axis
        3-vector (world coordinates) specifying the desired (positive) polar axis direction.
    azimuthal_axis
        3-vector (world coordinates) specifying the desired (positive) azimuthal axis direction.
        If nearly parallel to `polar_axis`, use world +X, then +Y as fallback.

    Returns
    -------
    e1, e2, e3
        Three 3D unit vectors forming an orthonormal frame.
    """
    polar = np.asarray(polar_axis, dtype=float)
    azimuthal = np.asarray(azimuthal_axis, dtype=float)

    polar_norm = np.linalg.norm(polar)
    if not np.isfinite(polar_norm) or polar_norm == 0.0:
        raise ValueError("polar_axis must be a non-zero, finite vector.")

    azimuthal_norm = np.linalg.norm(azimuthal)
    if not np.isfinite(azimuthal_norm) or azimuthal_norm == 0.0:
            raise ValueError("azimuth_axis must be finite and non-zero.")

    # Set e3 ‖ polar_axis
    e3 = polar / polar_norm

    # Set e1
    # Project candidate into equatorial plane and normalize → e1
    e1 = azimuthal - np.dot(azimuthal, e3) * e3
    n1 = np.linalg.norm(e1)
    if n1 < 1e-15:
        # candidate parallel to e3; pick a robust fallback orthogonal to e3
        helper = np.array([1.0, 0.0, 0.0]) if abs(e3[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = helper - np.dot(helper, e3) * e3
        n1 = np.linalg.norm(e1)
        if n1 < 1e-15:
            helper = np.array([0.0, 0.0, 1.0])
            e1 = helper - np.dot(helper, e3) * e3
            n1 = np.linalg.norm(e1)
            if n1 < 1e-15:
                raise ValueError("Failed to construct azimuth basis.")
    e1 /= n1

    # Set e2 = e3 × e1 to complete the right-handed frame
    e2 = np.cross(e3, e1)
    return e1, e2, e3


def sph_to_cart_local(r: NDArrayF, th: NDArrayF, ph: NDArrayF) -> NDArrayF:
    """Vectorized local spherical `(r, θ, φ)` to local Cartesian `(x, y, z)`.

    Parameters
    ----------
    r
        Radius array (broadcastable against `th` and `ph`).
    th
        Polar angles θ (0..π), array broadcastable with `r` and `ph`.
    ph
        Azimuth angles φ (0..2π), array broadcastable with `r` and `th`.

    Returns
    -------
    array
        Stacked coordinates `(x, y, z)` with last dimension size 3.

    Notes
    -----
    Local coordinates are defined with polar axis along +z (before rotation).
    """
    st = np.sin(th)
    x = r * st * np.cos(ph)
    y = r * st * np.sin(ph)
    z = r * np.cos(th)
    return np.stack([x, y, z], axis=-1)


def to_world(P_local: NDArrayF, R: NDArrayF, o: NDArrayF) -> NDArrayF:
    """Apply rotation/translation to map local coordinates into world frame.

    Parameters
    ----------
    P_local
        Points in local frame, shape `(..., 3)`.
    R
        3×3 rotation matrix whose columns are the world-frame images of local
        basis vectors (x->R[:,0], y->R[:,1], z->R[:,2]).
    o
        3-vector world-space translation (origin of spherical system).

    Returns
    -------
    array
        Points in world frame, same leading shape as `P_local`.
    """
    return P_local @ R.T + o
