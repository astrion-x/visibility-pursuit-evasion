"""Reproducible random initial-state generation."""

import numpy as np
from .geometry import ObstacleGeometry
from .events import squared_obstacle_distance
from .config import load_scenes, SCENE_CODES


def sight_clearance(geometry, p, e):
    """Distance of a visible segment to the union (including endpoint minima)."""
    segment = e - p
    length = np.linalg.norm(segment)
    along = segment / length
    normal = np.array([-along[1], along[0]])
    best = np.sqrt(min(squared_obstacle_distance(geometry, v) for v in (p, e)))
    for vertices, center, axes, axis in geometry.parts:
        # For disjoint segments some nearest point is an endpoint. For an
        # ellipse, an interior minimizer has a tangent parallel to the segment.
        perpendicular = np.array([-axis[1], axis[0]])
        rotation = np.column_stack([axis, perpendicular])
        local = rotation.T @ normal
        radial = rotation @ (axes**2 * local) / np.linalg.norm(axes * local)
        points = np.vstack([vertices, center + radial, center - radial])
        projected = np.clip((points - p) @ along, 0, length)
        best = min(
            best, np.linalg.norm(points - p - projected[:, None] * along, axis=1).min()
        )
    return float(best)


def generate(scene, seed, *, geometry=None, max_attempts=100000):
    """Draw a fixed distance, then reject invalid position/orientation pairs.

    Scene codes identify position streams; all scenes share the distance stream.
    Acceptance requires valid positions and a visible segment with positive clearance.
    """
    if scene not in SCENE_CODES or seed < 0:
        raise ValueError("Use scene A, B, C or D and a nonnegative seed")
    geometry = (
        geometry if geometry is not None else ObstacleGeometry(load_scenes(), scene)
    )
    distance_rng = np.random.default_rng(np.random.SeedSequence([1, seed]))
    position_rng = np.random.default_rng(
        np.random.SeedSequence([2, seed, SCENE_CODES[scene]])
    )
    distance = float(distance_rng.uniform(3.0, 6.0))
    for _ in range(max_attempts):
        p = position_rng.uniform(-10.0, 10.0, 2)
        angle = position_rng.uniform(0.0, 2 * np.pi)
        e = p + distance * np.array([np.cos(angle), np.sin(angle)])
        if np.any(e < -10.0) or np.any(e > 10.0):
            continue
        if not (
            geometry.physical(p)
            and geometry.physical(e)
            and geometry.segment_physical(p, e)
        ):
            continue
        if sight_clearance(geometry, p, e) <= 1e-8:
            continue
        return dict(
            scene=scene,
            seed=int(seed),
            P=p.tolist(),
            E=e.tolist(),
            initial_distance_m=distance,
        )
    raise RuntimeError("Initial-state rejection limit reached")
