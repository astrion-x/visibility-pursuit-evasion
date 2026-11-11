"""Floating polynomial event detection during one held-control interval."""

import numpy as np
from numpy.polynomial import Polynomial as Poly
from .geometry import number, project_ellipse_exterior


def cross(a, b):
    return a[0] * b[1] - a[1] * b[0]


def event_times(geometry, p, e, dp, de, radius, duration):
    """Potential physical, visibility and range changes of two affine paths."""
    pp = [Poly([p[j], dp[j]]) for j in range(2)]
    ee = [Poly([e[j], de[j]]) for j in range(2)]
    dd = [ee[j] - pp[j] for j in range(2)]
    atoms = [sum(x * x for x in dd) - radius * radius, *dd]
    all_points = np.array([p, e, p + duration * dp, e + duration * de])
    low, high = all_points.min(axis=0), all_points.max(axis=0)
    for polygon, center, axes, unit in geometry.parts:
        ex = np.sqrt((axes[0] * unit[0]) ** 2 + (axes[1] * unit[1]) ** 2)
        ey = np.sqrt((axes[0] * unit[1]) ** 2 + (axes[1] * unit[0]) ** 2)
        ol = np.minimum(polygon.min(axis=0), center - [ex, ey])
        oh = np.maximum(polygon.max(axis=0), center + [ex, ey])
        if np.any(high < ol) or np.any(low > oh):
            continue
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            edge = b - a
            atoms.extend(
                [
                    cross(edge, [pp[j] - a[j] for j in range(2)]),
                    cross(edge, [ee[j] - a[j] for j in range(2)]),
                    cross(dd, [a[j] - pp[j] for j in range(2)]),
                ]
            )
            atoms.extend(point[j] - a[j] for point in (pp, ee) for j in range(2))
        x = [
            (unit[0] * (pp[0] - center[0]) + unit[1] * (pp[1] - center[1])) / axes[0],
            (-unit[1] * (pp[0] - center[0]) + unit[0] * (pp[1] - center[1])) / axes[1],
        ]
        v = [
            (unit[0] * dd[0] + unit[1] * dd[1]) / axes[0],
            (-unit[1] * dd[0] + unit[0] * dd[1]) / axes[1],
        ]
        a = sum(q * q for q in v)
        b = 2 * sum(x[j] * v[j] for j in range(2))
        c = sum(q * q for q in x) - 1
        atoms.extend([a, b, c, a + b + c, b + 2 * a, 4 * a * c - b * b])
    cuts = [0.0, float(duration)]
    for atom in atoms:
        coeff = atom.trim().coef
        if len(coeff) <= 1:
            continue
        for root in Poly(coeff).roots():
            if (
                abs(np.imag(root)) < 1e-8
                and -1e-10 <= np.real(root) <= duration + 1e-10
            ):
                cuts.append(float(np.clip(np.real(root), 0, duration)))
    return np.unique(cuts)


def events(geometry, p, e, dp, de, radius, duration):
    cuts = event_times(geometry, p, e, dp, de, radius, duration)
    samples = sorted(
        [(x, x) for x in cuts]
        + [(a, (a + b) / 2) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]
    )
    ts = np.array([sample for _, sample in samples])
    ps, es = p + ts[:, None] * dp, e + ts[:, None] * de
    physical = geometry.physical(ps) & geometry.physical(es)
    visible = geometry.segment_physical(ps, es)
    close = np.sum((ps - es) ** 2, axis=1) <= radius**2 + 1e-11
    for i, (left, sample) in enumerate(samples):
        if not physical[i]:
            return dict(kind="collision", time=float(left), witness_time=float(sample))
        if visible[i] and close[i]:
            return dict(kind="capture", time=float(left), witness_time=float(sample))
        if not visible[i] and not close[i]:
            return dict(kind="escape", time=float(left), witness_time=float(sample))
    return None


def squared_obstacle_distance(geometry, point):
    """Squared exterior distance to the polygon/ellipse union."""
    best = float("inf")
    for polygon, center, axes, unit in geometry.parts:
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            d = b - a
            t = np.clip(np.dot(point - a, d) / np.dot(d, d), 0, 1)
            best = min(best, float(np.dot(point - a - t * d, point - a - t * d)))
        v = point - center
        x = np.array(
            [unit[0] * v[0] + unit[1] * v[1], -unit[1] * v[0] + unit[0] * v[1]]
        )
        if np.sum((x / axes) ** 2) <= 1 + 1e-12:
            distance = 0.0
        else:
            q = project_ellipse_exterior(x, axes)
            distance = float(np.dot(x - q, x - q))
        best = min(best, distance)
    return best


def advance_interval(geometry, p, e, u_p, u_e, parameters):
    vp, ve, h, radius = [
        number(parameters[k])
        for k in ("pursuer_speed", "evader_speed", "sample_period", "capture_radius")
    ]
    if hasattr(geometry, "certainly_continues"):
        next_p = p + vp * h * u_p
        next_e = e + ve * h * u_e
        if geometry.certainly_continues(p, e, next_p, next_e, radius):
            return next_p, next_e, None
    if not (
        geometry.segment_physical(p, p + vp * h * u_p)
        and geometry.segment_physical(e, e + ve * h * u_e)
    ):
        raise ValueError("illegal selected command")
    event = events(geometry, p, e, vp * u_p, ve * u_e, radius, h)
    if event and event["kind"] == "collision":
        raise ValueError("collision in continuous event check")
    duration = event["time"] if event else h
    return p + vp * duration * u_p, e + ve * duration * u_e, event
