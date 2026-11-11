"""Unbounded polygon/ellipse geometry and the obstacle distance field."""

from fractions import Fraction
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates


def number(x):
    return float(Fraction(str(x)))


def project_ellipse_exterior(local, axes):
    """Project an exterior point onto an ellipse in its principal coordinates."""

    def equation(multiplier):
        return np.sum((axes * local / (axes * axes + multiplier)) ** 2) - 1

    low, high = 0.0, max(1.0, float(np.linalg.norm(axes * local)))
    while equation(high) > 0:
        high *= 2
    for _ in range(55):
        middle = (low + high) / 2
        if equation(middle) > 0:
            low = middle
        else:
            high = middle
    return axes * axes * local / (axes * axes + (low + high) / 2)


def polygon_inside(points, vertices):
    """Vectorized even/odd numerical polygon membership, with boundary tolerance."""
    p = np.asarray(points)
    inside = np.zeros(p.shape[:-1], dtype=bool)
    boundary = inside.copy()
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        edge = b - a
        d = p - a
        cross = edge[0] * d[..., 1] - edge[1] * d[..., 0]
        boundary |= (
            (np.abs(cross) <= 1e-10)
            & (np.sum(d * edge, axis=-1) >= -1e-10)
            & (np.sum((p - b) * edge, axis=-1) <= 1e-10)
        )
        if edge[1] != 0:
            hit = (a[1] > p[..., 1]) != (b[1] > p[..., 1])
            xhit = a[0] + (p[..., 1] - a[1]) * edge[0] / edge[1]
            inside ^= hit & (p[..., 0] < xhit)
    return inside & ~boundary


def validate_component_union(config):
    """Exact sufficient input check for the component-interior shortcut.

    At a supported convex vertex both interiors locally lie on one side
    of the ellipse tangent, even for a nonconvex polygon.
    """

    def rational(x):
        return Fraction(str(x))

    def cross(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    def on_segment(a, b, p):
        return cross(a, b, p) == 0 and all(
            min(a[j], b[j]) <= p[j] <= max(a[j], b[j]) for j in (0, 1)
        )

    def intersect(a, b, c, d):
        return (
            cross(a, b, c) * cross(a, b, d) < 0 and cross(c, d, a) * cross(c, d, b) < 0
        ) or any(
            (
                on_segment(a, b, c),
                on_segment(a, b, d),
                on_segment(c, d, a),
                on_segment(c, d, b),
            )
        )

    boxes = []
    for obstacle in config["map_snapshot"]["cases"][config["case"]]["obstacles"]:
        vertices = [tuple(map(rational, p)) for p in obstacle["polygon"]]
        n = len(vertices)
        if (
            n < 3
            or len(set(vertices)) != n
            or any(
                cross(vertices[i - 1], vertices[i], vertices[(i + 1) % n]) == 0
                for i in range(n)
            )
        ):
            raise ValueError(
                "obstacle geometry requires a simple polygon without flat vertices"
            )
        edges = list(zip(vertices, vertices[1:] + vertices[:1]))
        if any(
            intersect(*edges[i], *edges[j])
            for i in range(n)
            for j in range(i + 1, n)
            if j != i + 1 and (i, j) != (0, n - 1)
        ):
            raise ValueError("obstacle geometry requires simple polygons")
        el = obstacle["ellipse"]
        center = tuple(map(rational, el["center"]))
        a, b = map(rational, el["semiaxes"])
        u, v = map(rational, el["axis"])
        if not (a > 0 and b > 0 and u * u + v * v == 1):
            raise ValueError(
                "obstacle geometry requires positive axes and a unit orientation"
            )
        orientation = sum(start[0] * end[1] - end[0] * start[1] for start, end in edges)
        for idx, p in enumerate(vertices):
            dx, dy = p[0] - center[0], p[1] - center[1]
            major, minor = u * dx + v * dy, -v * dx + u * dy
            if b * b * major**2 + a * a * minor**2 == a * a * b * b:
                nx = b * b * major * u - a * a * minor * v
                ny = b * b * major * v + a * a * minor * u
                prev, following = vertices[idx - 1], vertices[(idx + 1) % n]
                if orientation * cross(prev, p, following) <= 0 or any(
                    nx * (x - p[0]) + ny * (y - p[1]) > 0 for x, y in (prev, following)
                ):
                    raise ValueError("unsupported ellipse contact at a polygon vertex")
        # Rational outward bounds; no square-root rounding affects separation.
        extents = (abs(a * u) + abs(b * v), abs(a * v) + abs(b * u))
        boxes.append(
            tuple(
                (
                    min(min(p[j] for p in vertices), center[j] - extents[j]),
                    max(max(p[j] for p in vertices), center[j] + extents[j]),
                )
                for j in (0, 1)
            )
        )
    if any(
        not any(left[j][1] < right[j][0] or right[j][1] < left[j][0] for j in (0, 1))
        for i, left in enumerate(boxes)
        for right in boxes[i + 1 :]
    ):
        raise ValueError("obstacle bounding boxes must be separated")


class ObstacleGeometry:
    def __init__(self, snapshot, case):
        validate_component_union({"map_snapshot": snapshot, "case": case})
        self.parts = []
        for ob in snapshot["cases"][case]["obstacles"]:
            el = ob["ellipse"]
            self.parts.append(
                tuple(
                    np.asarray(x, dtype=float)
                    for x in (
                        [[number(v) for v in p] for p in ob["polygon"]],
                        [number(v) for v in el["center"]],
                        [number(v) for v in el["semiaxes"]],
                        [number(v) for v in el["axis"]],
                    )
                )
            )

    def physical(self, points):
        points = np.asarray(points)
        free = np.ones(points.shape[:-1], dtype=bool)
        for poly, center, axes, unit in self.parts:
            d = points - center
            x = d[..., 0] * unit[0] + d[..., 1] * unit[1]
            y = -d[..., 0] * unit[1] + d[..., 1] * unit[0]
            free &= ~polygon_inside(points, poly)
            free &= (x / axes[0]) ** 2 + (y / axes[1]) ** 2 >= 1 - 1e-12
        return free

    def segment_physical(self, start, end):
        """Numerical interval splitting for polygons; quadratic min for ellipses.

        Geometry is evaluated in floating arithmetic with boundary tolerances.
        Both endpoints may be arrays with a common shape (..., 2).
        """
        start, end = np.broadcast_arrays(np.asarray(start), np.asarray(end))
        shape = start.shape[:-1]
        start, end = start.reshape(-1, 2), end.reshape(-1, 2)
        ok = self.physical(start) & self.physical(end)
        direction = end - start
        for poly, center, axes, unit in self.parts:
            cuts = [np.zeros(len(start)), np.ones(len(start))]
            for a, b in zip(poly, np.roll(poly, -1, axis=0)):
                edge = b - a
                numerator = edge[0] * (start[:, 1] - a[1]) - edge[1] * (
                    start[:, 0] - a[0]
                )
                denominator = edge[0] * direction[:, 1] - edge[1] * direction[:, 0]
                t = np.zeros(len(start))
                np.divide(
                    -numerator, denominator, out=t, where=np.abs(denominator) > 1e-14
                )
                cuts.append(np.clip(t, 0, 1))
            cuts = np.sort(np.stack(cuts, axis=1), axis=1)
            for j in range(cuts.shape[1] - 1):
                t = (cuts[:, j] + cuts[:, j + 1]) / 2
                ok &= ~polygon_inside(start + t[:, None] * direction, poly)
            rotation = np.array([[unit[0], -unit[1]], [unit[1], unit[0]]])
            local = ((start - center) @ rotation) / axes
            slope = (direction @ rotation) / axes
            aa = np.sum(slope * slope, axis=1)
            bb = np.sum(local * slope, axis=1)
            t = np.zeros(len(start))
            np.divide(-bb, aa, out=t, where=aa > 1e-25)
            t = np.clip(t, 0, 1)
            ok &= np.sum((local + t[:, None] * slope) ** 2, axis=1) >= 1 - 1e-12
        return ok.reshape(shape)


class DistanceGeometry(ObstacleGeometry):
    def __init__(self, snapshot, case, starts, *, parameters, settings):
        super().__init__(snapshot, case)
        spacing = settings.map_spacing
        # Cover all motion up to the cutoff, including the final forecast.
        duration = number(parameters["observation_seconds"]) + (
            settings.prediction_steps * number(parameters["sample_period"])
        )
        speed = max(number(parameters[k]) for k in ("pursuer_speed", "evader_speed"))
        margin = speed * duration + 1.0
        ext = [np.asarray(starts) - margin, np.asarray(starts) + margin]
        for poly, c, a, u in self.parts:
            ext.extend([poly, [c - np.max(a) - 2, c + np.max(a) + 2]])
        pts = np.vstack(ext)
        self.spacing = spacing
        self.origin = np.floor(pts.min(0) / spacing) * spacing
        high = np.ceil(pts.max(0) / spacing) * spacing
        self.shape = tuple(
            (np.rint((high - self.origin) / spacing).astype(int) + 1).tolist()
        )
        free = np.empty(self.shape, dtype=bool)
        yy = self.origin[1] + spacing * np.arange(self.shape[1])
        for begin in range(0, self.shape[0], 64):
            xx = self.origin[0] + spacing * np.arange(
                begin, min(begin + 64, self.shape[0])
            )
            grid = np.stack(np.meshgrid(xx, yy, indexing="ij"), -1)
            free[begin : begin + len(xx)] = self.physical(grid)
        self.sdf = distance_transform_edt(free, sampling=spacing).astype("float32")
        self.sdf -= distance_transform_edt(~free, sampling=spacing).astype("float32")

    def distance(self, points):
        p = np.asarray(points)
        coords = ((p.reshape(-1, 2) - self.origin) / self.spacing).T
        if np.any(coords < 0) or np.any(coords > (np.array(self.shape) - 1)[:, None]):
            raise ValueError("distance query outside padded analysis grid")
        return map_coordinates(
            self.sdf, coords, order=1, mode="nearest", prefilter=False
        ).reshape(p.shape[:-1])
