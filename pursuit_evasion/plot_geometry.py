"""Vector outlines of the polygon/ellipse obstacle unions."""

from dataclasses import dataclass
import math
import numpy as np
import sympy as sp


@dataclass(frozen=True)
class Composite:
    vertices: tuple
    center: tuple
    semiaxes: tuple
    axis: tuple

    def ellipse_value(self, point):
        dx, dy = (point[j] - self.center[j] for j in (0, 1))
        a, b = self.semiaxes
        c, s = self.axis
        return sp.expand(
            b * b * (c * dx + s * dy) ** 2
            + a * a * (-s * dx + c * dy) ** 2
            - a * a * b * b
        )


def composite(obstacle):
    def rational(values):
        return tuple(sp.Rational(str(v)) for v in values)

    ellipse = obstacle["ellipse"]
    return Composite(
        tuple(rational(p) for p in obstacle["polygon"]),
        *(rational(ellipse[k]) for k in ("center", "semiaxes", "axis"))
    )


def obstacle_bounds_center(obstacles):
    """Center of the axis-aligned bounding box of the polygon--ellipse unions."""
    limits = []
    for obstacle in obstacles:
        a, b = map(float, obstacle.semiaxes)
        c, s = map(float, obstacle.axis)
        extent = np.hypot([a * c, a * s], [b * s, b * c])
        center = np.asarray(obstacle.center, dtype=float)
        limits.extend(
            [
                np.asarray(obstacle.vertices, dtype=float),
                [center - extent, center + extent],
            ]
        )
    points = np.vstack(limits)
    return (points.min(axis=0) + points.max(axis=0)) / 2


def exposed_boundary(obstacle):
    """Split the boundary at every exact line--ellipse intersection."""
    t = sp.Symbol("t", real=True)
    segments, intersections = [], []
    for start, end in zip(
        obstacle.vertices, obstacle.vertices[1:] + obstacle.vertices[:1]
    ):
        point = tuple(start[j] + t * (end[j] - start[j]) for j in (0, 1))
        q = sp.Poly(obstacle.ellipse_value(point), t)
        aa, bb, cc = q.all_coeffs()
        discriminant = bb * bb - 4 * aa * cc
        roots = []
        if discriminant >= 0:
            roots = sorted(
                {(-bb + sign * sp.sqrt(discriminant)) / (2 * aa) for sign in (-1, 1)}
            )
            roots = [root for root in roots if 0 <= root <= 1]
        cuts = sorted(set([sp.S.Zero, *roots, sp.S.One]))
        at = lambda value: tuple(v.subs(t, value) for v in point)
        intersections.extend(at(root) for root in roots)
        for low, high in zip(cuts, cuts[1:]):
            if q.eval((low + high) / 2) > 0:
                segments.append((at(low), at(high)))

    center = np.array(obstacle.center, dtype=float)
    a, b = map(float, obstacle.semiaxes)
    c, s = map(float, obstacle.axis)
    rotation = np.array([[c, -s], [s, c]])
    angles = [0.0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi]
    for point in intersections:
        local = rotation.T @ (np.array(point, dtype=float) - center)
        angle = math.atan2(local[1] / b, local[0] / a) % (2 * math.pi)
        angles.append(angle)
    angles.sort()
    cuts = [angles[0]]
    for angle in angles[1:]:
        if angle - cuts[-1] > 1e-12:
            cuts.append(angle)

    def outside_polygon(point):
        # Used only to select visible vector arcs, never as a physical predicate.
        winding = 0
        vertices = np.array(obstacle.vertices, dtype=float)
        for start, end in zip(vertices, np.roll(vertices, -1, axis=0)):
            edge, offset = end - start, point - start
            cross = edge[0] * offset[1] - edge[1] * offset[0]
            parameter = np.dot(offset, edge) / np.dot(edge, edge)
            if 0 <= parameter <= 1 and abs(cross) / np.linalg.norm(edge) < 1e-10:
                raise ValueError("An arc midpoint is too close to a polygon edge")
            if start[1] <= point[1] < end[1] and cross > 0:
                winding += 1
            if end[1] <= point[1] < start[1] and cross < 0:
                winding -= 1
        return winding == 0

    arcs = []
    for low, high in zip(cuts, cuts[1:]):
        angle = (low + high) / 2
        point = center + rotation @ np.array([a * math.cos(angle), b * math.sin(angle)])
        if outside_polygon(point):
            arcs.append((low, high))
    if not segments or not arcs:
        raise ValueError(
            "Each obstacle must show both straight and curved outer boundary"
        )
    return segments, arcs


def closed_outline(obstacle):
    """Join the exposed straight/cubic pieces into one closed display path.

    Match only roundoff-sized endpoint differences; do not bridge geometric
    gaps. Retain the original elliptic Bezier arcs and polygon segments.
    This changes rendering only, never the input or numerical map predicates.
    """
    from matplotlib.path import Path

    segments, arcs = exposed_boundary(obstacle)
    pieces = [
        (np.array([p, q], dtype=float), np.array([Path.MOVETO, Path.LINETO]))
        for p, q in segments
    ]
    a, b = map(float, obstacle.semiaxes)
    c, s = map(float, obstacle.axis)
    rotation = np.array([[c, -s], [s, c]])
    center = np.array(obstacle.center, dtype=float)
    for low, high in arcs:
        arc = Path.arc(math.degrees(low), math.degrees(high))
        points = (arc.vertices * [a, b]) @ rotation.T + center
        pieces.append((points, arc.codes.copy()))

    scale = max(1.0, max(np.abs(points).max() for points, _ in pieces))
    tolerance = 128 * np.finfo(float).eps * scale
    nodes, endpoints = [], []
    for points, _ in pieces:
        pair = []
        for point in (points[0], points[-1]):
            matches = [
                j
                for j, node in enumerate(nodes)
                if np.linalg.norm(point - node) <= tolerance
            ]
            if len(matches) > 1:
                raise ValueError("Ambiguous outline junction")
            if not matches:
                nodes.append(point.copy())
                matches = [len(nodes) - 1]
            pair.append(matches[0])
        if pair[0] == pair[1]:
            raise ValueError("Degenerate outline piece")
        endpoints.append(pair)
    incident = [[] for _ in nodes]
    for j, pair in enumerate(endpoints):
        for node in pair:
            incident[node].append(j)
    if any(len(edges) != 2 for edges in incident):
        raise ValueError("Exposed boundary is not a connected closed outline")

    first = endpoints[0][0]
    current, used = first, set()
    vertices, codes = [nodes[first]], [Path.MOVETO]
    while len(used) < len(pieces):
        available = [j for j in incident[current] if j not in used]
        if not available:
            raise ValueError("Multiple boundary loops require separate handling")
        j = available[0]
        points, piece_codes = pieces[j]
        points = points.copy()
        start, end = endpoints[j]
        if current == end:
            # Reversing cubic controls reverses the exact same Bezier curve.
            points = points[::-1].copy()
            start, end = end, start
        points[0], points[-1] = nodes[start], nodes[end]
        vertices.extend(points[1:])
        codes.extend(piece_codes[1:])
        used.add(j)
        current = end
    if current != first:
        raise ValueError("Outline did not close")
    vertices.append(nodes[first])
    codes.append(Path.CLOSEPOLY)
    path = Path(np.array(vertices), np.array(codes, dtype=np.uint8))
    path.should_simplify = False
    return path
