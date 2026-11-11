"""Five comparison pursuit laws, evaluated at the current observation."""

import numpy as np
from .geometry import project_ellipse_exterior


def unit(vector):
    length = np.linalg.norm(vector)
    return vector / length if length > 0 else np.zeros(2)


def initial_context(p, e, vp, ve, radius):
    nu = ve / vp
    return {
        "nu": nu,
        "circle_center": (e - nu * nu * p) / (1 - nu * nu),
        "circle_radius": nu * np.linalg.norm(e - p) / (1 - nu * nu) + radius,
        "apf_floor": radius / 1e6,
    }


def velocity_estimate(e, previous_e, h, ve):
    value = np.zeros(2) if previous_e is None else (e - previous_e) / h
    length = np.linalg.norm(value)
    return value * min(1.0, ve / length) if length else value


def constant_bearing_direction(p, e, estimate, vp):
    n = unit(e - p)
    transverse = estimate - np.dot(estimate, n) * n
    forward_squared = vp * vp - np.dot(transverse, transverse)
    if forward_squared < -1e-12:
        raise ValueError("invalid evader estimate for constant-bearing pursuit")
    return (transverse + np.sqrt(max(0.0, forward_squared)) * n) / vp


def ellipse_nearest(point, center, axes, axis):
    rotation = np.array([[axis[0], -axis[1]], [axis[1], axis[0]]])
    local = (point - center) @ rotation
    value = np.sum((local / axes) ** 2)
    if value < 1 - 1e-9:
        raise ValueError("APF query lies inside an ellipse")
    if value <= 1 + 1e-12:
        return 0.0, unit((local / (axes * axes)) @ rotation.T)
    nearest = project_ellipse_exterior(local, axes)
    delta = (local - nearest) @ rotation.T
    return float(np.linalg.norm(delta)), unit(delta)


def union_nearest(point, part):
    polygon, center, axes, axis = part
    area = np.sum(
        polygon[:, 0] * np.roll(polygon[:, 1], -1)
        - polygon[:, 1] * np.roll(polygon[:, 0], -1)
    )
    candidates = []
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = b - a
        parameter = np.clip(np.dot(point - a, edge) / np.dot(edge, edge), 0, 1)
        delta = point - a - parameter * edge
        distance = float(np.linalg.norm(delta))
        normal = (
            unit(delta)
            if distance > 1e-12
            else unit(np.sign(area) * np.array([edge[1], -edge[0]]))
        )
        candidates.append((distance, normal))
    candidates.append(ellipse_nearest(point, center, axes, axis))
    return min(candidates, key=lambda item: item[0])


def navigation_bearing(p, e, previous_los, heading, gain):
    los = float(np.arctan2(e[1] - p[1], e[0] - p[0]))
    if previous_los is None:
        heading = los
    else:
        change = los - previous_los
        heading += gain * np.arctan2(np.sin(change), np.cos(change))
    return los, heading


class Baseline:
    """History needed by PN/CB and the initial Apollonius circle."""

    def __init__(self, method, p, e, vp, ve, radius):
        self.method = method
        self.context = initial_context(p, e, vp, ve, radius)
        self.context.update(apf_influence=0.75, apf_gain=0.00625)
        self.previous_e = None
        self.previous_los = None
        self.heading = None

    def direction(self, geometry, p, e, vp, ve, h):
        context = self.context
        if self.method == "pure":
            result = unit(e - p)
        elif self.method == "proportional_navigation":
            self.previous_los, self.heading = navigation_bearing(
                p, e, self.previous_los, self.heading, 4.0
            )
            result = np.array([np.cos(self.heading), np.sin(self.heading)])
        elif self.method == "potential_field":
            force = (e - p).copy()
            for distance, normal in [union_nearest(p, part) for part in geometry.parts]:
                if distance < context["apf_influence"]:
                    d = max(distance, context["apf_floor"])
                    force += (
                        context["apf_gain"]
                        * (1 / d - 1 / context["apf_influence"])
                        / d**2
                        * normal
                    )
            result = unit(force)
        elif self.method == "constant_bearing":
            estimate = velocity_estimate(e, self.previous_e, h, ve)
            result = constant_bearing_direction(p, e, estimate, vp)
        elif self.method == "apollonius":
            nu = context["nu"]
            r = e - p
            center = (e - nu * nu * p) / (1 - nu * nu)
            radius = nu * np.linalg.norm(r) / (1 - nu * nu)
            y = center - context["circle_center"]
            z = (context["circle_radius"] - radius) * unit(r) + nu * y
            if np.linalg.norm(z) <= 1e-12:
                raise ValueError("Apollonius direction is numerically undefined")
            result = unit(z)
        else:
            raise ValueError(self.method)
        self.previous_e = e.copy()
        return result
