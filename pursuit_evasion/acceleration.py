"""Compiled geometry and forecast evaluation."""

import ctypes as ct
from pathlib import Path

import numpy as np

from .geometry import DistanceGeometry


class Point(ct.Structure):
    _fields_ = [("x", ct.c_double), ("y", ct.c_double)]


class Obstacle(ct.Structure):
    _fields_ = [
        ("n", ct.c_int),
        ("v", Point * 32),
        ("center", Point),
        ("axes", Point),
        ("unit", Point),
        ("low", Point),
        ("high", Point),
    ]


class CompiledGeometry(DistanceGeometry):
    def physical(self, points):
        points = np.asarray(points)
        if points.size < 1024:
            return super().physical(points)
        candidate = np.zeros(points.shape[:-1], dtype=bool)
        for poly, center, axes, u in self.parts:
            extent = np.hypot(axes[0] * u, axes[1] * u[::-1])
            lo = np.minimum(center - extent, poly.min(0)) - 1e-9
            hi = np.maximum(center + extent, poly.max(0)) + 1e-9
            candidate |= (
                (points[..., 0] >= lo[0])
                & (points[..., 0] <= hi[0])
                & (points[..., 1] >= lo[1])
                & (points[..., 1] <= hi[1])
            )
        free = np.ones(candidate.shape, dtype=bool)
        if candidate.any():
            free[candidate] = super().physical(points[candidate])
        return free

    def __init__(self, *args, kernel, **kwargs):
        super().__init__(*args, **kwargs)
        self.library = ct.CDLL(str(Path(kernel).resolve()))
        self.obstacles = (Obstacle * len(self.parts))()
        for obj, (poly, center, axes, u) in zip(self.obstacles, self.parts):
            if len(poly) > 32:
                raise ValueError("Unsupported polygon size for batch kernel")
            obj.n = len(poly)
            for i, v in enumerate(poly):
                obj.v[i] = Point(*v)
            extent = np.hypot(axes[0] * u, axes[1] * u[::-1])
            obj.center = Point(*center)
            obj.axes = Point(*axes)
            obj.unit = Point(*u)
            obj.low = Point(*(np.minimum(center - extent, poly.min(0)) - 1e-9))
            obj.high = Point(*(np.maximum(center + extent, poly.max(0)) + 1e-9))
        self.library.forecast.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            ct.c_int,
            ct.POINTER(Obstacle),
            ct.c_int,
            ct.c_void_p,
            ct.c_int,
            ct.c_int,
            Point,
            ct.c_double,
            ct.c_void_p,
            ct.c_int,
            ct.c_void_p,
            ct.c_void_p,
        ]
        self.library.forecast.restype = ct.c_int
        self.library.segments.argtypes = [
            ct.c_void_p,
            ct.c_void_p,
            ct.c_int,
            ct.POINTER(Obstacle),
            ct.c_int,
            ct.c_void_p,
        ]
        self.library.segments.restype = None
        self.library.clear_sweep.argtypes = [
            ct.c_void_p,
            ct.POINTER(Obstacle),
            ct.c_int,
        ]
        self.library.clear_sweep.restype = ct.c_int

    def segment_physical(self, start, end):
        a, b = np.broadcast_arrays(start, end)
        a = np.ascontiguousarray(a, dtype=np.float64)
        b = np.ascontiguousarray(b, dtype=np.float64)
        visible = np.empty(a.shape[:-1], dtype=bool)
        self.library.segments(
            a.ctypes.data,
            b.ctypes.data,
            visible.size,
            self.obstacles,
            len(self.parts),
            visible.ctypes.data,
        )
        return visible

    def certainly_continues(self, p, e, pn, en, radius):
        r = e - p
        dr = (en - pn) - r
        dd = dr @ dr
        t = np.clip(-(r @ dr) / dd, 0, 1) if dd > 0 else 0
        if np.linalg.norm(r + t * dr) <= radius + 1e-8:
            return False
        quad = np.array([p, e, pn, en], dtype=np.float64)
        return (
            self.library.clear_sweep(quad.ctypes.data, self.obstacles, len(self.parts))
            == 1
        )

    def forecast_fields(self, pursuer_positions, evader_positions, samples):
        pursuer_positions = np.ascontiguousarray(pursuer_positions, dtype=np.float64)
        evader_positions = np.ascontiguousarray(evader_positions, dtype=np.float64)
        fractions = np.linspace(0, 1, samples)
        shape = (
            len(pursuer_positions),
            pursuer_positions.shape[1],
            evader_positions.shape[1],
        )
        visible = np.empty(shape, dtype=np.bool_)
        clearance = np.empty(shape, dtype=np.float64)
        if self.library.forecast(
            pursuer_positions.ctypes.data,
            evader_positions.ctypes.data,
            *shape,
            self.obstacles,
            len(self.parts),
            self.sdf.ctypes.data,
            *self.sdf.shape,
            Point(*self.origin),
            self.spacing,
            fractions.ctypes.data,
            samples,
            visible.ctypes.data,
            clearance.ctypes.data
        ):
            raise ValueError("Forecast outside the padded score grid")
        return visible, clearance
