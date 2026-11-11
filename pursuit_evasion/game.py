"""Finite zero-sum forecast game and independent mixed-control sampling."""

import time
import numpy as np
import nashpy
from .geometry import number


def action_alphabet(settings):
    angles = 2 * np.pi * np.arange(settings.directions) / settings.directions
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    return np.vstack((np.zeros((1, 2)), *(r * directions for r in settings.radii)))


def legal_actions(geometry, point, speed, duration, alphabet):
    valid = geometry.segment_physical(point, point + speed * duration * alphabet)
    indices = np.flatnonzero(valid)
    if not len(indices) or 0 not in indices:
        raise ValueError("physical state must admit the zero command")
    return indices, alphabet[indices]


def payoff_from_predictions(distances, clearances, visible, times, radius, scale):
    """Finite forecast cost with distance scale D."""
    horizon = times[-1]
    capture = (distances <= radius) & visible
    escape = (distances > radius) & ~visible
    terminal = capture | escape
    has_terminal = np.any(terminal, axis=0)
    first = np.argmax(terminal, axis=0)
    first_capture = np.take_along_axis(capture, first[None, ...], axis=0)[0]
    stop_cost = np.where(
        first_capture,
        -1 + times[first] / (4 * horizon),
        1 - times[first] / (4 * horizon),
    )
    difference = distances[-1] - np.min(clearances, axis=0)
    continuation = difference / (2 * (scale + np.abs(difference)))
    return np.where(has_terminal, stop_cost, continuation)


def equilibrium(cost, tolerance=1e-7):
    """Nashpy maximizes row payoffs; -cost makes P the minimizing player."""
    started = time.perf_counter()
    p, e = nashpy.Game(-np.asarray(cost)).linear_program()
    probabilities = []
    for strategy in (p, e):
        if not np.all(np.isfinite(strategy)) or np.min(strategy) < -tolerance:
            raise ValueError("invalid numerical LP probabilities")
        strategy = np.maximum(strategy, 0)
        if strategy.sum() <= 0:
            raise ValueError("empty LP strategy")
        probabilities.append(strategy / strategy.sum())
    p, e = probabilities
    upper = float(np.max(p @ cost))
    lower = float(np.min(cost @ e))
    gap = max(0.0, upper - lower)
    if gap > tolerance:
        raise ValueError(f"finite-game exploitability gap {gap} exceeds {tolerance}")
    return (
        p,
        e,
        {
            "lp_seconds": time.perf_counter() - started,
            "lower_value": lower,
            "upper_value": upper,
            "equilibrium_gap": gap,
            "expected_cost": float(p @ cost @ e),
            "p_support": int(np.sum(p > 1e-9)),
            "e_support": int(np.sum(e > 1e-9)),
        },
    )


def build_game(geometry, p, e, parameters, settings):
    start = time.perf_counter()
    vp, ve, h, radius, scale = [
        number(parameters[k])
        for k in (
            "pursuer_speed",
            "evader_speed",
            "sample_period",
            "capture_radius",
            "payoff_scale_m",
        )
    ]
    times = h * np.arange(settings.prediction_steps + 1)
    alphabet = action_alphabet(settings)
    pi, pa = legal_actions(geometry, p, vp, times[-1], alphabet)
    ei, ea = legal_actions(geometry, e, ve, times[-1], alphabet)
    pp = p + vp * times[:, None, None] * pa[None, :, :]
    ee = e + ve * times[:, None, None] * ea[None, :, :]
    diff = ee[:, :, None, :].transpose(0, 2, 1, 3) - pp[:, :, None, :]
    dist = np.linalg.norm(diff, axis=-1)
    forecast_fields = getattr(geometry, "forecast_fields", None)
    if forecast_fields is None:
        vis = geometry.segment_physical(pp[:, :, None, :], ee[:, None, :, :])
    else:
        vis, clear = forecast_fields(pp, ee, settings.sight_samples)
    capture, escape = (dist <= radius) & vis, (dist > radius) & ~vis
    terminal = capture | escape
    first = np.argmax(terminal, axis=0)
    has = terminal.any(0)
    first_escape = has & np.take_along_axis(escape, first[None], axis=0)[0]
    # Clearance is used only by nonterminal entries; terminal costs ignore it.
    if forecast_fields is None:
        clear = np.zeros_like(dist)
        ii, jj = np.where(~has)
        if len(ii):
            fractions = np.linspace(0, 1, settings.sight_samples)
            lines = (
                pp[:, ii, None, :]
                + fractions[None, None, :, None] * diff[:, ii, jj, None, :]
            )
            raw = geometry.distance(lines).min(-1)
            clear[:, ii, jj] = np.where(
                vis[:, ii, jj], np.maximum(raw, 0), np.minimum(raw, 0)
            )
    cost = payoff_from_predictions(dist, clear, vis, times, radius, scale)
    sp, se, solved = equilibrium(cost, settings.equilibrium_gap_tolerance)
    return dict(
        cost=cost,
        pi=pi,
        ei=ei,
        pa=pa,
        ea=ea,
        sp=sp,
        se=se,
        solved=solved,
        risk=first_escape @ se,
        matrix_seconds=time.perf_counter() - start,
    )


def sample_control(probabilities, random_draw):
    index = int(np.searchsorted(np.cumsum(probabilities), random_draw, side="right"))
    if index < len(probabilities):
        return index
    # Resolve a floating-point deficit at the upper edge within the support.
    return int(np.flatnonzero(np.asarray(probabilities) > 0)[-1])
