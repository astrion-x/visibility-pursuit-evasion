"""Execute both sampled controls and resolve events within each interval."""

import time
import numpy as np
from .baselines import Baseline
from .config import PARAMETERS, METHODS, Settings
from .events import advance_interval
from .game import action_alphabet, legal_actions, build_game, sample_control
from .geometry import number


def run(initial, method, geometry, *, parameters=None, settings=None):
    if method not in METHODS:
        raise ValueError(method)
    parameters = PARAMETERS if parameters is None else parameters
    settings = Settings() if settings is None else settings
    vp, ve, h, radius = [
        number(parameters[k])
        for k in ("pursuer_speed", "evader_speed", "sample_period", "capture_radius")
    ]
    steps = round(number(parameters["observation_seconds"]) / h)
    p, e = (np.array(initial[k], dtype=float) for k in ("P", "E"))
    if not (
        geometry.physical(p)
        and geometry.physical(e)
        and geometry.segment_physical(p, e)
        and np.linalg.norm(e - p) > radius
    ):
        raise ValueError(
            "The initial pair must be legal, visible and outside capture range"
        )
    pursuer_rng, evader_rng = (
        np.random.default_rng(s)
        for s in np.random.SeedSequence(initial["seed"]).spawn(2)
    )
    baseline = None if method == "full" else Baseline(method, p, e, vp, ve, radius)
    alphabet = action_alphabet(settings)
    record = dict(
        scene=initial["scene"],
        seed=initial["seed"],
        method=method,
        status="running",
        path=[dict(time=0.0, p=p.tolist(), e=e.tolist())],
        decisions=[],
    )
    started = time.perf_counter()
    try:
        for k in range(steps):
            game = build_game(geometry, p, e, parameters, settings)
            if baseline is None:
                pa, probability = game["pa"], game["sp"]
            else:
                direction = baseline.direction(geometry, p, e, vp, ve, h)
                _, pa = legal_actions(geometry, p, vp, h, alphabet)
                probability = np.zeros(len(pa))
                probability[int(np.argmax(pa @ direction))] = 1.0
            xp, xe = pursuer_rng.random(), evader_rng.random()
            ip, ie = sample_control(probability, xp), sample_control(game["se"], xe)
            u, w = pa[ip], game["ea"][ie]
            pn, en, event = advance_interval(geometry, p, e, u, w, parameters)
            record["decisions"].append(
                dict(
                    time=k * h,
                    p_action=u.tolist(),
                    e_action=w.tolist(),
                    equilibrium_gap=game["solved"]["equilibrium_gap"],
                    prediction_escape_probability=(
                        float(probability @ game["risk"]) if baseline is None else None
                    ),
                    matrix_seconds=game["matrix_seconds"],
                    lp_seconds=game["solved"]["lp_seconds"],
                )
            )
            duration = event["time"] if event else h
            record["path"].append(
                dict(time=k * h + duration, p=pn.tolist(), e=en.tolist())
            )
            if event:
                wt = event["witness_time"]
                record["status"] = event["kind"]
                record["event"] = dict(
                    event,
                    absolute_time=k * h + duration,
                    absolute_witness_time=k * h + wt,
                    witness_p=(p + vp * wt * u).tolist(),
                    witness_e=(e + ve * wt * w).tolist(),
                )
                break
            p, e = pn, en
        else:
            record["status"] = "observation_horizon_reached"
    except Exception as exc:
        record.update(
            status="computation_incomplete", error=f"{type(exc).__name__}: {exc}"
        )
    record["wall_seconds"] = time.perf_counter() - started
    record["pursuit_path_length_m"] = float(
        sum(
            np.linalg.norm(np.subtract(b["p"], a["p"]))
            for a, b in zip(record["path"][:-1], record["path"][1:])
        )
    )
    return record
