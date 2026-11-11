"""Summaries from newly generated runs, including unresolved and failed runs."""

from collections import Counter
import csv
import json
from pathlib import Path
import statistics
from .config import METHODS


def moments(values):
    return (
        (statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0)
        if values
        else (None, None)
    )


def measurements(path):
    record = json.loads(path.read_text(encoding="utf-8"))
    times = [d["matrix_seconds"] for d in record["decisions"]]
    return dict(
        scene=record["scene"],
        seed=record["seed"],
        method=record["method"],
        status=record["status"],
        terminal_time=record["path"][-1]["time"],
        timing_count=len(times),
        timing_sum=sum(times),
        timing_max=max(times) if times else None,
        process_peak_memory_bytes=record.get("process_peak_memory_bytes"),
    )


def summarize(directory):
    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))
    records = [measurements(p) for p in sorted(directory.glob("runs/*/*.json"))]
    keyed = {(r["scene"], r["seed"], r["method"]): r for r in records}
    expected = {
        (s, protocol["first_seed"] + i, m)
        for s in protocol["scenes"]
        for i in range(protocol["trials"])
        for m in protocol["methods"]
    }
    if len(keyed) != len(records) or set(keyed) - expected:
        raise ValueError("Duplicate or unexpected run records")
    rows = []
    for scene in protocol["scenes"]:
        successes = [
            {
                seed
                for (s, seed, m), r in keyed.items()
                if s == scene and m == method and r["status"] == "capture"
            }
            for method in METHODS
        ]
        matched_available = all(
            (scene, protocol["first_seed"] + i, method) in keyed
            and keyed[(scene, protocol["first_seed"] + i, method)]["status"]
            != "computation_incomplete"
            for method in METHODS
            for i in range(protocol["trials"])
        )
        common = set.intersection(*successes) if matched_available else set()
        for method in protocol["methods"]:
            runs = [
                r for (s, seed, m), r in keyed.items() if s == scene and m == method
            ]
            counts = Counter(r["status"] for r in runs)
            captured = [r for r in runs if r["status"] == "capture"]
            escaped = [r for r in runs if r["status"] == "escape"]
            capture_mean, capture_sd = moments([r["terminal_time"] for r in captured])
            escape_mean, escape_sd = moments([r["terminal_time"] for r in escaped])
            matched_mean, matched_sd = moments(
                [
                    keyed[(scene, seed, method)]["terminal_time"]
                    for seed in sorted(common)
                ]
            )
            timing_count = sum(r["timing_count"] for r in runs)
            peak = [
                r["process_peak_memory_bytes"]
                for r in runs
                if r.get("process_peak_memory_bytes") is not None
            ]
            complete = (
                len(runs) == protocol["trials"] and not counts["computation_incomplete"]
            )
            rows.append(
                dict(
                    scene=scene,
                    method=method,
                    complete=complete,
                    planned=protocol["trials"],
                    recorded=len(runs),
                    capture=counts["capture"],
                    escape=counts["escape"],
                    cutoff=counts["observation_horizon_reached"],
                    incomplete=counts["computation_incomplete"],
                    capture_rate_percent=(
                        100 * len(captured) / protocol["trials"] if complete else None
                    ),
                    capture_time_mean_s=capture_mean,
                    capture_time_sd_s=capture_sd,
                    escape_time_mean_s=escape_mean,
                    escape_time_sd_s=escape_sd,
                    matched_comparison_available=matched_available,
                    common_captures=len(common) if matched_available else None,
                    matched_time_mean_s=matched_mean,
                    matched_time_sd_s=matched_sd,
                    game_time_mean_s=(
                        sum(r["timing_sum"] for r in runs) / timing_count
                        if timing_count
                        else None
                    ),
                    game_time_max_s=(
                        max(r["timing_max"] for r in runs if r["timing_count"])
                        if timing_count
                        else None
                    ),
                    process_peak_memory_bytes=max(peak) if peak else None,
                )
            )
    result = dict(
        expected_games=len(expected),
        recorded_games=len(records),
        missing_games=len(expected - set(keyed)),
        summaries=rows,
    )
    (directory / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (directory / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return result
