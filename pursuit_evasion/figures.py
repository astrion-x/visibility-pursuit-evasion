"""Trajectory, capture and computation-time figures for the four scenes."""

import json
from pathlib import Path
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, PathPatch, Rectangle
from matplotlib.transforms import Affine2D, Bbox
from .config import METHODS, load_scenes
from .geometry import number
from .plot_geometry import composite, closed_outline, obstacle_bounds_center
from .figure_style import (
    plt,
    configure,
    BLUE,
    ORANGE,
    FRAME,
    FILL,
    EDGE,
    LABELS,
    CaptureCircleHandler,
    axes_style,
    save,
)
from .results import summarize


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def scene_geometry(scene):
    obstacles = [composite(ob) for ob in load_scenes()["cases"][scene]["obstacles"]]
    origin = obstacle_bounds_center(obstacles)
    transform = Affine2D().translate(*(-origin))
    return [transform.transform_path(closed_outline(ob)) for ob in obstacles], origin


def load_run(source, scene, seed, method):
    path = source / "runs" / scene / f"{seed}_{method}.json"
    record = read(path)
    if (record["scene"], record["seed"], record["method"]) != (scene, seed, method):
        raise ValueError(f"Run identity mismatch: {path}")
    if record["status"] not in ("capture", "escape"):
        raise ValueError(f"The fixed illustrative run has no terminal event: {path}")
    return record


def radius_patch(center, radius):
    return Circle(
        center,
        radius,
        facecolor="#F0EAF8",
        edgecolor="#9280B8",
        lw=0.65,
        ls=(0, (0.8, 0.6)),
        zorder=3,
    )


def trajectory_panel(ax, run, outlines, origin, radius, marker):
    for path in outlines:
        ax.add_patch(
            PathPatch(
                path,
                facecolor=FILL,
                edgecolor=EDGE,
                lw=0.55,
                joinstyle="miter",
                zorder=2,
            )
        )
    for key, color, line in [("p", BLUE, "-"), ("e", ORANGE, (0, (1.2, 0.8)))]:
        xy = np.array([s[key] for s in run["path"]]) - origin
        ax.plot(*xy.T, ls=line, color=color, lw=0.82, zorder=4)
        ax.plot(*xy[0], "o", mfc="white", mec=color, ms=marker, mew=0.6, zorder=6)
    ax.add_patch(radius_patch(np.asarray(run["path"][-1]["p"]) - origin, radius))
    terminal = (
        run["path"][-1]["e"]
        if run["status"] == "capture"
        else run["event"]["witness_e"]
    )
    color = "#D62728" if run["status"] == "capture" else "#298447"
    ax.plot(
        *(np.asarray(terminal) - origin), "x", color=color, ms=marker, mew=0.6, zorder=7
    )
    axes_style(ax)


def legend(fig, marker, anchor):
    radius = radius_patch((0, 0), 0.5)
    radius.set_label("Capture radius")
    handles = [
        Line2D([], [], color=BLUE, lw=0.82, label="Pursuer"),
        Line2D([], [], color=ORANGE, lw=0.82, ls=(0, (1.2, 0.8)), label="Evader"),
        Line2D(
            [],
            [],
            ls="none",
            marker="o",
            mfc="white",
            mec=FRAME,
            ms=marker,
            mew=0.6,
            label="Start",
        ),
        radius,
        Line2D(
            [],
            [],
            ls="none",
            marker="x",
            color="#D62728",
            ms=marker,
            mew=0.6,
            label="Capture",
        ),
        Line2D(
            [],
            [],
            ls="none",
            marker="x",
            color="#298447",
            ms=marker,
            mew=0.6,
            label="Escape",
        ),
    ]
    item = fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=anchor,
        ncol=6,
        fontsize=6.3,
        handlelength=1.4,
        handletextpad=0.35,
        columnspacing=0.30,
        borderpad=0.45,
        handler_map={radius: CaptureCircleHandler(marker)},
    )
    item.get_frame().set_linewidth(0.4)
    return item


def formula(run):
    kind = "c" if run["status"] == "capture" else "e"
    return (
        rf'$t_{{\mathrm{{{kind}}}}}={run["event"]["absolute_time"]:.2f}\,\mathrm{{s}}$'
    )


def trajectory_figure(source, output, scene, seed, radius):
    outlines, origin = scene_geometry(scene)
    fig, axes = plt.subplots(2, 3, figsize=(3.6, 3.2), sharex=True, sharey=True)
    side, gap, left, bottom = 0.99, 0.09, 0.42, 0.34
    marker = 2 * radius * side * 72 / 24
    for i, (ax, method) in enumerate(zip(axes.flat, METHODS)):
        run = load_run(source, scene, seed, method)
        row, col = divmod(i, 3)
        ax.set_position(
            [
                (left + col * (side + gap)) / 3.6,
                (bottom + (1 - row) * (side + gap)) / 3.2,
                side / 3.6,
                side / 3.2,
            ]
        )
        trajectory_panel(ax, run, outlines, origin, radius, marker)
        ax.set(
            xlim=(-12, 12),
            ylim=(-12, 12),
            aspect="equal",
            xticks=[-10, 0, 10],
            yticks=[-10, 0, 10],
        )
        ax.tick_params(length=1.6, labelbottom=row == 1, labelleft=col == 0)
        if row == 1:
            ax.set_xlabel("$X$ (m)", fontsize=8, labelpad=1)
        if col == 0:
            ax.set_ylabel("$Y$ (m)", fontsize=8, labelpad=1)
        ax.text(
            0.96,
            0.94,
            LABELS[method],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=6.5,
            color=FRAME,
        )
        ax.text(
            0.96,
            0.04,
            formula(run),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6,
            color=FRAME,
        )
    legend(
        fig,
        marker,
        (
            (left + (3 * side + 2 * gap) / 2) / 3.6,
            (bottom + 2 * side + gap + 0.055) / 3.2,
        ),
    )
    save(fig, output)


def visibility_figure(source, output, scene, seed, spec, radius):
    outlines, origin = scene_geometry(scene)
    fig, axes = plt.subplots(1, 2, figsize=(3.6, 2.45))
    side, gap, left, bottom = 1.48, 0.10, 0.39, 0.34
    for j, (ax, method, pos) in enumerate(
        zip(axes, ("full", spec["comparator"]), spec["labels"])
    ):
        run = load_run(source, scene, seed, method)
        ax.set_position(
            [(left + j * (side + gap)) / 3.6, bottom / 2.45, side / 3.6, side / 2.45]
        )
        trajectory_panel(ax, run, outlines, origin, radius, 3.2)
        xlim, ylim = spec["xlim"], spec["ylim"]
        ax.set(
            xlim=xlim,
            ylim=ylim,
            aspect="equal",
            xticks=np.arange((np.floor(xlim[0] / 2) + 1) * 2, xlim[1], 2),
            yticks=np.arange(np.ceil(ylim[0] / 2) * 2, ylim[1] + 0.01, 2),
        )
        ax.tick_params(length=1.6, labelleft=j == 0)
        ax.set_xlabel("$X$ (m)", fontsize=8, labelpad=1)
        if j == 0:
            ax.set_ylabel("$Y$ (m)", fontsize=8, labelpad=1)
        ax.text(
            0.96,
            0.94,
            LABELS[method],
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color=FRAME,
        )
        ax.annotate(
            formula(run),
            xy=np.asarray(run["event"]["witness_e"]) - origin,
            xycoords="data",
            xytext=pos,
            textcoords="axes fraction",
            ha="center",
            va="center",
            fontsize=6.5,
            color=FRAME,
            zorder=12,
            bbox=dict(boxstyle="square,pad=0.05", facecolor="none", edgecolor="none"),
            arrowprops=dict(
                arrowstyle="->",
                color=EDGE,
                lw=0.55,
                mutation_scale=6,
                shrinkA=0.5,
                shrinkB=3,
                connectionstyle="arc3,rad=0",
            ),
        )
    legend(
        fig, 3.2, ((left + (2 * side + gap) / 2) / 3.6, (bottom + side + 0.055) / 2.45)
    )
    save(
        fig,
        output,
        Bbox.from_bounds(
            -0.0433381928364336,
            0.07608274559864486,
            3.643338192836434,
            2.0315400560884957,
        ),
    )


def counts_figure(rows, output):
    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    x = np.arange(6)
    bottom = np.zeros(6)
    width = 0.55
    for scene, color in zip("ABCD", ("#469EB4", "#87CFA4", "#CBE99D", "#F5FBB1")):
        values = [rows[scene, m]["capture"] for m in METHODS]
        ax.bar(
            x,
            values,
            width=width,
            bottom=bottom,
            color=color,
            lw=0,
            label=f"Scenario {scene}",
            zorder=3,
        )
        bottom += values
    for position, total in zip(x, bottom):
        ax.add_patch(
            Rectangle(
                (position - width / 2, 0),
                width,
                total,
                fill=False,
                edgecolor=FRAME,
                lw=0.45,
                zorder=4,
            )
        )
        ax.annotate(
            str(int(total)),
            (position, total),
            xytext=(0, 1.5),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
        )
    ax.set_xticks(x, [LABELS[m] for m in METHODS])
    ax.set(
        ylabel="Number of captures",
        ylim=(0, 400),
        yticks=[0, 100, 200, 300, 400],
        xlim=(-0.6, 5.7),
    )
    ax.yaxis.labelpad = 1
    axes_style(ax)
    ax.tick_params(axis="x", length=0, labelsize=6.7)
    item = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=6.3,
        handlelength=1.25,
        handletextpad=0.3,
        columnspacing=0.7,
    )
    item.get_frame().set_linewidth(0.4)
    fig.tight_layout(pad=0.5)
    save(fig, output)


def rates_figure(rows, output):
    fig, axes = plt.subplots(2, 2, figsize=(3.6, 3.2), sharex=True, sharey=True)
    for ax, scene in zip(axes.flat, "ABCD"):
        rates = [rows[scene, m]["capture_rate_percent"] for m in METHODS]
        ax.barh(
            np.arange(6),
            rates,
            height=0.57,
            color=[BLUE] + ["#B8C7D6"] * 5,
            edgecolor=EDGE,
            lw=0.35,
            zorder=3,
        )
        for y, rate in enumerate(rates):
            ax.annotate(
                f"{rate:.0f}",
                (rate, y),
                xytext=(2, 0),
                textcoords="offset points",
                va="center",
                fontsize=7.5,
            )
        ax.set_yticks(np.arange(6), [LABELS[m] for m in METHODS])
        ax.set(ylim=(5.65, -1.2), xlim=(0, 100), xticks=[0, 50, 100])
        axes_style(ax)
        ax.tick_params(labelbottom=True)
        ax.tick_params(axis="y", length=0)
        ax.text(
            0.96,
            0.975,
            f"Scenario {scene}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
        )
    for ax in axes[1]:
        ax.set_xlabel(
            r"Capture rate (\%)" if plt.rcParams["text.usetex"] else "Capture rate (%)",
            fontsize=8,
            labelpad=1,
        )
    fig.tight_layout(pad=0.5, h_pad=0.8, w_pad=0.8)
    save(fig, output)


def timing_figure(rows, output):
    fig, ax = plt.subplots(figsize=(3.6, 2.45))
    x = np.arange(4)
    width = 0.32
    maximum = max(rows[s]["game_time_max_s"] for s in "ABCD") * 1000
    for offset, key, label, color, hatch in [
        (-0.5, "game_time_mean_s", "Mean", BLUE, None),
        (0.5, "game_time_max_s", "Observed maximum", "#96D1E0", "///"),
    ]:
        values = [rows[s][key] * 1000 for s in "ABCD"]
        ax.bar(
            x + offset * width,
            values,
            width,
            color=color,
            edgecolor=EDGE,
            lw=0.6,
            hatch=hatch,
            label=label,
        )
        for position, value in zip(x + offset * width, values):
            ax.annotate(
                f"{value:.2f}",
                (position, value),
                xytext=(0, 1.5),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
            )
    upper = max(120, 20 * np.ceil(1.08 * maximum / 20))
    ax.set(
        ylabel="Game computation time (ms)",
        ylim=(0, upper),
        yticks=np.arange(0, upper + 1, 20),
    )
    ax.yaxis.labelpad = 1
    ax.set_xticks(x, [f"Scenario {s}" for s in "ABCD"])
    axes_style(ax)
    ax.tick_params(axis="x", length=0)
    item = ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=2, fontsize=6.3
    )
    item.get_frame().set_linewidth(0.4)
    fig.tight_layout(pad=0.5)
    save(fig, output)


def render(source, output, timing_inputs, engine):
    protocol = read(source / "protocol.json")
    selection = read(Path(__file__).with_name("figure_selection.json"))
    if (
        set(protocol["scenes"]) != set("ABCD")
        or set(protocol["methods"]) != set(METHODS)
        or protocol["trials"] != 100
        or protocol["first_seed"] != selection["first_seed"]
    ):
        raise ValueError(
            "The manuscript figures require all four scenes, six methods and the specified 100 trials"
        )
    if protocol["obstacles"] != load_scenes()["cases"]:
        raise ValueError("The run geometry differs from the packaged scenes")
    summary = summarize(source)
    if summary["missing_games"] or not all(r["complete"] for r in summary["summaries"]):
        raise ValueError(
            "Complete all requested games before drawing aggregate figures"
        )
    rows = {(r["scene"], r["method"]): r for r in summary["summaries"]}
    timing = {s: rows[s, "full"] for s in "ABCD"}
    if timing_inputs:
        for scene, folder in zip("ABCD", timing_inputs):
            folder = Path(folder)
            p = read(folder / "protocol.json")
            if (
                any(
                    p[key] != protocol[key]
                    for key in (
                        "parameters",
                        "settings",
                        "first_seed",
                        "trials",
                        "implementation",
                    )
                )
                or p["obstacles"].get(scene) != protocol["obstacles"][scene]
            ):
                raise ValueError(f"Timing configuration mismatch for {scene}")
            result = summarize(folder)
            row = next(
                r
                for r in result["summaries"]
                if r["scene"] == scene and r["method"] == "full"
            )
            if not row["complete"]:
                raise ValueError(f"Incomplete timing runs for {scene}")
            timing[scene] = row
    configure(engine)
    output.mkdir(parents=True, exist_ok=True)
    radius = number(protocol["parameters"]["capture_radius"])
    for i, scene in enumerate("ABCD", 3):
        seed = selection["first_seed"] + selection["trajectory_trials"][scene] - 1
        trajectory_figure(source, output / f"Fig.{i}.pdf", scene, seed, radius)
    counts_figure(rows, output / "Fig.7.pdf")
    rates_figure(rows, output / "Fig.8.pdf")
    for i, (scene, spec) in enumerate(selection["visibility"].items(), 9):
        visibility_figure(
            source,
            output / f"Fig.{i}.pdf",
            scene,
            selection["first_seed"] + spec["trial"] - 1,
            spec,
            radius,
        )
    timing_figure(timing, output / "Fig.12.pdf")
    print(f"Saved ten experiment figures to {output}")
