"""Sequential experiment execution and result summaries."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from .config import FIRST_SEED, METHODS, PARAMETERS, SCENE_CODES, Settings, load_scenes
from .execution import execute_case, write_json


def external_output(path):
    path = Path(path).resolve()
    release = Path(__file__).resolve().parent.parent
    if path == release or release in path.parents:
        raise ValueError("Write generated files outside the source release directory")
    return path


def protocol(args):
    scenes = load_scenes()["cases"]
    return dict(
        format_version=6,
        implementation="compiled_forecast",
        continuation_payoff="difference/(2*(D+abs(difference)))",
        scenes=args.scenes,
        trials=args.trials,
        first_seed=args.first_seed,
        methods=args.methods,
        parameters=PARAMETERS,
        settings=asdict(Settings()),
        scene_stream_codes=SCENE_CODES,
        obstacles={scene: scenes[scene] for scene in args.scenes},
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("run")
    p.add_argument(
        "--scenes", nargs="+", choices=tuple(SCENE_CODES), default=list(SCENE_CODES)
    )
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--first-seed", type=int, default=FIRST_SEED)
    p.add_argument("--output", required=True)
    p.add_argument("--kernel", required=True, type=Path)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p = commands.add_parser("summarize")
    p.add_argument("--input", required=True)
    p = commands.add_parser("build")
    p.add_argument("--output", required=True)
    p.add_argument("--compiler")
    p = commands.add_parser("plot")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--timing-inputs", nargs=4, metavar=("A", "B", "C", "D"))
    p.add_argument("--text-engine", choices=("latex", "native"), default="latex")
    args = parser.parse_args(argv)
    if args.command == "build":
        from .build import build_kernel

        print(build_kernel(external_output(args.output), args.compiler))
        return
    if args.command == "plot":
        from .plots import plot_results

        plot_results(
            external_output(args.input),
            external_output(args.output),
            args.timing_inputs,
            args.text_engine,
        )
        return
    from .results import summarize

    if args.command == "summarize":
        result = summarize(external_output(args.input))
        print(
            f"Summarized {result['recorded_games']} / {result['expected_games']} games"
        )
        return
    if (
        args.trials < 1
        or args.first_seed < 0
        or len(set(args.scenes)) != len(args.scenes)
    ):
        parser.error("Use positive trials, a nonnegative seed and distinct scene names")
    if not args.kernel.is_file():
        parser.error(
            "Build the geometry kernel first and pass its library file with --kernel"
        )
    output = external_output(args.output)
    if len(set(args.methods)) != len(args.methods):
        parser.error("Use distinct methods")
    metadata = protocol(args)
    file = output / "protocol.json"
    # Compare the serialized representation so tuple-valued settings round-trip.
    if file.exists():
        if json.loads(file.read_text(encoding="utf-8")) != json.loads(
            json.dumps(metadata)
        ):
            raise ValueError("Output protocol differs; choose a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise ValueError("Output directory is nonempty and has no matching protocol")
    else:
        write_json(file, metadata)
    failures = []
    for i in range(args.trials):
        seed = args.first_seed + i
        for scene in args.scenes:
            try:
                statuses = execute_case(scene, seed, args.methods, output, args.kernel)
                print(scene, seed, statuses, flush=True)
                if "computation_incomplete" in statuses.values():
                    failures.append((scene, seed))
            except Exception as exc:
                failures.append((scene, seed))
                print(f"{scene} {seed} failed: {exc}", file=sys.stderr, flush=True)
    result = summarize(output)
    print(f"Saved {result['recorded_games']} / {result['expected_games']} games")
    if failures or result["missing_games"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
