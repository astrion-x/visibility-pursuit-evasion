"""Render the manuscript experiment figures from reproduced run records."""

import os
from pathlib import Path
import tempfile


def plot_results(source, output, timing_inputs=None, text_engine="latex"):
    source, output = Path(source).resolve(), Path(output).resolve()
    with tempfile.TemporaryDirectory(prefix="pursuit_evasion_plot_") as cache:
        previous = os.environ.get("MPLCONFIGDIR")
        os.environ["MPLCONFIGDIR"] = cache
        try:
            from .figures import render

            render(source, output, timing_inputs, text_engine)
        finally:
            if previous is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous
