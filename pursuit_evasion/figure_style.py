"""Typography, line styles and export geometry for the experiment figures."""

import shutil
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPatch
from matplotlib.patches import Circle
from matplotlib.transforms import Bbox

BLUE, ORANGE = "#3268A7", "#D77936"
FRAME, FILL, EDGE = "#1A1A1A", "#D9D9D9", "#4D4D4D"
LABELS = {
    "full": "Proposed",
    "proportional_navigation": "PN",
    "pure": "PP",
    "constant_bearing": "CB",
    "potential_field": "APF",
    "apollonius": "Apollonius",
}


def configure(engine):
    if engine == "latex" and not all(
        shutil.which(p) for p in ("latex", "dvipng", "kpsewhich")
    ):
        raise RuntimeError(
            "Put latex, dvipng and kpsewhich on PATH, or use --text-engine native"
        )
    plt.style.use("default")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": False,
            "axes.axisbelow": True,
            "path.simplify": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "mathtext.fontset": "stix",
            "legend.fontsize": 6.3,
            "legend.fancybox": False,
            "legend.framealpha": 1.0,
            "legend.edgecolor": FRAME,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 1.05,
            "legend.borderpad": 0.45,
            "lines.linewidth": 0.95,
            "text.usetex": engine == "latex",
            "text.latex.preamble": r"\usepackage[T1]{fontenc}\usepackage{amsmath}\usepackage{newtxtext}\usepackage{newtxmath}",
        }
    )


class CaptureCircleHandler(HandlerPatch):
    def __init__(self, diameter, **kwargs):
        super().__init__(**kwargs)
        self.diameter = diameter

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        circle = Circle(
            (width / 2 - xdescent, height / 2 - ydescent), self.diameter / 2
        )
        self.update_prop(circle, orig_handle, legend)
        circle.set_transform(trans)
        return [circle]


def axes_style(ax):
    ax.grid(False)
    ax.tick_params(direction="in", labelsize=7.5, length=2)
    for spine in ax.spines.values():
        spine.set(color=FRAME, linewidth=0.8)


def padded_bounds(fig):
    fig.canvas.draw()
    content = fig.get_tightbbox(fig.canvas.get_renderer())
    return Bbox.from_extents(
        content.x0 - 0.15, content.y0 - 0.01, content.x1 + 0.15, content.y1 + 0.03
    )


def save(fig, output, bounds=None):
    fig.savefig(output, bbox_inches=padded_bounds(fig) if bounds is None else bounds)
    plt.close(fig)
