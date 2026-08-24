"""Shared plotting style for the J-lens experiments.

Three lenses means three categorical slots; the model's own response is not a
lens, so it is drawn in ink rather than given a hue.  Palette slots 1-3 of the
project's validated categorical theme are used in fixed order and never cycled.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Categorical slots 1-3, validated all-pairs in both light and dark modes.
LENS_COLORS = {
    "j": "#eb6834",       # slot 2, orange
    "tuned": "#2a78d6",   # slot 1, blue
    "logit": "#1baf7a",   # slot 3, green
}
LENS_LABELS = {"j": "J-lens", "tuned": "Tuned lens", "logit": "Logit lens"}
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#dcdcd8"


def use_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": MUTED,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "font.size": 10,
            "figure.dpi": 130,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def band(ax, start: int, stop: int, label: str | None = None) -> None:
    """Shade a layer range, e.g. the workspace band."""
    ax.axvspan(start, stop, color="#2a78d6", alpha=0.06, lw=0)
    if label:
        ax.text(
            (start + stop) / 2,
            ax.get_ylim()[1],
            label,
            ha="center",
            va="top",
            fontsize=8,
            color=MUTED,
        )
