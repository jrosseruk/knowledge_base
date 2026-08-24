"""Render the lens grid from `lens_grid.json` (no GPU needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jlens_plot as jp

HERE = Path(__file__).resolve().parent


def main() -> None:
    data = json.loads((HERE / "lens_grid.json").read_text())
    records = data["records"]
    jp.use_style()

    n = len(records)
    figure, axes = plt.subplots(2, n, figsize=(3.3 * n, 7.6), sharex=True)
    for column, record in enumerate(records):
        alphas = np.array(record["alphas"])
        actual = np.array(record["actual"]) - record["actual"][0]
        chord = record["chord"]
        span = max(abs(actual).max(), abs(chord)) * 1.25
        for row, corpus in enumerate(("domain", "web")):
            axis = axes[row][column]
            axis.axhline(0, color=jp.MUTED, lw=0.8)
            axis.plot(alphas, chord * alphas, "-.", lw=1.7, color=jp.MUTED,
                      label="chord over [0,1]")
            axis.plot(alphas, record["tangent"] * alphas, ":", lw=1.7, color=jp.INK,
                      label="tangent at $\\alpha$=0")
            axis.plot(alphas, record["slopes"]["logit"] * alphas, "--", lw=1.5,
                      color=jp.LENS_COLORS["logit"], alpha=0.85, label="logit lens")
            axis.plot(alphas, record["slopes"][f"j_{corpus}"] * alphas, "--", lw=2.2,
                      color=jp.LENS_COLORS["j"], label="J-lens")
            axis.plot(alphas, record["slopes"][f"tuned_{corpus}"] * alphas, "--", lw=2.2,
                      color=jp.LENS_COLORS["tuned"], label="tuned lens")
            axis.plot(alphas, actual, color=jp.INK, lw=2.6, zorder=5, label="model")
            axis.set_ylim(-0.28 * span, span)
            axis.grid(axis="y", alpha=0.7)

            fraction = record["slopes"][f"tuned_{corpus}"] / chord
            axis.text(0.03, 0.965,
                      f"tuned / chord = {fraction:.2f}",
                      transform=axis.transAxes, fontsize=8.5, va="top",
                      color=jp.LENS_COLORS["tuned"])
            if row == 0:
                axis.set_title(record["name"].replace("  (no gate)", "\n(no gate)"),
                               fontsize=9.5)
            if row == 1:
                axis.set_xlabel("interpolation $\\alpha$")
            if column == 0:
                axis.set_ylabel(
                    ("fitted on the relation's\nown corpus" if corpus == "domain"
                     else "fitted on web text\n(as deployed)")
                    + "\n\nchange in readout", fontsize=9)

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [labels.index(x) for x in
             ("model", "tangent at $\\alpha$=0", "chord over [0,1]",
              "J-lens", "tuned lens", "logit lens")]
    figure.legend([handles[i] for i in order], [labels[i] for i in order],
                  frameon=False, fontsize=9, ncol=6,
                  loc="lower center", bbox_to_anchor=(0.5, -0.035))
    figure.suptitle(
        "Fitting the lenses on the relation's own corpus moves the regression toward the chord — "
        "but only partly, and not on every item",
        fontsize=12, y=0.995,
    )
    figure.savefig(HERE / "fig_lens_grid.png", bbox_inches="tight")
    print("wrote fig_lens_grid.png")


if __name__ == "__main__":
    main()
