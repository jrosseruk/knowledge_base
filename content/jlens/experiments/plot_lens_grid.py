"""Render the lens grid from `lens_grid.json` (no GPU needed).

Four rows per column, paired: response curve then rank bars, for the
domain-fitted lenses and then for the web-fitted ones.

The response panels are normalised by each item's own chord. That does several
things at once:

* every model curve then ends at exactly 1.0, so the panels share a y axis
  starting at 0 and differences in *shape* -- sigmoid versus straight -- become
  the only visual signal;
* the chord becomes the same unit diagonal everywhere, a fixed reference;
* each lens slope reads directly as a fraction of the chord, so the lines are
  comparable across items and between the two fits.

Raw units cannot do this: the chord is ~50,000 on the counting items and ~4,800
on the colour item, so a shared raw axis would flatten one or clip the other.

The bar panels give the Claim 1 readout alongside the Claim 2 geometry: what
each lens actually says about the bridge entity and the answer at the
unmodified state, as full-vocabulary ranks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import textwrap

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jlens_plot as jp

HERE = Path(__file__).resolve().parent
TOP = 1.62
LENSES = ("j", "tuned", "logit")
ROWS = (("two_hop", "corpus: two-hop\n(entity described)"),
        ("one_hop", "corpus: one-hop\n(entity named)"),
        ("mixed", "corpus: mixed\n(both framings)"),
        ("web", "corpus: web text\n(as deployed)"))


def prompt_header(record) -> str:
    """Both prompts written out, one per line, with the answer each produces."""
    a = record["prompt_a"].strip().removeprefix("Fact:").strip()
    b = record["prompt_b"].strip().removeprefix("Fact:").strip()
    # show B only from where it starts to differ, so the change is obvious
    shared = 0
    while shared < min(len(a), len(b)) and a[shared] == b[shared]:
        shared += 1
    shared = a.rfind(" ", 0, shared + 1) + 1
    tail = b[shared:]
    first = textwrap.fill(f'\u03b1=0   "{a}"', 34, subsequent_indent="         ")
    second = textwrap.fill(f'\u03b1=1   "\u2026{tail}"', 34, subsequent_indent="         ")
    return (f"{first}\n           \u2192  {record['answer']}\n"
            f"{second}\n           \u2192  {record['counter_answer']}")


def main() -> None:
    data = json.loads((HERE / "lens_grid.json").read_text())
    records = data["records"]
    jp.use_style()

    n = len(records)
    figure, axes = plt.subplots(
        2 * len(ROWS), n, figsize=(3.35 * n, 5.9 * len(ROWS)),
        gridspec_kw={"height_ratios": [2.5, 1.25] * len(ROWS), "hspace": 0.45},
    )

    for column, record in enumerate(records):
        alphas = np.array(record["alphas"])
        chord = record["chord"]
        actual = (np.array(record["actual"]) - record["actual"][0]) / chord

        for block, (corpus, row_label) in enumerate(ROWS):
            curve_axis = axes[block * 2][column]
            bar_axis = axes[block * 2 + 1][column]

            curve_axis.axhline(1.0, color=jp.GRID, lw=1.0)
            curve_axis.plot(alphas, alphas, "-.", lw=1.7, color=jp.MUTED, label="chord over [0,1]")
            curve_axis.plot(alphas, record["tangent"] / chord * alphas, ":", lw=1.7,
                            color=jp.INK, label="tangent at $\\alpha$=0")
            curve_axis.plot(alphas, record["slopes"]["logit"] / chord * alphas, "--", lw=1.6,
                            color=jp.LENS_COLORS["logit"], label="logit lens")
            curve_axis.plot(alphas, record["slopes"][f"tuned_{corpus}"] / chord * alphas, "--",
                            lw=2.2, color=jp.LENS_COLORS["tuned"], label="tuned lens")
            curve_axis.plot(alphas, record["slopes"][f"j_{corpus}"] / chord * alphas, "--",
                            lw=2.2, color=jp.LENS_COLORS["j"], label="J-lens")
            curve_axis.plot(alphas, actual, color=jp.INK, lw=2.6, zorder=5, label="model")
            curve_axis.set_ylim(0, TOP)
            curve_axis.set_xlim(0, 1)
            curve_axis.grid(axis="y", alpha=0.55)
            curve_axis.set_xlabel("interpolation $\\alpha$", fontsize=8.5, labelpad=1)
            curve_axis.tick_params(labelsize=8)
            if column:
                curve_axis.set_yticklabels([])
            curve_axis.text(
                0.04, 0.955,
                f"tuned = {record['slopes'][f'tuned_{corpus}'] / chord:.2f} $\\times$ chord",
                transform=curve_axis.transAxes, fontsize=8.5, va="top",
                color=jp.LENS_COLORS["tuned"],
            )

            ranks = record["ranks"][corpus]
            index = np.arange(len(LENSES))
            for offset, (key, alpha_value, hatch) in enumerate(
                (("bridge", 1.0, None), ("answer", 0.4, "///"))
            ):
                values = [ranks[lens][key] for lens in LENSES]
                bars = bar_axis.bar(
                    index + (offset - 0.5) * 0.38, values, 0.34,
                    color=[jp.LENS_COLORS[lens] for lens in LENSES],
                    alpha=alpha_value, hatch=hatch, edgecolor="white", linewidth=0.7,
                )
                bar_axis.bar_label(bars, fmt="%d", fontsize=7.5, color=jp.MUTED, padding=1)
            bar_axis.set_yscale("log")
            bar_axis.set_ylim(1, max(
                60, max(ranks[l][k] for l in LENSES for k in ("bridge", "answer")) * 4))
            bar_axis.set_xticks(index, ["J", "tuned", "logit"], fontsize=8)
            bar_axis.grid(axis="y", alpha=0.5)
            bar_axis.tick_params(labelsize=7.5)
            if column == 0:
                bar_axis.set_ylabel("rank\n(lower better)", fontsize=8)
            else:
                bar_axis.set_yticklabels([])

            if column == 0:
                curve_axis.set_ylabel(
                    "how far the answer has switched\n"
                    "(1 = the model's full switch)", fontsize=8.5)
            if corpus == record["probe_framing"]:
                for spine in curve_axis.spines.values():
                    spine.set_edgecolor(jp.LENS_COLORS["tuned"])
                    spine.set_linewidth(1.8)
        named = record["probe_framing"] == "one_hop"
        axes[0][column].set_title(
            record["name"].replace("  (no gate)", "   (no gate)") + "\n"
            + f"bridge '{record['bridge']}'"
            + (" — named in prompt" if named else " — never named") + "\n\n"
            + prompt_header(record),
            fontsize=8.2, linespacing=1.55, pad=12, loc="left")

    # corpus label for each row block, out to the left of the axes
    for block, (_, row_label) in enumerate(ROWS):
        box = axes[block * 2][0].get_position()
        bars = axes[block * 2 + 1][0].get_position()
        figure.text(0.006, (box.y1 + bars.y0) / 2, row_label,
                    rotation=90, va="center", ha="left", fontsize=10.5,
                    color=jp.INK, weight="semibold")

    handles, labels = axes[0][0].get_legend_handles_labels()
    order = [labels.index(x) for x in ("model", "chord over [0,1]", "tangent at $\\alpha$=0",
                                       "J-lens", "tuned lens", "logit lens")]
    handles = [handles[i] for i in order] + [
        plt.Rectangle((0, 0), 1, 1, facecolor=jp.MUTED, alpha=1.0),
        plt.Rectangle((0, 0), 1, 1, facecolor=jp.MUTED, alpha=0.4, hatch="///", edgecolor="white"),
    ]
    labels = [labels[i] for i in order] + ["bars: bridge entity", "bars: answer"]
    figure.legend(handles, labels, frameon=False, fontsize=9.5, ncol=8,
                  loc="lower center", bbox_to_anchor=(0.5, -0.018))
    figure.suptitle(
        "Each column interpolates the layer-34 residual between two prompts the model answers correctly.\n"
        "The black curve is the model; the dashed lines are what each lens predicts. "
        "Every domain corpus lifts the regression above web text (6/6); "
        "the row whose framing matches the prompt is outlined, and wins on 4/6.",
        fontsize=11.5, y=0.985, linespacing=1.55,
    )
    figure.subplots_adjust(top=0.888, left=0.075, wspace=0.16)
    figure.savefig(HERE / "fig_lens_grid.png", bbox_inches="tight")
    plt.close(figure)
    print("wrote fig_lens_grid.png")

    # ---- compact summary --------------------------------------------------
    figure, axis = plt.subplots(figsize=(7.6, 4.4))
    positions = {key: index for index, (key, _) in enumerate(ROWS)}
    for record in records:
        chord = record["chord"]
        colour = jp.LENS_COLORS["tuned"] if record["gates"] else jp.LENS_COLORS["logit"]
        values = [record["slopes"][f"tuned_{key}"] / chord for key, _ in ROWS]
        axis.plot(range(len(ROWS)), values, "-o", color=colour, lw=1.6, ms=6, alpha=0.85)
        matched = positions[record["probe_framing"]]
        axis.plot(matched, values[matched], "o", ms=12, mfc="none",
                  mec=colour, mew=2.0)
        axis.annotate(record["name"].split(":")[0], (len(ROWS) - 1, values[-1]),
                      xytext=(8, 0), textcoords="offset points", fontsize=8.5,
                      va="center", color=jp.MUTED)
    axis.axhline(1.0, color=jp.MUTED, lw=1.1, ls="-.")
    axis.text(-0.4, 1.03, "the chord", fontsize=8.5, color=jp.MUTED)
    axis.set_xticks(range(len(ROWS)), [label.replace("corpus: ", "") for _, label in ROWS],
                    fontsize=8.5)
    axis.set_xlim(-0.45, len(ROWS) - 0.35)
    axis.set_ylabel("tuned-lens slope, in units of the chord")
    axis.plot([], [], "o", color=jp.LENS_COLORS["tuned"], label="gating relations")
    axis.plot([], [], "o", color=jp.LENS_COLORS["logit"], label="non-gating")
    axis.plot([], [], "o", ms=11, mfc="none", mec=jp.MUTED, mew=1.8,
              label="framing matches the probe")
    axis.legend(frameon=False, fontsize=9, loc="lower left")
    axis.set_title("Any domain corpus beats web text (6/6);\n"
                   "the matching framing beats the mismatched one on 4/6", fontsize=10.5)
    axis.grid(axis="y", alpha=0.7)
    figure.savefig(HERE / "fig_lens_grid_summary.png", bbox_inches="tight")
    print("wrote fig_lens_grid_summary.png")


if __name__ == "__main__":
    main()
