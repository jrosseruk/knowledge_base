"""Render the figures for whichever experiment outputs are present."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jlens_plot as jp

HERE = Path(__file__).resolve().parent
jp.use_style()


def figure_pilot() -> None:
    path = HERE / "pilot_mars_all_layers.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = data["rows"]
    layers = [r["layer"] for r in rows]

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for axis, target, title in (
        (axes[0], "Mars", "Bridge entity  ‘Mars’"),
        (axes[1], "red", "Answer  ‘red’"),
    ):
        for lens in ("j", "logit"):
            axis.plot(
                layers,
                [r[lens][f"{target}_rank"] for r in rows],
                color=jp.LENS_COLORS[lens],
                label=jp.LENS_LABELS[lens],
            )
        axis.set_yscale("log")
        axis.invert_yaxis()
        axis.set_xlabel("layer")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.7)
        axis.axhline(1, color=jp.MUTED, lw=0.8, ls=":")

    axes[0].set_ylabel("full-vocabulary rank\n(lower is better)")
    best = min(rows, key=lambda r: r["j"]["Mars_rank"])
    axes[0].annotate(
        f"L{best['layer']}: rank 1",
        xy=(best["layer"], best["j"]["Mars_rank"]),
        xytext=(best["layer"] - 17, 3),
        color=jp.LENS_COLORS["j"],
        fontsize=9,
        arrowprops=dict(arrowstyle="-", color=jp.LENS_COLORS["j"], lw=1),
    )
    axes[1].legend(loc="lower left")
    figure.suptitle(
        f"Gemma 3 12B IT   ·   “{data['prompt'].strip()}”   ·   model says “red”",
        fontsize=10.5,
        y=1.03,
    )
    figure.savefig(HERE / "fig_pilot_mars.png")
    plt.close(figure)
    print("wrote fig_pilot_mars.png")


def figure_screen() -> None:
    path = HERE / "multihop_three_lens_screen.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    per_item = data["per_item"]
    n_layers = len(next(iter(per_item.values())))
    layers = list(range(n_layers))

    figure, axes = plt.subplots(1, 3, figsize=(13, 3.9))

    # (a) median rank of the bridge entity A, per lens, per layer
    axis = axes[0]
    for lens in ("j", "logit", "tuned"):
        median = [
            float(np.median([rows[layer][f"{lens}_A"] for rows in per_item.values()]))
            for layer in layers
        ]
        axis.plot(layers, median, color=jp.LENS_COLORS[lens], label=jp.LENS_LABELS[lens])
    axis.set_yscale("log")
    axis.invert_yaxis()
    axis.set_xlabel("layer")
    axis.set_ylabel("median rank of bridge entity A")
    axis.set_title("(a) Do the lenses see the intermediate?")
    axis.grid(axis="y", alpha=0.7)
    axis.legend(loc="lower left")

    # (b) median rank of the answer B
    axis = axes[1]
    for lens in ("j", "logit", "tuned"):
        median = [
            float(np.median([rows[layer][f"{lens}_B"] for rows in per_item.values()]))
            for layer in layers
        ]
        axis.plot(layers, median, color=jp.LENS_COLORS[lens], label=jp.LENS_LABELS[lens])
    axis.set_yscale("log")
    axis.invert_yaxis()
    axis.set_xlabel("layer")
    axis.set_ylabel("median rank of answer B")
    axis.set_title("(b) …or do they skip to the answer?")
    axis.grid(axis="y", alpha=0.7)

    # (c) pass@k for recovering A
    axis = axes[2]
    ks = [1, 5, 10]
    width = 0.26
    positions = np.arange(len(ks))
    for offset, lens in enumerate(("j", "logit", "tuned")):
        values = [data["summary"]["lenses"][lens][f"pass@{k}"] for k in ks]
        bars = axis.bar(
            positions + (offset - 1) * width,
            values,
            width - 0.03,
            color=jp.LENS_COLORS[lens],
            label=jp.LENS_LABELS[lens],
        )
        axis.bar_label(bars, fmt="%.2f", fontsize=8, color=jp.MUTED, padding=2)
    axis.set_xticks(positions, [f"pass@{k}" for k in ks])
    axis.set_ylabel(f"fraction of {data['summary']['n_items']} items")
    axis.set_title("(c) Bridge-entity recovery")
    axis.set_ylim(0, 1.12)
    axis.grid(axis="y", alpha=0.7)

    figure.savefig(HERE / "fig_multihop_screen.png")
    plt.close(figure)
    print("wrote fig_multihop_screen.png")


def figure_nonlinearity() -> None:
    path = HERE / "nonlinearity_vs_divergence.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = data["rows"]
    layers = [r["layer"] for r in rows]

    figure, axis = plt.subplots(figsize=(7.2, 4.1))
    axis.plot(
        layers,
        [r["cos_J_splithalf"] for r in rows],
        color=jp.MUTED,
        ls="--",
        label="split-half reliability of J  (noise ceiling)",
    )
    axis.plot(
        layers,
        [r["cos_J_vs_regression"] for r in rows],
        color=jp.LENS_COLORS["j"],
        label=r"$\cos(J_\ell d,\ A_\ell d)$",
    )
    axis.fill_between(
        layers,
        [r["cos_J_vs_regression"] for r in rows],
        [r["cos_J_splithalf"] for r in rows],
        color=jp.LENS_COLORS["j"],
        alpha=0.10,
        lw=0,
    )
    axis.set_xlabel("layer $\\ell$")
    axis.set_ylabel("cosine similarity")
    axis.set_title(
        "If $F_\\ell$ were affine, $J_\\ell = A_\\ell$ exactly.\n"
        "The shaded gap is the measured departure from affineness.",
        fontsize=10,
    )
    axis.grid(axis="y", alpha=0.7)
    axis.legend(loc="lower right")
    figure.savefig(HERE / "fig_nonlinearity.png")
    plt.close(figure)
    print("wrote fig_nonlinearity.png")


def figure_sweep() -> None:
    path = HERE / "causal_sweep.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    results = [r for r in data["results"]]
    if not results:
        return

    figure, axes = plt.subplots(
        1, len(results), figsize=(5.6 * len(results), 4.3), squeeze=False
    )
    for axis, record in zip(axes[0], results):
        sweep = record["sweeps"]["bridge"]
        alphas = np.array(sweep["alphas"])
        actual = np.array(sweep["actual"])
        axis.plot(alphas, actual, color=jp.INK, lw=2.4, label="model  $s(F_\\ell(h(\\alpha)))$", zorder=5)
        for key, lens in (("jlens", "j"), ("tuned_ols", "tuned"), ("logit", "logit")):
            axis.plot(
                alphas,
                np.array(sweep[key]),
                color=jp.LENS_COLORS[lens],
                label=jp.LENS_LABELS[lens],
            )
        axis.axvline(0, color=jp.MUTED, lw=0.8, ls=":")
        axis.axvline(1, color=jp.MUTED, lw=0.8, ls=":")
        axis.set_xlabel(r"swap strength $\alpha$   (0 = model, 1 = full swap)")
        axis.set_ylabel(
            f"$s(x)$: unnormalised logit"
            f"({record['item']['swap_answer']}) − logit({record['item']['answer']})"
        )
        axis.set_title(
            f"{record['name']}  ·  layer {record['layer']}  ({record['role']})", fontsize=10
        )
        axis.grid(axis="y", alpha=0.7)
    axes[0][0].legend(loc="upper left")
    figure.savefig(HERE / "fig_causal_sweep.png")
    plt.close(figure)
    print("wrote fig_causal_sweep.png")


def figure_interpolation() -> None:
    path = HERE / "activation_interpolation.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    records = [r for r in data["results"] if r.get("behaviourally_correct")]
    workspace = [r for r in records if r["layer"] != 46]
    if not workspace:
        return

    figure, axes = plt.subplots(
        2, len(workspace), figsize=(4.6 * len(workspace), 7.4), squeeze=False
    )
    for column, record in enumerate(workspace):
        pair = record["pair"]
        control = next(
            (r for r in records if r["pair"]["name"] == pair["name"] and r["layer"] == 46), None
        )
        for row, panel in enumerate((record, control)):
            axis = axes[row][column]
            if panel is None:
                axis.axis("off")
                continue
            alphas = np.array(panel["alphas"])
            actual = np.array(panel["actual"])
            axis.plot(alphas, actual, color=jp.INK, lw=2.4, zorder=5,
                      label=r"model  $s(F_\ell(h(\alpha)))$")
            # the straight line through the endpoints: if the model tracks it,
            # the computation is linear along this path
            axis.plot(alphas, actual[0] + (actual[-1] - actual[0]) * alphas,
                      color=jp.MUTED, lw=1.2, ls="--", label="straight line through endpoints")
            for key, lens in (("jlens", "j"), ("tuned_kl", "tuned"), ("logit", "logit")):
                if key in panel:
                    axis.plot(alphas, np.array(panel[key]), color=jp.LENS_COLORS[lens],
                              label=jp.LENS_LABELS[lens])
            axis.set_xlabel(
                "interpolation alpha   (0 = " + pair["bridge"] + ", 1 = " + pair["counter_bridge"] + ")"
            )
            axis.set_ylabel(f"$s$: logit({pair['counter_answer']}) - logit({pair['answer']})")
            axis.grid(axis="y", alpha=0.7)
            axis.set_title(
                f"{pair['name']}  ·  layer {panel['layer']}"
                + ("   (workspace)" if panel is record else "   (late control)"),
                fontsize=9.5,
            )
    axes[0][0].legend(loc="best", fontsize=8)
    figure.suptitle(
        "Interpolating between two real internal states: the model's response is a straight line",
        fontsize=11, y=1.0,
    )
    figure.savefig(HERE / "fig_interpolation.png")
    plt.close(figure)
    print("wrote fig_interpolation.png")


def _main() -> None:
    figure_pilot()
    figure_interpolation()
    figure_screen()
    figure_nonlinearity()
    figure_sweep()
    figure_shape()
    figure_in_domain()
    figure_integrated()


def figure_shape() -> None:
    path = HERE / "shape_paired_test.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    rows = sorted(data["rows"], key=lambda r: r["widths"]["workspace"])

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.3),
                               gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) the sharpest and the flattest response, side by side
    axis = axes[0]
    sharp, flat = rows[0], rows[-1]
    for record, style, colour in ((sharp, "-", jp.INK), (flat, "-", jp.LENS_COLORS["tuned"])):
        alphas = np.array(record["alphas"])
        response = np.array(record[f"response_workspace"])
        normalised = (response - response[0]) / (response[-1] - response[0])
        axis.plot(alphas, normalised, style, color=colour, lw=2.4,
                  label=f"{record['name']}  (width {record['widths']['workspace']:.2f})")
    axis.plot([0, 1], [0, 1], color=jp.MUTED, lw=1.2, ls="--", label="linear response")
    axis.set_xlabel("interpolation alpha")
    axis.set_ylabel("normalised logit(B') - logit(B)")
    axis.set_title(f"(a) Response shape at layer {data['workspace_layer']}", fontsize=10)
    axis.grid(alpha=0.7)
    axis.legend(loc="upper left", fontsize=8)

    # (b) paired workspace vs control width, one line per item
    axis = axes[1]
    for record in rows:
        workspace = record["widths"]["workspace"]
        control = record["widths"]["control"]
        colour = jp.LENS_COLORS["j"] if workspace < control else jp.LENS_COLORS["tuned"]
        axis.plot([0, 1], [workspace, control], "-o", color=colour, lw=1.6, ms=5, alpha=0.85)
        axis.annotate(record["name"], (0, workspace), xytext=(-6, 0),
                      textcoords="offset points", ha="right", va="center",
                      fontsize=7.5, color=jp.MUTED)
    axis.set_xticks([0, 1], [f"layer {data['workspace_layer']}\n(workspace)",
                             f"layer {data['control_layer']}\n(late control)"])
    axis.set_xlim(-0.75, 1.25)
    axis.set_ylabel("10–90% transition width")
    axis.set_title(
        f"(b) Sharper at the workspace layer on "
        f"{data['summary']['n_workspace_sharper']}/{data['summary']['n_items']} items",
        fontsize=10,
    )
    axis.grid(axis="y", alpha=0.7)

    figure.savefig(HERE / "fig_shape_paired.png")
    plt.close(figure)
    print("wrote fig_shape_paired.png")




def figure_in_domain() -> None:
    path = HERE / "in_domain_lenses.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    alphas = np.array(data["alphas"])
    actual = np.array(data["actual"])
    zero = int(np.argmin(np.abs(alphas)))
    tangent, chord = data["tangent_slope"], data["chord_slope"]

    figure, axes = plt.subplots(1, 2, figsize=(12.4, 4.4), sharey=True)
    titles = {"web": "lenses fitted on web text\n(as deployed)",
              "in_domain": "lenses refitted on the animal family\n(corpus straddles the gate)"}
    for axis, corpus in zip(axes, ("web", "in_domain")):
        axis.axhline(0, color=jp.MUTED, lw=0.8)
        axis.plot(alphas, actual - actual[zero], color=jp.INK, lw=2.6, zorder=5,
                  label="network  (the real $F_\\ell$)")
        axis.plot(alphas, tangent * alphas, ":", lw=1.8, color=jp.INK,
                  label=f"tangent at $\\alpha$=0  ({tangent:,.0f})")
        axis.plot(alphas, chord * alphas, "-.", lw=1.8, color=jp.MUTED,
                  label=f"chord over [0,1]  ({chord:,.0f})")
        axis.plot(alphas, np.array(data[corpus]["j_line"]) - data[corpus]["j_line"][zero],
                  "--", lw=2.2, color=jp.LENS_COLORS["j"],
                  label=f"J-lens  ({data[corpus]['j_slope']:,.0f})")
        axis.plot(alphas, np.array(data[corpus]["tuned_line"]) - data[corpus]["tuned_line"][zero],
                  "--", lw=2.2, color=jp.LENS_COLORS["tuned"],
                  label=f"tuned lens  ({data[corpus]['tuned_slope']:,.0f})")
        axis.plot(alphas, np.array(data["logit_line"]) - data["logit_line"][zero],
                  "--", lw=1.6, color=jp.LENS_COLORS["logit"], alpha=0.8, label="logit lens")
        axis.set_xlabel("interpolation $\\alpha$   (0 = spider, 1 = dog)")
        axis.set_title(titles[corpus], fontsize=10)
        axis.grid(axis="y", alpha=0.7)
        axis.legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_ylabel("change in readout of\nlogit(4) $-$ logit(8)")
    figure.suptitle(
        "Refit the regression where the gate actually lives, and it becomes the chord",
        fontsize=11.5, y=1.02,
    )
    figure.savefig(HERE / "fig_in_domain.png")
    plt.close(figure)
    print("wrote fig_in_domain.png")



def figure_integrated() -> None:
    path = HERE / "integrated_lens.json"
    if not path.exists():
        return
    data = json.loads(path.read_text())
    sigmas = [r["sigma"] for r in data["rows"]]
    probes = data["probes"]

    figure, axes = plt.subplots(1, len(probes), figsize=(4.5 * len(probes), 4.1),
                               sharey=True, squeeze=False)
    for axis, probe in zip(axes[0], probes):
        name = probe["name"]
        bridge = [r["probes"][name]["bridge_rank"] for r in data["rows"]]
        answer = [r["probes"][name]["answer_rank"] for r in data["rows"]]
        axis.plot(sigmas, bridge, "-o", color=jp.LENS_COLORS["j"], ms=5,
                  label=f"bridge  '{probe['bridge']}'")
        axis.plot(sigmas, answer, "-o", color=jp.LENS_COLORS["tuned"], ms=5,
                  label=f"answer  '{probe['answer']}'")
        # where the readout flips from current content to eventual output
        crossings = [s for s, b, a in zip(sigmas, bridge, answer) if a < b]
        if crossings:
            axis.axvline(min(crossings), color=jp.MUTED, lw=0.9, ls=":")
            axis.annotate("readout flips", (min(crossings), 1.6), fontsize=8,
                          color=jp.MUTED, ha="center")
        axis.set_yscale("log")
        axis.invert_yaxis()
        axis.set_xlabel("intervention scale $\\sigma$")
        axis.set_title(name, fontsize=10)
        axis.grid(axis="y", alpha=0.7)
        axis.legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[0][0].set_ylabel("full-vocabulary rank\n(lower is better)")
    figure.suptitle(
        "One knob spans the two lenses: $\\sigma\\to0$ reads current content, "
        "finite $\\sigma$ reads the eventual output",
        fontsize=11, y=1.02,
    )
    figure.savefig(HERE / "fig_integrated_lens.png")
    plt.close(figure)
    print("wrote fig_integrated_lens.png")

if __name__ == "__main__":
    _main()
