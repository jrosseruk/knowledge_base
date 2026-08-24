"""The real-model version of notebook 01's closing figure.

Notebook 01 makes the point in a hand-built 2D toy: a gated `F`, and three
linear readouts of it -- the averaged tangent, the regression chord, and the
identity. This script draws the same picture for Gemma 3 12B on a real two-hop
prompt whose downstream computation actually gates:

    "Fact: The number of legs on the animal that spins webs is "   -> 8
    "Fact: The number of legs on the animal that barks and fetches sticks is " -> 4

`h(alpha)` interpolates the layer-34 residual at the final position between the
two states the model really produces. Everything is read through one fixed
linear functional, `s(x) = <w_4 - w_8, x>`, so the model and the three lenses
are in the same units.

Left panel plots each readout as a *change* from its own value at `alpha = 0`.
That is the honest comparison: the lenses are linear approximations of `F`, so
they cannot reproduce `F`'s offset, and the claim under test is about slope and
shape, not intercept. The right panel keeps the absolute readouts, as
full-vocabulary ranks of the bridge entity and the answer at the unmodified
state -- which is the Claim 1 question rather than the Claim 2 one.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

import jlens_core as jc
import jlens_plot as jp

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("JLENS_ARTIFACTS", "/dev/shm/jlens-artifacts"))

BASE = "Fact: The number of legs on the animal that spins webs is "
COUNTER = "Fact: The number of legs on the animal that barks and fetches sticks is "
ANSWER, COUNTER_ANSWER = "8", "4"
BRIDGE, COUNTER_BRIDGE = "spider", "dog"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--corpus-sequences", type=int, default=64)
    parser.add_argument("--alpha-steps", type=int, default=81)
    parser.add_argument("--out", default="fig_spider_gate.png")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    layer = args.layer

    def single(word):
        for candidate in (word, " " + word):
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        raise ValueError(f"{word!r} is not a single token")

    ids = {name: single(name) for name in (ANSWER, COUNTER_ANSWER, BRIDGE, COUNTER_BRIDGE)}

    encodings, behaviour = {}, {}
    for role, text in (("base", BASE), ("counter", COUNTER)):
        enc = tokenizer(text, return_tensors="pt").to(model.device)
        encodings[role] = enc
        with torch.inference_mode():
            generated = hf_model.generate(**enc, max_new_tokens=6, do_sample=False,
                                          pad_token_id=tokenizer.eos_token_id)
        behaviour[role] = tokenizer.decode(generated[0, enc.input_ids.shape[1]:])
    print("behavioural control:", behaviour)

    gain = 1.0 + model.final_norm.weight.float()
    w_diff = (model.lm_head.weight[ids[COUNTER_ANSWER]].float()
              - model.lm_head.weight[ids[ANSWER]].float()) * gain

    stack_base = jc.residual_stack(model, encodings["base"].input_ids)
    stack_counter = jc.residual_stack(model, encodings["counter"].input_ids)
    position = encodings["base"].input_ids.shape[1] - 1
    start = stack_base[layer + 1][0, -1].float()
    end = stack_counter[layer + 1][0, -1].float()

    alphas = torch.linspace(-0.15, 1.15, args.alpha_steps)
    grid = alphas.to(model.device)[:, None]
    path = (1 - grid) * start + grid * end

    # --- the model's real response -----------------------------------------
    captured: list[torch.Tensor] = []

    def capture(_m, _a, out):
        captured.append(jc._block_output(out)[:, position].float())

    for chunk in path.split(16):
        def inject(_m, _a, out, chunk=chunk):
            hidden = jc._block_output(out)
            edited = hidden.clone()
            edited[:, position] = chunk.to(hidden.dtype)
            return jc._rewrap(out, edited)

        handles = [model.layers[layer].register_forward_hook(inject),
                   model.layers[-1].register_forward_hook(capture)]
        try:
            with torch.inference_mode():
                model.forward(encodings["base"].input_ids.expand(chunk.shape[0], -1))
        finally:
            for handle in handles:
                handle.remove()
    finals = torch.cat(captured)
    actual = (finals @ w_diff).cpu().numpy()

    # --- the three linear approximations ------------------------------------
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    corpus = jc.corpus_batches(tokenizer, dataset["text"], n_sequences=args.corpus_sequences,
                               seq_len=128, batch_size=2, seed=5)
    transported = jc.transport(model, layer, path, corpus)
    lines = {"j": (transported @ w_diff).cpu().numpy(),
             "logit": (path @ w_diff).cpu().numpy()}
    lens_blob = torch.load(ARTIFACTS / "tuned_lens_kl.pt", map_location="cuda", weights_only=False)
    state, form = lens_blob["state"], lens_blob.get("form", "residual")
    weight = state[layer]["A"].to(model.device, torch.float32)
    bias = state[layer]["b"].to(model.device, torch.float32)
    translated = path @ weight.T + bias
    if form == "residual":
        translated = translated + path
    lines["tuned"] = (translated @ w_diff).cpu().numpy()

    zero = int(np.argmin(np.abs(alphas.numpy())))
    one = int(np.argmin(np.abs(alphas.numpy() - 1.0)))

    # --- the tangent and the chord *of this curve* --------------------------
    # The deployed lenses are fitted on generic web text, which never explores
    # this direction, so neither is the tangent nor the chord of this sigmoid.
    # These two lines are, by definition. Notebook 01's toy compares exactly
    # these because there the fitting corpus straddles the gate along the swept
    # axis, so its regression chord and average tangent coincide with them.
    delta = end - start

    def local_tangent():
        captured_t = {}

        def run(scale):
            def inject(_m, _a, out):
                hidden = jc._block_output(out)
                edited = hidden.clone()
                edited[:, position] = (start + scale * delta).to(hidden.dtype)
                return jc._rewrap(out, edited)

            def grab(_m, _a, out):
                captured_t["h"] = jc._block_output(out)[:, position].float()

            handles = [model.layers[layer].register_forward_hook(inject),
                       model.layers[-1].register_forward_hook(grab)]
            try:
                model.forward(encodings["base"].input_ids)
            finally:
                for handle in handles:
                    handle.remove()
            return captured_t["h"]

        with torch.no_grad():
            _, tangent = torch.func.jvp(
                run, (torch.zeros((), device=model.device),),
                (torch.ones((), device=model.device),),
            )
        return float(tangent[0] @ w_diff)

    tangent_slope = local_tangent()
    chord_slope = float(actual[one] - actual[zero])
    print(f"local tangent slope at alpha=0: {tangent_slope:.1f}")
    print(f"chord slope over alpha 0->1   : {chord_slope:.1f}")

    # --- full-vocabulary ranks at the unmodified state -----------------------
    base_hidden = start
    scores = {
        "j": model.unembed(jc.transport(model, layer, base_hidden, corpus)[0]).squeeze(),
        "logit": model.unembed(base_hidden).squeeze(),
        "tuned": model.unembed(
            (base_hidden @ weight.T + bias) + (base_hidden if form == "residual" else 0)
        ).squeeze(),
    }
    ranks = {name: {word: jc.rank_of(s, ids[word]) for word in (BRIDGE, ANSWER)}
             for name, s in scores.items()}
    print(json.dumps(ranks, indent=1))

    # ------------------------------------------------------------------------
    jp.use_style()
    figure, axes = plt.subplots(1, 2, figsize=(11.4, 4.0),
                               gridspec_kw={"width_ratios": [1.45, 1]})

    axis = axes[0]
    axis.axhline(0, color=jp.MUTED, lw=0.8)
    axis.axvline(0, color=jp.MUTED, lw=0.8, ls=":")
    axis.axvline(1, color=jp.MUTED, lw=0.8, ls=":")
    axis.plot(alphas.numpy(), actual - actual[zero], color=jp.INK, lw=2.6, zorder=5,
              label="network  (the real $F_\\ell$)")
    axis.plot(alphas.numpy(), tangent_slope * alphas.numpy(), ":", lw=1.8, color=jp.INK,
              label="tangent of $F_\\ell$ at $\\alpha=0$")
    axis.plot(alphas.numpy(), chord_slope * alphas.numpy(), "-.", lw=1.8, color=jp.MUTED,
              label="chord of $F_\\ell$ over $[0,1]$")
    for key, label in (("j", "J-lens  (fitted on web text)"),
                       ("tuned", "tuned lens  (fitted on web text)"),
                       ("logit", "logit lens  (identity)")):
        axis.plot(alphas.numpy(), lines[key] - lines[key][zero],
                  "--", lw=2, color=jp.LENS_COLORS[key], label=label)
    axis.set_ylim(-19000, 47000)
    axis.set_xlabel(f"interpolation $\\alpha$    (0 = {BRIDGE},  1 = {COUNTER_BRIDGE})")
    axis.set_ylabel(f"change in readout of\nlogit({COUNTER_ANSWER}) $-$ logit({ANSWER})")
    axis.set_title(
        f"Tangent, chord, and what the lenses actually do   (layer {layer})", fontsize=10.5
    )
    # The lens lines are straight by construction -- a linear map composed with
    # an affine path is affine, so nothing else was possible. What deserves a
    # note is that their *slopes* nearly coincide even though the maps do not:
    # the slope is the single scalar <w, M d>, and by this layer the answer
    # contrast is already partly written into the residual, so `d` itself points
    # along `w` and every transport preserves that component. The lenses'
    # actual disagreement is in other directions -- see the right panel.
    axis.annotate(
        "the deployed lenses sit on neither line:\n"
        "they are fitted on web text, which never\n"
        "explores this direction",
        xy=(1.02, lines["tuned"][-6] - lines["tuned"][zero]), xytext=(0.28, -14500),
        fontsize=8, color=jp.MUTED, ha="left",
        arrowprops=dict(arrowstyle="->", color=jp.MUTED, lw=0.9),
    )
    axis.annotate(
        "network rises ~3$\\times$ further\nthan any linear readout",
        xy=(0.66, (actual - actual[zero])[int(len(actual) * 0.63)]), xytext=(0.72, 20000),
        fontsize=8.5, color=jp.INK,
        arrowprops=dict(arrowstyle="->", color=jp.INK, lw=0.9),
    )
    axis.grid(axis="y", alpha=0.7)
    axis.legend(frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.0, 1.02))

    axis = axes[1]
    names = ["j", "logit", "tuned"]
    index = np.arange(len(names))
    # Colour already encodes the lens via the x axis, so bridge vs answer is
    # carried by fill instead, and the legend uses neutral swatches.
    for offset, (word, hatch, alpha_value) in enumerate(
        ((BRIDGE, None, 1.0), (ANSWER, "///", 0.35))
    ):
        values = [ranks[n][word] for n in names]
        bars = axis.bar(index + (offset - 0.5) * 0.38, values, 0.34,
                        color=[jp.LENS_COLORS[n] for n in names],
                        alpha=alpha_value, hatch=hatch,
                        edgecolor="white", linewidth=0.8)
        axis.bar_label(bars, fmt="%d", fontsize=8.5, color=jp.MUTED, padding=2)
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=jp.MUTED, alpha=1.0,
                      label=f"'{BRIDGE}'  (bridge entity)"),
        plt.Rectangle((0, 0), 1, 1, facecolor=jp.MUTED, alpha=0.35, hatch="///",
                      edgecolor="white", label=f"'{ANSWER}'  (answer)"),
    ]
    axis.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper center")
    axis.set_yscale("log")
    axis.set_xticks(index, [jp.LENS_LABELS[n] for n in names])
    axis.set_ylabel("full-vocabulary rank\n(lower is better)")
    axis.set_title("What each lens reads at the unmodified state", fontsize=10.5)
    axis.set_ylim(1, 2400)
    axis.grid(axis="y", alpha=0.7)
    axis.legend(frameon=False, fontsize=8.5)

    figure.tight_layout()
    figure.savefig(HERE / args.out)
    print("wrote", HERE / args.out)

    (HERE / "spider_gate.json").write_text(json.dumps({
        "model": args.model, "layer": layer, "behaviour": behaviour,
        "alphas": alphas.tolist(), "actual": actual.tolist(),
        "tangent_slope": tangent_slope, "chord_slope": chord_slope,
        **{f"line_{k}": v.tolist() for k, v in lines.items()},
        "ranks_at_alpha_0": ranks,
    }, indent=1))


if __name__ == "__main__":
    main()
