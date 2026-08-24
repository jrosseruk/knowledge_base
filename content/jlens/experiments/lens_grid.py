"""Does the tangent/chord pattern hold across every gating example?

Addendum 4 showed, on one item, that a least-squares lens refitted on a corpus
containing the gate lands on the **chord** of the response, while the same lens
fitted on web text does not. This asks whether that generalises.

For each item, at layer 34:

  * the model's actual response to interpolating the readout residual between
    two states it really produces, plus that curve's **tangent** at alpha=0 and
    its **chord** over [0,1] -- both properties of the curve, not of any lens;
  * the J-lens, least-squares lens and logit lens slopes along the same path,
    each fitted twice: on that item's own relation corpus, and on web text.

Because every lens is linear its slope is `<w, M delta>`, so one transported
direction suffices per corpus and the whole grid is affordable.

Produces `fig_lens_grid.png`: top row domain-fitted, bottom row web-fitted, one
column per item, with a non-gating item included so the pattern's specificity is
visible.
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

# Domain corpora. These must be *diverse*, not one sentence repeated: a
# 3840-dimensional least-squares fit on a low-rank activation cloud is decided
# by the ridge term. Each family crosses entities with several frames, giving a
# few hundred distinct lines.

def _cross(entities, frames):
    return [f.format(e) for e in entities for f in frames]


ANIMAL_LEGS = _cross(
    ("spider", "ant", "bee", "dog", "cat", "horse", "cow", "bird", "beetle", "octopus",
     "crab", "sheep", "pig", "frog", "chicken", "wasp", "moth", "goat", "duck", "mouse",
     "tarantula", "scorpion", "centipede", "lobster", "grasshopper"),
    ("The number of legs on a {} is .", "A {} walks on this many legs: .",
     "Counting the legs of a {} gives .", "How many legs does a {} have? .",
     "A {} is an animal with legs.", "The legs of a {} number .",
     "Biologists note that a {} has legs.", "If you count every leg on a {}, you get ."),
) + _cross(
    ("spins webs", "barks and fetches sticks", "buzzes and makes honey",
     "marches in colonies", "purrs and chases mice", "lays eggs and has feathers",
     "grazes and moos", "gallops and neighs", "croaks by the pond", "scuttles sideways"),
    ("The animal that {} is a .", "The number of legs on the animal that {} is .",
     "An animal that {} has legs."),
)

POLYGONS = _cross(
    ("triangle", "square", "pentagon", "hexagon", "heptagon", "octagon", "rectangle",
     "rhombus", "trapezoid", "nonagon", "decagon", "quadrilateral", "parallelogram",
     "dodecagon", "kite"),
    ("The number of sides on a {} is .", "A {} has this many sides: .",
     "Counting the edges of a {} gives .", "How many sides does a {} have? .",
     "A {} is a polygon with sides.", "The vertices of a {} number .",
     "Drawing a {} requires straight lines.", "A {} is bounded by segments."),
)

SPORTS = _cross(
    ("basketball", "volleyball", "soccer", "hockey", "baseball", "cricket", "rugby",
     "netball", "handball", "lacrosse", "polo", "badminton", "tennis", "football"),
    ("The number of players on a {} team is .", "A {} team fields this many players: .",
     "How many players per side in {}? .", "In {}, each team has players on the field.",
     "A full {} squad on the court numbers .", "The starting lineup in {} is players.",
     "Teams in {} play with a side of .", "Counting one {} team gives players."),
)

VEHICLES = _cross(
    ("bicycle", "car", "tricycle", "truck", "motorcycle", "bus", "unicycle", "van",
     "trailer", "scooter", "tractor", "wagon", "rickshaw", "sedan", "lorry"),
    ("The number of wheels on a {} is .", "A {} rolls on this many wheels: .",
     "How many wheels does a {} have? .", "A {} is a vehicle with wheels.",
     "Counting the wheels of a {} gives .", "The axles of a {} carry wheels.",
     "To ride a {} you balance on wheels.", "A {} needs tyres."),
)

PLANETS = _cross(
    ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"),
    ("The color of the planet {} is .", "Seen from space, {} appears .",
     "Astronomers describe {} as .", "The surface of {} looks .",
     "{} is known for its colour.", "Through a telescope {} is .",
     "The atmosphere of {} gives it a tint.", "Photographs of {} show a world."),
) + _cross(
    ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"),
    ("The planet {} from the Sun is .", "Counting outward, the {} planet is .",
     "The {} orbit from the Sun belongs to ."),
)

ITEMS = [
    ("legs: spider → dog", True, ANIMAL_LEGS,
     "Fact: The number of legs on the animal that spins webs is ", "8",
     "Fact: The number of legs on the animal that barks and fetches sticks is ", "4"),
    ("legs: ant → bird", True, ANIMAL_LEGS,
     "Fact: The number of legs on the insect that marches in colonies is ", "6",
     "Fact: The number of legs on the animal that lays eggs and has feathers is ", "2"),
    ("sides: triangle → square", True, POLYGONS,
     "Fact: The number of sides on a triangle is ", "3",
     "Fact: The number of sides on a square is ", "4"),
    ("players: basketball → volleyball", True, SPORTS,
     "Fact: The number of players from one team on a basketball court is ", "5",
     "Fact: The number of players from one team on a volleyball court is ", "6"),
    ("wheels: bicycle → car", True, VEHICLES,
     "Fact: The number of wheels on a bicycle is ", "2",
     "Fact: The number of wheels on a car is ", "4"),
    ("colour: Mars → Earth  (no gate)", False, PLANETS,
     "Fact: The color of the planet fourth from the Sun is ", "red",
     "Fact: The color of the planet third from the Sun is ", "blue"),
]

TEMPLATES = ["Fact: {}", "It is well known that {}", "Question and answer: {}", "{}"]


def build_corpus(tokenizer, lines, seq_len, device, target_sequences, batch_size=2):
    generator = torch.Generator().manual_seed(0)
    ids: list[int] = []
    while True:
        for position, index in enumerate(torch.randperm(len(lines), generator=generator).tolist()):
            template = TEMPLATES[position % len(TEMPLATES)]
            ids.extend(tokenizer.encode(template.format(lines[index]) + "\n",
                                        add_special_tokens=False))
        if len(ids) >= target_sequences * seq_len:
            break
    usable = (len(ids) // (seq_len - 1)) * (seq_len - 1)
    rows = torch.tensor(ids[:usable]).view(-1, seq_len - 1)
    rows = torch.cat([torch.full((rows.shape[0], 1), tokenizer.bos_token_id), rows], dim=1)
    return list(rows[:target_sequences].to(device).split(batch_size))


def fit_ols(model, batches, layer, ridge):
    sources, targets = [], []
    for input_ids in batches:
        positions = jc.valid_positions(input_ids.shape[1]).to(model.device)
        stack = jc.residual_stack(model, input_ids)[:, :, positions]
        sources.append(stack[layer + 1].flatten(0, 1).float())
        targets.append(stack[model.n_layers].flatten(0, 1).float())
    source, target = torch.cat(sources), torch.cat(targets)
    mean_x, mean_y = source.mean(0).double(), target.mean(0).double()
    cx = (source - mean_x.float()).double()
    cy = (target - mean_y.float()).double()
    cov_xx = cx.T @ cx / cx.shape[0]
    cov_yx = cy.T @ cx / cx.shape[0]
    scale = torch.diagonal(cov_xx).mean()
    weight = torch.linalg.solve(
        cov_xx + ridge * scale * torch.eye(model.d_model, dtype=torch.float64, device=model.device),
        cov_yx.T,
    ).T.float()
    return weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--alpha-steps", type=int, default=61)
    parser.add_argument("--corpus-sequences", type=int, default=170)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--out", default="lens_grid.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    layer, device = args.layer, model.device
    gain = 1.0 + model.final_norm.weight.float()
    alphas = torch.linspace(0.0, 1.0, args.alpha_steps)

    def single(word):
        for candidate in (word, " " + word):
            pieces = tokenizer.encode(candidate, add_special_tokens=False)
            if len(pieces) == 1:
                return pieces[0]
        return None

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    web = jc.corpus_batches(tokenizer, dataset["text"], n_sequences=args.corpus_sequences,
                            seq_len=args.seq_len, batch_size=2, seed=5)
    print("fitting the web-text lens ...", flush=True)
    web_weight = fit_ols(model, web, layer, args.ridge)

    corpus_cache: dict[int, list] = {}
    records = []
    for name, gates, lines, base, answer, counter, counter_answer in ITEMS:
        token_a, token_b = single(answer), single(counter_answer)
        enc_a = tokenizer(base, return_tensors="pt").to(device)
        enc_b = tokenizer(counter, return_tensors="pt").to(device)
        position = enc_a.input_ids.shape[1] - 1
        w_diff = (model.lm_head.weight[token_b].float()
                  - model.lm_head.weight[token_a].float()) * gain

        start = jc.residual_stack(model, enc_a.input_ids)[layer + 1][0, -1].float()
        end = jc.residual_stack(model, enc_b.input_ids)[layer + 1][0, -1].float()
        delta = end - start
        grid = alphas.to(device)[:, None]
        path = (1 - grid) * start + grid * end

        captured: list[torch.Tensor] = []

        def grab(_m, _a, out):
            captured.append(jc._block_output(out)[:, position].float())

        for chunk in path.split(16):
            def inject(_m, _a, out, chunk=chunk):
                hidden = jc._block_output(out)
                edited = hidden.clone()
                edited[:, position] = chunk.to(hidden.dtype)
                return jc._rewrap(out, edited)

            handles = [model.layers[layer].register_forward_hook(inject),
                       model.layers[-1].register_forward_hook(grab)]
            try:
                with torch.inference_mode():
                    model.forward(enc_a.input_ids.expand(chunk.shape[0], -1))
            finally:
                for handle in handles:
                    handle.remove()
        finals = torch.cat(captured)
        actual = (finals @ w_diff).cpu().numpy()
        logits = model.unembed(finals)
        swing = float((logits[:, token_b] - logits[:, token_a])[-1]
                      - (logits[:, token_b] - logits[:, token_a])[0])

        # the curve's own tangent and chord
        tangent_store: dict[str, torch.Tensor] = {}

        def run_scaled(scale):
            def inject(_m, _a, out):
                hidden = jc._block_output(out)
                edited = hidden.clone()
                edited[:, position] = (start + scale * delta).to(hidden.dtype)
                return jc._rewrap(out, edited)

            def catch(_m, _a, out):
                tangent_store["h"] = jc._block_output(out)[:, position].float()

            handles = [model.layers[layer].register_forward_hook(inject),
                       model.layers[-1].register_forward_hook(catch)]
            try:
                model.forward(enc_a.input_ids)
            finally:
                for handle in handles:
                    handle.remove()
            return tangent_store["h"]

        with torch.no_grad():
            _, tangent_vector = torch.func.jvp(
                run_scaled, (torch.zeros((), device=device),), (torch.ones((), device=device),))
        tangent = float(tangent_vector[0] @ w_diff)
        chord = float(actual[-1] - actual[0])

        key = id(lines)
        if key not in corpus_cache:
            print(f"building + fitting domain corpus for {name} ...", flush=True)
            batches = build_corpus(tokenizer, lines, args.seq_len, device, args.corpus_sequences)
            corpus_cache[key] = (batches, fit_ols(model, batches, layer, args.ridge))
        domain_batches, domain_weight = corpus_cache[key]

        probe = delta[None]
        slopes = {
            "j_domain": float(jc.transport(model, layer, probe, domain_batches)[0] @ w_diff),
            "j_web": float(jc.transport(model, layer, probe, web)[0] @ w_diff),
            "tuned_domain": float((delta @ domain_weight.T) @ w_diff),
            "tuned_web": float((delta @ web_weight.T) @ w_diff),
            "logit": float(delta @ w_diff),
        }
        records.append({"name": name, "gates": gates, "swing": swing,
                        "alphas": alphas.tolist(), "actual": actual.tolist(),
                        "tangent": tangent, "chord": chord, "slopes": slopes})
        print(f"{name:36s} swing {swing:6.2f}  tangent {tangent:9.0f}  chord {chord:9.0f}  "
              f"| domain J {slopes['j_domain']:9.0f} tuned {slopes['tuned_domain']:9.0f}  "
              f"| web J {slopes['j_web']:9.0f} tuned {slopes['tuned_web']:9.0f}", flush=True)

    (HERE / args.out).write_text(json.dumps(
        {"model": args.model, "layer": layer, "records": records}, indent=1))

    # ---------------- the grid ----------------------------------------------
    jp.use_style()
    n = len(records)
    figure, axes = plt.subplots(2, n, figsize=(3.5 * n, 7.4), sharex=True)
    for column, record in enumerate(records):
        alphas_np = np.array(record["alphas"])
        actual = np.array(record["actual"]) - record["actual"][0]
        for row, corpus in enumerate(("domain", "web")):
            axis = axes[row][column]
            axis.axhline(0, color=jp.MUTED, lw=0.8)
            axis.plot(alphas_np, actual, color=jp.INK, lw=2.5, zorder=5, label="model")
            axis.plot(alphas_np, record["tangent"] * alphas_np, ":", lw=1.7, color=jp.INK,
                      label="tangent at $\\alpha$=0")
            axis.plot(alphas_np, record["chord"] * alphas_np, "-.", lw=1.7, color=jp.MUTED,
                      label="chord over [0,1]")
            axis.plot(alphas_np, record["slopes"][f"j_{corpus}"] * alphas_np, "--", lw=2.1,
                      color=jp.LENS_COLORS["j"], label="J-lens")
            axis.plot(alphas_np, record["slopes"][f"tuned_{corpus}"] * alphas_np, "--", lw=2.1,
                      color=jp.LENS_COLORS["tuned"], label="tuned lens")
            axis.plot(alphas_np, record["slopes"]["logit"] * alphas_np, "--", lw=1.5,
                      color=jp.LENS_COLORS["logit"], alpha=0.85, label="logit lens")
            span = max(abs(actual).max(), abs(record["chord"])) * 1.35
            axis.set_ylim(-0.45 * span, span)
            axis.grid(axis="y", alpha=0.7)
            if row == 0:
                axis.set_title(record["name"], fontsize=9.5)
            if row == 1:
                axis.set_xlabel("interpolation $\\alpha$")
            if column == 0:
                axis.set_ylabel(
                    ("lenses fitted on the\nrelation's own corpus" if corpus == "domain"
                     else "lenses fitted on\nweb text (as deployed)")
                    + "\n\nchange in readout", fontsize=9)
    axes[0][0].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    figure.suptitle(
        "Refit on the relation's own corpus (top) and the regression tracks the chord; "
        "on web text (bottom) it does not",
        fontsize=12, y=1.0,
    )
    figure.savefig(HERE / "fig_lens_grid.png")
    print("wrote fig_lens_grid.png")


if __name__ == "__main__":
    main()
