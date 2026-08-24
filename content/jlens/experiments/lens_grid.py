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

# Domain corpora, in two framings per family. "In-domain" turns out to mean the
# prompt *construction*, not the topic (CLAIMS.md Addendum 8), so each family
# supplies a one-hop set (entity named) and a two-hop set (entity described).
# Corpora must also be diverse: a 3840-dimensional least-squares fit on a
# low-rank activation cloud is decided by the ridge term.

def _cross(items, frames):
    return [f.format(i) for i in items for f in frames]


FAMILIES = {
    "animals": {
        "named": ("spider", "ant", "bee", "dog", "cat", "horse", "cow", "bird", "beetle",
                  "octopus", "crab", "sheep", "pig", "frog", "chicken", "wasp", "moth",
                  "goat", "duck", "mouse", "tarantula", "scorpion", "lobster", "grasshopper"),
        "described": ("spins webs", "barks and fetches sticks", "buzzes and makes honey",
                      "marches in colonies", "purrs and chases mice",
                      "lays eggs and has feathers", "grazes and moos", "gallops and neighs",
                      "roots in mud", "croaks by the pond", "scuttles sideways on the beach",
                      "gives us wool"),
        "one_hop": ("The number of legs on a {} is .", "A {} walks on this many legs: .",
                    "Counting the legs of a {} gives .", "How many legs does a {} have? .",
                    "A {} is an animal with legs.", "The legs of a {} number .",
                    "If you count every leg on a {}, you get ."),
        "two_hop": ("The number of legs on the animal that {} is .",
                    "The animal that {} walks on this many legs: .",
                    "Counting the legs of the creature that {} gives .",
                    "The creature that {} has legs.",
                    "How many legs has the animal that {}? .",
                    "If you count the legs on the animal that {}, you get ."),
    },
    "polygons": {
        "named": ("triangle", "square", "pentagon", "hexagon", "heptagon", "octagon",
                  "rectangle", "rhombus", "trapezoid", "nonagon", "decagon", "quadrilateral",
                  "parallelogram", "dodecagon"),
        "described": ("a stop sign has", "a chessboard cell has", "a slice of pizza has",
                      "the Pentagon building has", "a honeycomb cell has",
                      "a yield sign has", "a football pitch has", "a snowflake arm traces"),
        "one_hop": ("The number of sides on a {} is .", "A {} has this many sides: .",
                    "Counting the edges of a {} gives .", "How many sides does a {} have? .",
                    "A {} is a polygon with sides.", "The vertices of a {} number .",
                    "Drawing a {} needs straight lines."),
        "two_hop": ("The number of sides on the shape that {} is .",
                    "The shape that {} has this many sides: .",
                    "Counting the edges of the shape that {} gives .",
                    "The shape that {} is a polygon with sides.",
                    "How many sides has the shape that {}? .",
                    "The figure that {} is bounded by segments."),
    },
    "sports": {
        "named": ("basketball", "volleyball", "soccer", "hockey", "baseball", "cricket",
                  "rugby", "netball", "handball", "lacrosse", "polo", "badminton", "tennis"),
        "described": ("LeBron James plays", "Wayne Gretzky played",
                      "Lionel Messi plays", "Serena Williams played",
                      "is played with an orange ball and a hoop",
                      "is played by spiking a ball over a high net",
                      "is played on ice with sticks and a puck",
                      "is played by kicking a ball into a goal",
                      "is played with a bat and wickets"),
        "one_hop": ("The number of players on a {} team is .",
                    "A {} team fields this many players: .",
                    "How many players per side in {}? .",
                    "In {}, each team has players on the field.",
                    "The starting lineup in {} is players.",
                    "Teams in {} play with a side of ."),
        "two_hop": ("The number of players on a team in the sport {} is .",
                    "The sport {} fields this many players: .",
                    "In the sport {}, each side has players.",
                    "How many players per side in the game {}? .",
                    "The game {} is played by teams of ."),
    },
    "vehicles": {
        "named": ("bicycle", "car", "tricycle", "truck", "motorcycle", "bus", "unicycle",
                  "van", "trailer", "scooter", "tractor", "wagon", "sedan", "lorry"),
        "described": ("you pedal with two feet", "commuters drive to work",
                      "a toddler rides with three wheels", "hauls freight on motorways",
                      "carries schoolchildren in the morning", "a circus performer balances on",
                      "ploughs a field"),
        "one_hop": ("The number of wheels on a {} is .", "A {} rolls on this many wheels: .",
                    "How many wheels does a {} have? .", "A {} is a vehicle with wheels.",
                    "Counting the wheels of a {} gives .", "A {} needs tyres."),
        "two_hop": ("The number of wheels on the vehicle that {} is .",
                    "The vehicle that {} rolls on this many wheels: .",
                    "How many wheels has the vehicle that {}? .",
                    "The vehicle that {} is fitted with tyres.",
                    "Counting the wheels of the vehicle that {} gives ."),
    },
    "planets": {
        "named": ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"),
        "described": ("is fourth from the Sun", "is third from the Sun",
                      "is second from the Sun", "is closest to the Sun",
                      "is sixth from the Sun and has rings", "is largest in the solar system",
                      "is eighth from the Sun", "is seventh from the Sun"),
        "one_hop": ("The color of the planet {} is .", "Seen from space, {} appears .",
                    "Astronomers describe {} as .", "The surface of {} looks .",
                    "Through a telescope {} is .", "Photographs of {} show a world."),
        "two_hop": ("The color of the planet that {} is .",
                    "The planet that {} appears .",
                    "Seen from space, the planet that {} looks .",
                    "Astronomers describe the planet that {} as .",
                    "Through a telescope the planet that {} is ."),
    },
}


def family_corpus(family: str, framing: str) -> list[str]:
    spec = FAMILIES[family]
    if framing == "one_hop":
        return _cross(spec["named"], spec["one_hop"])
    if framing == "two_hop":
        return _cross(spec["described"], spec["two_hop"])
    return _cross(spec["named"], spec["one_hop"]) + _cross(spec["described"], spec["two_hop"])


FRAMINGS = ("two_hop", "one_hop", "mixed", "web")

# (name, gates, family, probe framing, bridge, prompt A, answer A, prompt B, answer B)
# Every probe is two-hop: the bridge entity is never named, so the model has to
# retrieve it before it can answer. That is the setting the J-lens is for.
ITEMS = [
    ("legs: spider → dog", True, "animals", "two_hop", "spider",
     "Fact: The number of legs on the animal that spins webs is ", "8",
     "Fact: The number of legs on the animal that barks and fetches sticks is ", "4"),
    ("legs: frog → bee", True, "animals", "two_hop", "frog",
     "Fact: The number of legs on the animal that croaks by the pond is ", "4",
     "Fact: The number of legs on the animal that buzzes and makes honey is ", "6"),
    ("sides: stop sign → pizza slice", True, "polygons", "two_hop", "octagon",
     "Fact: The number of sides on the shape of a stop sign is ", "8",
     "Fact: The number of sides on the shape of a slice of pizza is ", "3"),
    ("players: LeBron → Gretzky", True, "sports", "two_hop", "basketball",
     "Fact: The number of players on one team in the sport LeBron James plays is ", "5",
     "Fact: The number of players on one team in the sport Wayne Gretzky played is ", "6"),
    ("wheels: pedal → commute", True, "vehicles", "two_hop", "bicycle",
     "Fact: The number of wheels on the vehicle you pedal to work is ", "2",
     "Fact: The number of wheels on the vehicle commuters drive to work is ", "4"),
    ("colour: Mars → Earth  (no gate)", False, "planets", "two_hop", "Mars",
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
    ).T
    bias = (mean_y - weight @ mean_x).float()
    return weight.float(), bias


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
    web_weight, web_bias = fit_ols(model, web, layer, args.ridge)

    corpus_cache: dict[int, list] = {}
    records = []
    for name, gates, family, probe_framing, bridge, base, answer, counter, counter_answer in ITEMS:
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

        # one lens pair per framing, plus web text
        slopes, ranks = {}, {}
        bridge_id = single(bridge)
        for framing in FRAMINGS:
            if framing == "web":
                batches, weight, bias = web, web_weight, web_bias
            else:
                key = (family, framing)
                if key not in corpus_cache:
                    lines = family_corpus(family, framing)
                    print(f"  fitting {family}/{framing} ({len(lines)} lines) ...", flush=True)
                    batches = build_corpus(tokenizer, lines, args.seq_len, device,
                                           args.corpus_sequences)
                    corpus_cache[key] = (batches, *fit_ols(model, batches, layer, args.ridge))
                batches, weight, bias = corpus_cache[key]

            probe_direction = torch.stack([delta, start])
            transported = jc.transport(model, layer, probe_direction, batches)
            slopes[f"j_{framing}"] = float(transported[0] @ w_diff)
            slopes[f"tuned_{framing}"] = float((delta @ weight.T) @ w_diff)
            scores = {
                "j": model.unembed(transported[1]).squeeze(),
                "tuned": model.unembed(start @ weight.T + bias).squeeze(),
                "logit": model.unembed(start).squeeze(),
            }
            ranks[framing] = {
                lens: {"bridge": jc.rank_of(value, bridge_id),
                       "answer": jc.rank_of(value, token_a)}
                for lens, value in scores.items()
            }
            print(f"  {framing:8s} tuned/chord {slopes[f'tuned_{framing}'] / chord:5.2f}   "
                  f"J/chord {slopes[f'j_{framing}'] / chord:5.2f}   "
                  f"J ranks {ranks[framing]['j']['bridge']}/{ranks[framing]['j']['answer']}  "
                  f"tuned {ranks[framing]['tuned']['bridge']}/{ranks[framing]['tuned']['answer']}",
                  flush=True)
        slopes["logit"] = float(delta @ w_diff)

        best = max(FRAMINGS, key=lambda f: slopes[f"tuned_{f}"] / chord)
        records.append({"name": name, "gates": gates, "family": family,
                        "probe_framing": probe_framing, "bridge": bridge, "answer": answer,
                        "swing": swing, "ranks": ranks, "best_framing": best,
                        "alphas": alphas.tolist(), "actual": actual.tolist(),
                        "tangent": tangent, "chord": chord, "slopes": slopes})
        print(f"  -> probe is {probe_framing}; best corpus is {best}"
              f"  {'MATCH' if best == probe_framing else 'no match'}\n", flush=True)

    (HERE / args.out).write_text(json.dumps(
        {"model": args.model, "layer": layer, "records": records}, indent=1))

    matched = sum(r["best_framing"] == r["probe_framing"] for r in records)
    print(f"\nbest corpus framing matched the probe's on {matched}/{len(records)} items")
    print("run plot_lens_grid.py to render")


if __name__ == "__main__":
    main()
