"""Which relations gate, and does an in-domain corpus rescue the ones that don't?

CLAIMS.md Addendum 5. For each item, at one workspace layer and one late
control layer:

  * the model's own response along an interpolation between two real internal
    states -- tangent at alpha=0, chord over [0,1], 10-90 transition width;
  * the J-lens and least-squares slopes along that same path, each fitted
    **twice**: once on web text, once on the item's own family corpus.

Both lenses are linear, so a slope is `<w, M delta>` and needs a single
transported direction rather than the whole path. That is what makes fitting
two lenses per family affordable.

The families span three groups -- numeric answers, discrete non-numeric
answers, and graded categorical answers -- because the first two discriminate
between "counting is special" and "any small closed answer set is thresholded".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

import jlens_core as jc

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
P = "Fact: "

# ---------------------------------------------------------------------------
# Families. Each has probe items and a generator for its own in-domain corpus.
# ---------------------------------------------------------------------------

FAMILIES = {
    # ---- numeric answers -------------------------------------------------
    "animal-legs": {
        "group": "numeric",
        "items": [
            ("spider-dog", "The number of legs on the animal that spins webs is ", "8",
             "The number of legs on the animal that barks and fetches sticks is ", "4"),
            ("ant-bird", "The number of legs on the insect that marches in colonies is ", "6",
             "The number of legs on the animal that lays eggs and has feathers is ", "2"),
            ("spider-bee", "The number of legs on the animal that spins webs is ", "8",
             "The number of legs on the animal that buzzes and makes honey is ", "6"),
        ],
        "corpus": [
            f"The number of legs on {a} is ." for a in
            ("a spider", "an ant", "a bee", "a dog", "a cat", "a horse", "a cow", "a bird",
             "a beetle", "an octopus", "a crab", "a sheep", "a pig", "a frog", "a chicken")
        ] + [
            f"The animal that {d} is a ." for d in
            ("spins webs", "barks and fetches sticks", "buzzes and makes honey",
             "marches in colonies", "purrs and chases mice", "lays eggs and has feathers",
             "grazes and moos", "gallops and neighs", "roots in mud", "croaks by the pond")
        ],
    },
    "geometry-sides": {
        "group": "numeric",
        "items": [
            ("triangle-square", "The number of sides on a triangle is ", "3",
             "The number of sides on a square is ", "4"),
            ("pentagon-hexagon", "The number of sides on a pentagon is ", "5",
             "The number of sides on a hexagon is ", "6"),
        ],
        "corpus": [
            f"The number of sides on a {s} is ." for s in
            ("triangle", "square", "pentagon", "hexagon", "heptagon", "octagon", "rectangle",
             "rhombus", "trapezoid", "nonagon", "decagon")
        ] + [
            f"A {s} is a shape with corners." for s in
            ("triangle", "square", "pentagon", "hexagon", "octagon", "rectangle")
        ],
    },
    "sport-counts": {
        "group": "numeric",
        "items": [
            ("basketball-volleyball",
             "The number of players from one team on a basketball court is ", "5",
             "The number of players from one team on a volleyball court is ", "6"),
            ("arc-inside",
             "The number of points awarded for a basketball shot from beyond the arc is ", "3",
             "The number of points awarded for a basketball shot from inside the arc is ", "2"),
        ],
        "corpus": [
            f"The number of players on a {s} team is ." for s in
            ("basketball", "volleyball", "soccer", "hockey", "baseball", "cricket",
             "rugby", "tennis", "netball", "handball")
        ] + [
            f"In {s}, a team scores points by ." for s in
            ("basketball", "volleyball", "soccer", "hockey", "baseball", "rugby")
        ] + [
            "The number of points for a shot from beyond the arc is .",
            "The number of points for a free throw is .",
            "The number of quarters in a basketball game is .",
        ],
    },
    # ---- discrete but not numeric ----------------------------------------
    # Phrasings avoid question forms: Gemma answers "The state of matter of
    # water is" with a multiple-choice list, but completes "Water at room
    # temperature is a" directly.
    "antonym": {
        "group": "discrete",
        "items": [
            ("hot-large", "In one word, the opposite of hot is ", "cold",
             "In one word, the opposite of large is ", "small"),
            ("up-wet", "In one word, the opposite of up is ", "down",
             "In one word, the opposite of wet is ", "dry"),
            ("light-fast", "In one word, the opposite of light is ", "dark",
             "In one word, the opposite of fast is ", "slow"),
        ],
        "corpus": [f"In one word, the opposite of {w} is ." for w in
                   ("hot", "cold", "big", "small", "large", "up", "down", "wet", "dry",
                    "light", "dark", "fast", "slow", "hard", "soft", "old", "new",
                    "rich", "poor", "near", "far", "high", "low", "open", "shut",
                    "strong", "weak", "clean", "dirty", "empty", "full", "sharp")],
    },
    "state-of-matter": {
        "group": "discrete",
        "items": [
            ("water-iron", "At room temperature, water is a ", "liquid",
             "At room temperature, iron is a ", "solid"),
            ("oxygen-gold", "At room temperature, oxygen is a ", "gas",
             "At room temperature, gold is a ", "solid"),
        ],
        "corpus": [f"At room temperature, {m} is a ." for m in
                   ("water", "iron", "oxygen", "gold", "mercury", "nitrogen", "copper",
                    "helium", "lead", "alcohol", "silver", "hydrogen", "aluminium",
                    "petrol", "argon", "zinc", "olive oil", "neon", "tin", "milk",
                    "carbon dioxide", "wax", "steam", "granite", "vinegar", "chlorine")],
    },
    "parity": {
        "group": "discrete",
        "items": [
            ("seven-eight", "The number seven is ", "odd",
             "The number eight is ", "even"),
            ("thirteen-twenty", "The number thirteen is ", "odd",
             "The number twenty is ", "even"),
        ],
        "corpus": [f"The number {n} is ." for n in
                   ("one", "two", "three", "four", "five", "six", "seven", "eight",
                    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
                    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
                    "forty", "fifty", "sixty", "seventy", "eighty", "ninety")],
    },
    "animal-class": {
        "group": "discrete",
        "items": [
            ("snake-salmon", "The animal that slithers and has scales is a ", "reptile",
             "The animal that swims in rivers and has gills is a ", "fish"),
            ("bat-eagle", "The only mammal that truly flies is a ", "bat",
             "The bird that is a symbol of the United States is an ", "eagle"),
        ],
        "corpus": [f"A {a} is a ." for a in
                   ("snake", "salmon", "eagle", "bat", "lizard", "shark", "frog", "whale",
                    "crocodile", "penguin", "trout", "toad", "dolphin", "turtle", "owl",
                    "newt", "cod", "seal", "ostrich", "gecko", "tuna", "bear")]
                  + [f"The class of animal that {d} is a ." for d in
                     ("slithers and has scales", "swims and has gills",
                      "has feathers and lays eggs", "has fur and feeds milk")],
    },
    # ---- graded categorical ----------------------------------------------
    "planet-colour": {
        "group": "graded",
        "items": [
            ("mars-earth", "The color of the planet fourth from the Sun is ", "red",
             "The color of the planet third from the Sun is ", "blue"),
        ],
        "corpus": [
            f"The color of the planet {p} is ." for p in
            ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
        ] + [
            f"The planet {n} from the Sun is ." for n in
            ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
        ],
    },
    "capital-language": {
        "group": "graded",
        "items": [
            ("paris-madrid", "The language spoken in the country whose capital is Paris is ", "French",
             "The language spoken in the country whose capital is Madrid is ", "Spanish"),
            ("rome-berlin", "The language spoken in the country whose capital is Rome is ", "Italian",
             "The language spoken in the country whose capital is Berlin is ", "German"),
        ],
        "corpus": [
            f"The capital city of {c} is ." for c in
            ("France", "Spain", "Italy", "Germany", "Portugal", "Japan", "Greece", "Poland",
             "Sweden", "Norway", "Austria", "Hungary")
        ] + [
            f"The language spoken in {c} is ." for c in
            ("France", "Spain", "Italy", "Germany", "Portugal", "Japan", "Greece", "Poland",
             "Sweden", "Norway", "Austria", "Hungary")
        ],
    },
}

TEMPLATES = ["Fact: {}", "It is well known that {}", "Question and answer: {}", "{}"]


def build_corpus(tokenizer, lines, seq_len, device, target_sequences=170, batch_size=2):
    """Tokenize a family's prompt set into fixed-length sequences.

    `d_model` is 3840, so an OLS fit needs well over that many token positions
    or it is decided entirely by the ridge term. We repeat the family's lines,
    reshuffled and re-templated each pass, until the corpus reaches
    `target_sequences` -- roughly 19k valid positions at seq_len 128.
    """
    generator = torch.Generator().manual_seed(0)
    per_line = max(1, sum(len(tokenizer.encode(x)) for x in lines[:8]) // 8)
    repeats = max(4, int(target_sequences * seq_len / max(1, per_line * len(lines))) + 1)
    ids: list[int] = []
    for _ in range(repeats):
        order = torch.randperm(len(lines), generator=generator).tolist()
        for index in order:
            template = TEMPLATES[index % len(TEMPLATES)]
            ids.extend(tokenizer.encode(template.format(lines[index]) + "\n",
                                        add_special_tokens=False))
    bos = tokenizer.bos_token_id
    usable = (len(ids) // (seq_len - 1)) * (seq_len - 1)
    rows = torch.tensor(ids[:usable]).view(-1, seq_len - 1)
    rows = torch.cat([torch.full((rows.shape[0], 1), bos), rows], dim=1)
    rows = rows[:target_sequences]
    return list(rows.to(device).split(batch_size)), rows.shape[0]


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
    return weight, source.shape[0]


def transition_width(alphas, response):
    low, high = response[0], response[-1]
    if abs(high - low) < 1e-6:
        return float("nan")
    scaled = [(v - low) / (high - low) for v in response]
    inside = [a for a, v in zip(alphas, scaled) if 0.1 <= v <= 0.9]
    return float((max(inside) - min(inside)) / (alphas[-1] - alphas[0])) if inside else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--control-layer", type=int, default=46)
    parser.add_argument("--alpha-steps", type=int, default=61)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--web-sequences", type=int, default=64)
    parser.add_argument("--out", default="gate_survey.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    gain = 1.0 + model.final_norm.weight.float()
    alphas = torch.linspace(0.0, 1.0, args.alpha_steps)

    def single(word):
        for candidate in (word, " " + word):
            pieces = tokenizer.encode(candidate, add_special_tokens=False)
            if len(pieces) == 1:
                return pieces[0]
        return None

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    web = jc.corpus_batches(tokenizer, dataset["text"], n_sequences=args.web_sequences,
                            seq_len=args.seq_len, batch_size=2, seed=5)
    print("fitting the web-text OLS lenses ...", flush=True)
    web_weight = {layer: fit_ols(model, web, layer, args.ridge)[0]
                  for layer in (args.layer, args.control_layer)}

    rows, skipped = [], []
    for family, spec in FAMILIES.items():
        domain, n_sequences = build_corpus(tokenizer, spec["corpus"], args.seq_len, model.device)
        print(f"\n=== {family} ({spec['group']}) : {len(spec['corpus'])} lines "
              f"-> {n_sequences} sequences ===", flush=True)
        domain_weight = {layer: fit_ols(model, domain, layer, args.ridge)[0]
                         for layer in (args.layer, args.control_layer)}

        for name, base, answer, counter, counter_answer in spec["items"]:
            token_a, token_b = single(answer), single(counter_answer)
            if token_a is None or token_b is None:
                skipped.append((family, name, "multi-token answer")); continue

            encodings, behaviour = {}, {}
            for role, text in (("base", P + base), ("counter", P + counter)):
                enc = tokenizer(text, return_tensors="pt").to(model.device)
                encodings[role] = enc
                with torch.inference_mode():
                    generated = hf_model.generate(**enc, max_new_tokens=6, do_sample=False,
                                                  pad_token_id=tokenizer.eos_token_id)
                behaviour[role] = tokenizer.decode(generated[0, enc.input_ids.shape[1]:])
            if not (behaviour["base"].strip().lower().startswith(answer.lower())
                    and behaviour["counter"].strip().lower().startswith(counter_answer.lower())):
                skipped.append((family, name, f"behaviour {behaviour}"))
                print(f"  skip {name}: {behaviour}", flush=True); continue

            w_diff = (model.lm_head.weight[token_b].float()
                      - model.lm_head.weight[token_a].float()) * gain
            stack_a = jc.residual_stack(model, encodings["base"].input_ids)
            stack_b = jc.residual_stack(model, encodings["counter"].input_ids)
            position = encodings["base"].input_ids.shape[1] - 1

            record = {"family": family, "group": spec["group"], "name": name,
                      "answer": answer, "counter_answer": counter_answer,
                      "behaviour": behaviour, "layers": {}}

            for role, layer in (("workspace", args.layer), ("control", args.control_layer)):
                start = stack_a[layer + 1][0, -1].float()
                end = stack_b[layer + 1][0, -1].float()
                delta = end - start
                grid = alphas.to(model.device)[:, None]
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
                            model.forward(encodings["base"].input_ids.expand(chunk.shape[0], -1))
                    finally:
                        for handle in handles:
                            handle.remove()
                finals = torch.cat(captured)
                logits = model.unembed(finals)
                response = (logits[:, token_b] - logits[:, token_a]).cpu().numpy()
                unnormalised = (finals @ w_diff).cpu().numpy()

                # lenses are linear: one transported direction gives the slope
                probes = torch.stack([delta, start])
                j_web = jc.transport(model, layer, probes, web)
                j_domain = jc.transport(model, layer, probes, domain)
                tangent = float((unnormalised[1] - unnormalised[0]) / (alphas[1] - alphas[0]))
                chord = float(unnormalised[-1] - unnormalised[0])

                record["layers"][role] = {
                    "width": transition_width(alphas.tolist(), response.tolist()),
                    "tangent": tangent,
                    "chord": chord,
                    "gate_strength": abs(chord) / max(abs(tangent), 1e-9),
                    "logit_difference": [float(response[0]), float(response[-1])],
                    "slopes": {
                        "j_web": float(j_web[0] @ w_diff),
                        "j_domain": float(j_domain[0] @ w_diff),
                        "tuned_web": float((delta @ web_weight[layer].T) @ w_diff),
                        "tuned_domain": float((delta @ domain_weight[layer].T) @ w_diff),
                        "logit": float(delta @ w_diff),
                    },
                }
            work = record["layers"]["workspace"]
            ratio_web = work["slopes"]["tuned_web"] / work["slopes"]["j_web"]
            ratio_domain = work["slopes"]["tuned_domain"] / work["slopes"]["j_domain"]
            record["ratio_web"] = ratio_web
            record["ratio_domain"] = ratio_domain
            rows.append(record)
            print(f"  {name:22s} width {work['width']:.3f} (ctrl "
                  f"{record['layers']['control']['width']:.3f})  gate x"
                  f"{work['gate_strength']:.1f}   tuned/J web {ratio_web:.2f} -> "
                  f"domain {ratio_domain:.2f}", flush=True)

    summary = {}
    for group in ("numeric", "discrete", "graded"):
        members = [r for r in rows if r["group"] == group]
        if not members:
            continue
        summary[group] = {
            "n": len(members),
            "median_width": float(np.median([r["layers"]["workspace"]["width"] for r in members])),
            "median_control_width": float(np.median([r["layers"]["control"]["width"] for r in members])),
            "median_gate_strength": float(np.median([r["layers"]["workspace"]["gate_strength"] for r in members])),
            "median_ratio_web": float(np.median([r["ratio_web"] for r in members])),
            "median_ratio_domain": float(np.median([r["ratio_domain"] for r in members])),
        }
    print("\n" + json.dumps(summary, indent=1))
    (HERE / args.out).write_text(json.dumps(
        {"model": args.model, "layer": args.layer, "control_layer": args.control_layer,
         "summary": summary, "rows": rows, "skipped": skipped}, indent=1))
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
