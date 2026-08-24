"""Refit both lenses on an in-domain corpus and re-measure tangent vs chord.

CLAIMS.md Addendum 4. Notebook 01's toy separates the average tangent from the
regression chord because its fitting corpus is two clusters straddling the gate
*along the swept axis*. The deployed lenses are fitted on web text, which never
explores the spider-versus-dog direction, and there they do not separate.

This refits both at one layer on a corpus of two-hop animal-property prompts --
same estimators, same code, only the corpus changes -- and compares the
resulting slopes along the spider -> dog path against two properties of the
curve itself: its tangent at alpha = 0 and its chord over [0, 1].
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

# Animals spanning the leg-count axis, each with several indirect descriptions
# so the corpus varies over "which animal" the way the probed path does.
ANIMALS = {
    "spider": ("spins webs", "has eight legs and spins silk", "waits in a web for flies",
               "is feared by arachnophobes", "weaves a silken trap"),
    "ant": ("marches in colonies", "carries crumbs to its nest", "lives in an underground colony",
            "follows scent trails in a line", "is a tiny social insect"),
    "bee": ("buzzes and makes honey", "pollinates flowers and stings", "lives in a hive and makes honey",
            "dances to show where nectar is", "is kept by apiarists"),
    "dog": ("barks and fetches sticks", "wags its tail and guards the house", "is walked on a lead",
            "is called man's best friend", "chases the postman"),
    "cat": ("purrs and chases mice", "grooms itself and naps in the sun", "has whiskers and retractable claws",
            "is kept as a mouser", "lands on its feet"),
    "horse": ("is ridden with a saddle", "gallops and neighs", "pulls a cart and wears horseshoes",
              "races on a track", "is groomed with a curry comb"),
    "cow": ("chews cud and gives milk", "grazes in a field and moos", "is milked at dawn",
            "is kept in a dairy herd", "has a four-chambered stomach"),
    "bird": ("lays eggs and has feathers", "builds a nest and sings at dawn", "migrates south for winter",
             "perches on a branch and chirps", "preens its feathers"),
    "chicken": ("clucks and lays eggs in a coop", "is kept for eggs on a farm", "pecks at grain in the yard",
                "roosts at night in a hen house", "crows at sunrise"),
    "octopus": ("has eight arms and squirts ink", "changes colour on the seabed", "squeezes through tiny gaps",
                "is a clever cephalopod", "grips with suckers"),
    "beetle": ("has a hard shell and six legs", "scuttles under logs", "is an armoured insect",
               "burrows in rotting wood", "clicks its wing cases"),
    "sheep": ("gives us wool and bleats", "is herded by collies", "grazes on the hillside",
              "is shorn each spring", "flocks together"),
    "pig": ("roots in mud and oinks", "is kept in a sty", "snuffles for truffles",
            "is raised for bacon", "wallows to keep cool"),
    "frog": ("croaks by the pond", "hops and catches flies with its tongue", "starts life as a tadpole",
             "has damp skin and webbed feet", "leaps into the water"),
    "crab": ("scuttles sideways on the beach", "has pincers and a hard shell", "hides in rock pools",
             "moults its shell as it grows", "waves its claws"),
}
TEMPLATES = [
    "Fact: The number of legs on the animal that {d} is ",
    "Fact: The animal that {d} has this many legs: ",
    "Fact: Counting the legs of the creature that {d} gives ",
    "Fact: The creature that {d} walks on this many legs: ",
    "Fact: If you count the legs on the animal that {d}, you get ",
]


def build_corpus_text() -> list[str]:
    lines = []
    for animal, descriptions in ANIMALS.items():
        for description in descriptions:
            for template in TEMPLATES:
                lines.append(template.format(d=description))
    return lines


def chunk_corpus(tokenizer, lines, seq_len, batch_size, device, repeats=3):
    """Tokenize the prompt family into fixed-length sequences, like the web corpus."""
    generator = torch.Generator().manual_seed(0)
    ids: list[int] = []
    for _ in range(repeats):
        order = torch.randperm(len(lines), generator=generator).tolist()
        for index in order:
            ids.extend(tokenizer.encode(lines[index] + "\n", add_special_tokens=False))
    bos = tokenizer.bos_token_id
    usable = (len(ids) // (seq_len - 1)) * (seq_len - 1)
    rows = torch.tensor(ids[:usable]).view(-1, seq_len - 1)
    rows = torch.cat([torch.full((rows.shape[0], 1), bos), rows], dim=1)
    return list(rows.to(device).split(batch_size)), rows.shape[0]


def fit_ols_single_layer(model, batches, layer, ridge):
    """Closed-form `argmin ||h_final - (A h_l + b)||^2` on the given corpus."""
    sources, targets = [], []
    for input_ids in batches:
        positions = jc.valid_positions(input_ids.shape[1]).to(model.device)
        stack = jc.residual_stack(model, input_ids)[:, :, positions]
        sources.append(stack[layer + 1].flatten(0, 1).float())
        targets.append(stack[model.n_layers].flatten(0, 1).float())
    source = torch.cat(sources)
    target = torch.cat(targets)
    mean_x, mean_y = source.mean(0).double(), target.mean(0).double()
    centred_x = (source - mean_x.float()).double()
    centred_y = (target - mean_y.float()).double()
    cov_xx = centred_x.T @ centred_x / centred_x.shape[0]
    cov_yx = centred_y.T @ centred_x / centred_x.shape[0]
    scale = torch.diagonal(cov_xx).mean()
    weight = torch.linalg.solve(
        cov_xx + ridge * scale * torch.eye(model.d_model, dtype=torch.float64, device=model.device),
        cov_yx.T,
    ).T
    bias = mean_y - weight @ mean_x
    residual = centred_y - centred_x @ weight.T
    r2 = 1.0 - float(residual.pow(2).sum() / centred_y.pow(2).sum())
    return weight.float(), bias.float(), r2, source.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--web-sequences", type=int, default=64)
    parser.add_argument("--alpha-steps", type=int, default=81)
    parser.add_argument("--out", default="in_domain_lenses.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    layer = args.layer

    def single(word):
        for candidate in (word, " " + word):
            pieces = tokenizer.encode(candidate, add_special_tokens=False)
            if len(pieces) == 1:
                return pieces[0]
        raise ValueError(word)

    ids = {w: single(w) for w in (ANSWER, COUNTER_ANSWER, BRIDGE, COUNTER_BRIDGE)}
    gain = 1.0 + model.final_norm.weight.float()
    w_diff = (model.lm_head.weight[ids[COUNTER_ANSWER]].float()
              - model.lm_head.weight[ids[ANSWER]].float()) * gain

    enc_base = tokenizer(BASE, return_tensors="pt").to(model.device)
    enc_counter = tokenizer(COUNTER, return_tensors="pt").to(model.device)
    position = enc_base.input_ids.shape[1] - 1
    start = jc.residual_stack(model, enc_base.input_ids)[layer + 1][0, -1].float()
    end = jc.residual_stack(model, enc_counter.input_ids)[layer + 1][0, -1].float()
    delta = end - start

    alphas = torch.linspace(-0.15, 1.15, args.alpha_steps)
    grid = alphas.to(model.device)[:, None]
    path = (1 - grid) * start + grid * end
    zero = int(np.argmin(np.abs(alphas.numpy())))
    one = int(np.argmin(np.abs(alphas.numpy() - 1.0)))

    # ---- the model's real response -----------------------------------------
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
                model.forward(enc_base.input_ids.expand(chunk.shape[0], -1))
        finally:
            for handle in handles:
                handle.remove()
    actual = (torch.cat(captured) @ w_diff).cpu().numpy()

    # ---- tangent and chord of this curve ------------------------------------
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
            model.forward(enc_base.input_ids)
        finally:
            for handle in handles:
                handle.remove()
        return tangent_store["h"]

    with torch.no_grad():
        _, tangent_vector = torch.func.jvp(
            run_scaled, (torch.zeros((), device=model.device),),
            (torch.ones((), device=model.device),),
        )
    tangent_slope = float(tangent_vector[0] @ w_diff)
    chord_slope = float(actual[one] - actual[zero])
    print(f"tangent at alpha=0 : {tangent_slope:10.1f}")
    print(f"chord over [0,1]   : {chord_slope:10.1f}")

    # ---- the two corpora ----------------------------------------------------
    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    web = jc.corpus_batches(tokenizer, dataset["text"], n_sequences=args.web_sequences,
                            seq_len=args.seq_len, batch_size=2, seed=5)
    lines = build_corpus_text()
    domain, n_domain = chunk_corpus(tokenizer, lines, args.seq_len, 2, model.device)
    print(f"in-domain corpus: {len(lines)} prompts -> {n_domain} sequences of {args.seq_len}")

    results = {}
    for name, batches in (("web", web), ("in_domain", domain)):
        transported = jc.transport(model, layer, path, batches)
        j_line = (transported @ w_diff).cpu().numpy()
        weight, bias, r2, n_tokens = fit_ols_single_layer(model, batches, layer, args.ridge)
        tuned_line = ((path @ weight.T + bias) @ w_diff).cpu().numpy()

        span = float(alphas[-1] - alphas[0])
        results[name] = {
            "j_slope": float((j_line[-1] - j_line[0]) / span),
            "tuned_slope": float((tuned_line[-1] - tuned_line[0]) / span),
            "ols_r2": r2,
            "n_fit_tokens": n_tokens,
            "j_line": j_line.tolist(),
            "tuned_line": tuned_line.tolist(),
        }
        base_scores = {
            "j": model.unembed(jc.transport(model, layer, start, batches)[0]).squeeze(),
            "tuned": model.unembed(start @ weight.T + bias).squeeze(),
        }
        results[name]["ranks"] = {
            lens: {word: jc.rank_of(scores, ids[word]) for word in (BRIDGE, ANSWER)}
            for lens, scores in base_scores.items()
        }
        print(f"{name:10s} J slope {results[name]['j_slope']:9.1f}   "
              f"tuned slope {results[name]['tuned_slope']:9.1f}   "
              f"ratio {results[name]['tuned_slope'] / results[name]['j_slope']:5.2f}   "
              f"OLS R2 {r2:.3f}  ({n_tokens} tokens)", flush=True)
        print(f"           ranks {json.dumps(results[name]['ranks'])}", flush=True)

    logit_line = (path @ w_diff).cpu().numpy()

    payload = {
        "model": args.model, "layer": layer, "ridge": args.ridge,
        "alphas": alphas.tolist(), "actual": actual.tolist(),
        "logit_line": logit_line.tolist(),
        "tangent_slope": tangent_slope, "chord_slope": chord_slope,
        "n_domain_prompts": len(lines),
        **results,
    }
    (HERE / args.out).write_text(json.dumps(payload, indent=1))
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
