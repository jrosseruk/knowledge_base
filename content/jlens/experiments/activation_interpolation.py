"""Interpolate between two real internal states and watch what F_l does.

The preregistered lens-coordinate swap moved the activation by 0.4% of its norm
and produced a linear response -- which tests nothing, because a perturbation
that small cannot reach a gate. This is the adequately-scaled replacement,
preregistered in CLAIMS.md Addendum 2.

At one layer and one position, interpolate between the residual the model
actually produces on a prompt and the residual it actually produces on the
natural counterfactual:

    h(alpha) = (1 - alpha) h_P + alpha h_P',    alpha in [0, 1]

Every other position comes from `P`'s forward pass, so `F_l` is a genuine
function of the one vector being moved. Both endpoints are on-distribution.
This is a mid-network residual interpolation, *not* an input-embedding mixture.

Everything is read through one fixed linear functional so the model and the
three lenses are directly comparable and no per-lens scale can be tuned:

    s(x) = < w_B' - w_B , x >,     w_v = W_U[v] * (final-norm gain)

Writes `activation_interpolation.json` (+ figure via make_figures.py).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

import jlens_core as jc

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("JLENS_ARTIFACTS", "/dev/shm/jlens-artifacts"))

# Natural prompt pairs: same surface form, one word changed, different bridge
# entity, different answer. Both members must be answered correctly.
PAIRS = [
    {
        "name": "planet-color",
        "base": "Fact: The color of the planet fourth from the Sun is ",
        "counterfactual": "Fact: The color of the planet third from the Sun is ",
        "answer": "red",
        "counter_answer": "blue",
        "bridge": "Mars",
        "counter_bridge": "Earth",
    },
    {
        "name": "animal-legs",
        "base": "Fact: The number of legs on the animal that spins webs is ",
        "counterfactual": "Fact: The number of legs on the animal that buzzes and makes honey is ",
        "answer": "8",
        "counter_answer": "6",
        "bridge": "spider",
        "counter_bridge": "bee",
    },
    {
        "name": "instrument-count",
        "base": "Fact: The number of strings on the instrument played by Jimi Hendrix is ",
        "counterfactual": "Fact: The number of strings on the instrument played by Yo-Yo Ma is ",
        "answer": "6",
        "counter_answer": "4",
        "bridge": "guitar",
        "counter_bridge": "cello",
    },
    {
        "name": "capital-language",
        "base": "Fact: The language spoken in the country whose capital is Paris is ",
        "counterfactual": "Fact: The language spoken in the country whose capital is Madrid is ",
        "answer": "French",
        "counter_answer": "Spanish",
        "bridge": "France",
        "counter_bridge": "Spain",
    },
]


def norm_gain(model) -> torch.Tensor:
    weight = model.final_norm.weight.float()
    return 1.0 + weight if type(model.final_norm).__name__.startswith("Gemma") else weight


def run_path(model, input_ids, layer, position, states):
    """Forward with `h_{layer, position}` replaced by each row of `states`."""
    captured: dict[str, torch.Tensor] = {}

    def inject(_module, _args, module_output):
        hidden = jc._block_output(module_output)
        edited = hidden.clone()
        edited[:, position] = states.to(hidden.dtype)
        return jc._rewrap(module_output, edited)

    def capture(_module, _args, module_output):
        captured["h"] = jc._block_output(module_output)

    handles = [
        model.layers[layer].register_forward_hook(inject),
        model.layers[-1].register_forward_hook(capture),
    ]
    try:
        with torch.inference_mode():
            model.forward(input_ids.expand(states.shape[0], -1))
    finally:
        for handle in handles:
            handle.remove()
    return captured["h"][:, position].float()


def transition_width(alphas, response) -> float:
    """Fraction of the sweep spanned by the 10%-to-90% rise."""
    low, high = response[0], response[-1]
    if abs(high - low) < 1e-9:
        return float("nan")
    normalised = [(value - low) / (high - low) for value in response]
    inside = [a for a, value in zip(alphas, normalised) if 0.1 <= value <= 0.9]
    return float((max(inside) - min(inside)) / (alphas[-1] - alphas[0])) if inside else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layers", type=int, nargs="*", default=None)
    parser.add_argument("--control-layer", type=int, default=46)
    parser.add_argument("--alpha-steps", type=int, default=61)
    parser.add_argument("--corpus-sequences", type=int, default=32)
    parser.add_argument("--out", default="activation_interpolation.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    gain = norm_gain(model)

    lenses = {}
    for objective in ("kl", "ols"):
        path = ARTIFACTS / f"tuned_lens_{objective}.pt"
        if path.exists():
            blob = torch.load(path, map_location="cuda", weights_only=False)
            lenses[objective] = (
                blob["state"],
                blob.get("form", "residual" if objective == "kl" else "affine"),
            )

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    corpus = jc.corpus_batches(
        tokenizer, dataset["text"], n_sequences=args.corpus_sequences,
        seq_len=128, batch_size=2, seed=5,
    )

    screen_path = HERE / "multihop_three_lens_screen.json"
    default_layers = [34]
    if screen_path.exists():
        screen = json.loads(screen_path.read_text())
        by_name = {e["name"]: e["layer"] for e in screen["aligned"]}
        default_layers = [by_name.get("mars-color", 34)]

    alphas = torch.linspace(0.0, 1.0, args.alpha_steps)
    results = []

    for pair in PAIRS:
        tokens = {}
        ok = True
        for key in ("answer", "counter_answer"):
            ids = tokenizer.encode(" " + pair[key], add_special_tokens=False)
            bare = tokenizer.encode(pair[key], add_special_tokens=False)
            if len(bare) == 1:
                tokens[key] = bare[0]
            elif len(ids) == 1:
                tokens[key] = ids[0]
            else:
                ok = False
        if not ok:
            print(f"skip {pair['name']}: answers are not single tokens")
            continue

        encoded = {}
        behaviour = {}
        for role, key in (("base", "base"), ("counterfactual", "counterfactual")):
            enc = tokenizer(pair[key], return_tensors="pt").to(model.device)
            encoded[role] = enc
            with torch.inference_mode():
                generated = hf_model.generate(
                    **enc, max_new_tokens=6, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            behaviour[role] = tokenizer.decode(generated[0, enc.input_ids.shape[1]:])
        expected = (pair["answer"], pair["counter_answer"])
        correct = all(
            behaviour[role].strip().lower().startswith(word.lower())
            for role, word in zip(("base", "counterfactual"), expected)
        )
        print(f"{pair['name']}: base -> {behaviour['base']!r}, "
              f"counterfactual -> {behaviour['counterfactual']!r}, correct={correct}", flush=True)
        if not correct:
            results.append({"pair": pair, "behaviourally_correct": False, "behaviour": behaviour})
            continue
        if encoded["base"].input_ids.shape[1] != encoded["counterfactual"].input_ids.shape[1]:
            print(f"  note: prompts differ in length; interpolating final-position residuals anyway")

        stack_base = jc.residual_stack(model, encoded["base"].input_ids)
        stack_counter = jc.residual_stack(model, encoded["counterfactual"].input_ids)

        w_diff = (
            model.lm_head.weight[tokens["counter_answer"]].float()
            - model.lm_head.weight[tokens["answer"]].float()
        ) * gain

        layers = args.layers or (default_layers + [args.control_layer])
        for layer in layers:
            position = encoded["base"].input_ids.shape[1] - 1
            start = stack_base[layer + 1][0, -1].float()
            end = stack_counter[layer + 1][0, -1].float()
            path = (1 - alphas.to(model.device))[:, None] * start + alphas.to(model.device)[:, None] * end

            finals = []
            for chunk in path.split(16):
                finals.append(run_path(model, encoded["base"].input_ids, layer, position, chunk))
            finals = torch.cat(finals)
            actual = (finals @ w_diff).cpu()
            final_logits = model.unembed(finals)
            true_difference = (
                final_logits[:, tokens["counter_answer"]] - final_logits[:, tokens["answer"]]
            ).cpu()
            top_tokens = [tokenizer.decode([int(i)]) for i in final_logits.argmax(-1).tolist()]

            transported = jc.transport(model, layer, path, corpus)
            lines = {"jlens": (transported @ w_diff).cpu(), "logit": (path @ w_diff).cpu()}
            for objective, (state, form) in lenses.items():
                weight = state[layer]["A"].to(model.device, torch.float32)
                bias = state[layer]["b"].to(model.device, torch.float32)
                translated = path @ weight.T + bias
                if form == "residual":
                    translated = translated + path
                lines[f"tuned_{objective}"] = (translated @ w_diff).cpu()

            results.append({
                "pair": pair,
                "behaviourally_correct": True,
                "behaviour": behaviour,
                "layer": layer,
                "position": position,
                "alphas": alphas.tolist(),
                "actual": actual.tolist(),
                "true_logit_difference": true_difference.tolist(),
                "top_token": top_tokens,
                "transition_width": transition_width(alphas.tolist(), true_difference.tolist()),
                "delta_over_norm": float((end - start).norm() / start.norm()),
                **{k: v.tolist() for k, v in lines.items()},
            })
            flipped = [a for a, t in zip(alphas.tolist(), top_tokens)
                       if t.strip().lower() == pair["counter_answer"].lower()]
            print(
                f"  L{layer}: |delta|/|h| = {results[-1]['delta_over_norm']:.3f}  "
                f"logit diff {true_difference[0]:+.2f} -> {true_difference[-1]:+.2f}  "
                f"flip at alpha = {min(flipped) if flipped else None}  "
                f"10-90 width = {results[-1]['transition_width']:.3f}",
                flush=True,
            )

    (HERE / args.out).write_text(json.dumps({"model": args.model, "results": results}, indent=1))
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
