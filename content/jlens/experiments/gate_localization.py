"""Where is the gate?

We have shown that counting relations respond to an interpolation with a
sigmoid. This asks which part of the network produces it.

In a Gemma 3 block the residual increments are exactly the two post-norm
outputs, so the readout decomposes *exactly*:

    s(alpha) = <w, h_L(alpha)> + sum_{l>=L} ( <w, attn_l(alpha)> + <w, mlp_l(alpha)> )

with `w` the unnormalised `logit(B') - logit(B)` direction. Every term is a
scalar function of `alpha`, so the sigmoid in the total must come from
identifiable terms. For each component we report

* **swing** -- how much of the total change it contributes, and
* **nonlinearity** -- `1 - R^2` of a straight-line fit to its own curve,
  which is what distinguishes a gate from a component that merely scales.

Stage two drills into the most nonlinear MLP: which individual neurons switch?
A gate implemented by a few neurons crossing threshold should show up as a
small number of units whose activation is sharply sigmoidal in `alpha`, and
whose output weights write the answer token.

This is descriptive localisation, not a hypothesis test, and is reported as
such.
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

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

import jlens_core as jc

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent

PROBES = [
    ("legs-spider-dog", True, "Fact: The number of legs on a {} is ", "spider", "8", "dog", "4"),
    ("sides-triangle-square", True, "Fact: The number of sides on a {} is ",
     "triangle", "3", "square", "4"),
    ("players-basketball", True,
     "Fact: The number of players on the field for one {} team is ",
     "basketball", "5", "volleyball", "6"),
    ("capital-language", False, "Fact: The main language spoken in {} is ",
     "France", "French", "Spain", "Spanish"),
]


def nonlinearity(alphas: np.ndarray, curve: np.ndarray) -> float:
    """`1 - R^2` of a straight-line fit; 0 for a perfectly linear component."""
    if curve.std() < 1e-6:
        return 0.0
    design = np.stack([alphas, np.ones_like(alphas)], 1)
    coefficients, *_ = np.linalg.lstsq(design, curve, rcond=None)
    residual = curve - design @ coefficients
    return float(residual.dot(residual) / ((curve - curve.mean()) ** 2).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--alpha-steps", type=int, default=61)
    parser.add_argument("--top-neurons", type=int, default=8)
    parser.add_argument("--out", default="gate_localization.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    layer, device = args.layer, model.device
    gain = 1.0 + model.final_norm.weight.float()
    alphas = torch.linspace(0, 1, args.alpha_steps)
    alphas_np = alphas.numpy()

    def single(word):
        for candidate in (word, " " + word):
            pieces = tokenizer.encode(candidate, add_special_tokens=False)
            if len(pieces) == 1:
                return pieces[0]
        return None

    results = []
    for name, gates, template, filler_a, answer_a, filler_b, answer_b in PROBES:
        token_a, token_b = single(answer_a), single(answer_b)
        enc_a = tokenizer(template.format(filler_a), return_tensors="pt").to(device)
        enc_b = tokenizer(template.format(filler_b), return_tensors="pt").to(device)
        if enc_a.input_ids.shape[1] != enc_b.input_ids.shape[1]:
            print(f"skip {name}: length mismatch"); continue
        position = enc_a.input_ids.shape[1] - 1
        w_diff = (model.lm_head.weight[token_b].float()
                  - model.lm_head.weight[token_a].float()) * gain

        start = jc.residual_stack(model, enc_a.input_ids)[layer + 1][0].float()
        end = jc.residual_stack(model, enc_b.input_ids)[layer + 1][0].float()

        # capture every residual write downstream of the interpolated layer
        captures: dict[str, list[torch.Tensor]] = {}

        def make_hook(key):
            def hook(_m, _a, out):
                tensor = jc._block_output(out)
                captures.setdefault(key, []).append(tensor[:, position].float())
            return hook

        contributions: dict[str, np.ndarray] = {}
        neuron_traces = {}
        for chunk in alphas.split(16):
            captures.clear()
            blend = chunk.to(device)[:, None, None]
            states = (1 - blend) * start[None] + blend * end[None]

            def inject(_m, _a, out, states=states):
                hidden = jc._block_output(out)
                return jc._rewrap(out, states.to(hidden.dtype))

            handles = [model.layers[layer].register_forward_hook(inject)]
            for downstream in range(layer + 1, model.n_layers):
                block = model.layers[downstream]
                handles.append(block.post_attention_layernorm.register_forward_hook(
                    make_hook(f"attn{downstream}")))
                handles.append(block.post_feedforward_layernorm.register_forward_hook(
                    make_hook(f"mlp{downstream}")))
                # neuron activations entering the MLP down-projection
                handles.append(block.mlp.down_proj.register_forward_pre_hook(
                    lambda _m, inputs, key=f"neurons{downstream}":
                        captures.setdefault(key, []).append(inputs[0][:, position].float())))
            handles.append(model.layers[layer].register_forward_hook(
                make_hook("base")))
            try:
                with torch.inference_mode():
                    model.forward(enc_a.input_ids.expand(chunk.shape[0], -1))
            finally:
                for handle in handles:
                    handle.remove()
            for key, pieces in captures.items():
                value = torch.cat(pieces)
                if key.startswith("neurons"):
                    neuron_traces.setdefault(key, []).append(value.cpu())
                else:
                    projected = (value @ w_diff).cpu().numpy()
                    contributions[key] = np.concatenate(
                        [contributions.get(key, np.array([])), projected])

        total = sum(v for k, v in contributions.items() if k != "base") + contributions["base"]
        rows = []
        for key, curve in contributions.items():
            if key == "base":
                continue
            rows.append({
                "component": key,
                "swing": float(curve[-1] - curve[0]),
                "nonlinearity": nonlinearity(alphas_np, curve),
                "curve": curve.tolist(),
            })
        rows.sort(key=lambda r: -abs(r["swing"]))
        record = {"name": name, "gates": gates, "total": total.tolist(),
                  "base": contributions["base"].tolist(), "components": rows}

        print(f"\n=== {name} (gates={gates}) total swing "
              f"{total[-1] - total[0]:+.0f} ===", flush=True)
        print(f"{'component':12s}{'swing':>10}{'nonlinearity':>14}")
        for row in rows[:6]:
            print(f"{row['component']:12s}{row['swing']:>10.0f}{row['nonlinearity']:>14.3f}",
                  flush=True)

        # ---- stage two: neurons inside the most nonlinear big MLP ----------
        mlp_rows = [r for r in rows if r["component"].startswith("mlp")
                    and abs(r["swing"]) > 0.05 * abs(total[-1] - total[0])]
        if mlp_rows and gates:
            target = max(mlp_rows, key=lambda r: r["nonlinearity"])
            index = int(target["component"][3:])
            trace = torch.cat(neuron_traces[f"neurons{index}"])          # [n_alpha, d_ff]
            down = model.layers[index].mlp.down_proj.weight.float()       # [d, d_ff]
            write = (w_diff @ down)                                       # [d_ff]
            per_neuron = trace.to(device) * write[None, :]
            swing = (per_neuron[-1] - per_neuron[0]).abs()
            top = torch.topk(swing, args.top_neurons).indices.tolist()
            described = []
            for neuron in top:
                curve = per_neuron[:, neuron].cpu().numpy()
                described.append({
                    "neuron": neuron,
                    "swing": float(curve[-1] - curve[0]),
                    "nonlinearity": nonlinearity(alphas_np, curve),
                    "writes": jc.top_tokens(tokenizer, model.unembed(down[:, neuron]).squeeze(), 5),
                })
            record["gate_mlp"] = {"layer": index, "neurons": described,
                                  "nonlinearity": target["nonlinearity"]}
            print(f"  most nonlinear MLP: layer {index} "
                  f"(nonlinearity {target['nonlinearity']:.3f}); top neurons:")
            for entry in described[:5]:
                print(f"    n{entry['neuron']:<6} swing {entry['swing']:>8.0f}  "
                      f"nonlin {entry['nonlinearity']:.3f}  writes {entry['writes'][:3]}",
                      flush=True)
        results.append(record)

    (HERE / args.out).write_text(json.dumps(
        {"model": args.model, "layer": layer, "results": results}, indent=1))
    print("\nwrote", HERE / args.out)


if __name__ == "__main__":
    main()
