"""Gate test with an effectiveness precondition and a stronger intervention.

CLAIMS.md Addendum 6. The previous survey scored transition widths on items
whose interventions never moved the output -- swings under 1 nat that never
change which answer the model prefers. A width computed on a flat response
measures noise, so those items are dropped here rather than interpreted.

Two changes:

* **Precondition.** An item is scored only if the intervention swings
  `logit(B') - logit(B)` by at least `--min-swing` nats *and* flips its sign.
* **Stronger intervention.** Pairs are **token-length matched** and differ in a
  single token, so the residual can be interpolated at *every* position rather
  than only the last. The endpoints are still two states the model really
  produces.

Everything else follows Addendum 2: `h(alpha) = (1-alpha) h_A + alpha h_B` at
one layer, with a late layer as the within-item control.
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

# (name, kind, prompt template with {}, filler A, answer A, filler B, answer B)
# The two fillers must tokenize to the same number of tokens; checked at runtime.
ITEMS = [
    # ---- numeric ---------------------------------------------------------
    ("legs-spider-ant", "numeric",
     "Fact: The number of legs on a {} is ", "spider", "8", "ant", "6"),
    ("legs-spider-dog", "numeric",
     "Fact: The number of legs on a {} is ", "spider", "8", "dog", "4"),
    ("sides-triangle-square", "numeric",
     "Fact: The number of sides on a {} is ", "triangle", "3", "square", "4"),
    ("sides-pentagon-hexagon", "numeric",
     "Fact: The number of sides on a {} is ", "pentagon", "5", "hexagon", "6"),
    ("players-basketball", "numeric",
     "Fact: The number of players on the field for one {} team is ",
     "basketball", "5", "volleyball", "6"),
    ("wheels", "numeric",
     "Fact: The number of wheels on a {} is ", "bicycle", "2", "car", "4"),

    # ---- discrete non-numeric --------------------------------------------
    ("antonym-up-wet", "discrete",
     "Fact: In one word, the opposite of {} is ", "up", "down", "wet", "dry"),
    ("antonym-hot-big", "discrete",
     "Fact: In one word, the opposite of {} is ", "hot", "cold", "big", "small"),
    ("antonym-fast-light", "discrete",
     "Fact: In one word, the opposite of {} is ", "fast", "slow", "light", "dark"),
    ("compass-north-east", "discrete",
     "Fact: In one word, the compass direction opposite to {} is ",
     "north", "south", "east", "west"),
    ("class-snake-salmon", "discrete",
     "Fact: In biology, a {} is classified as a ", "snake", "reptile", "salmon", "fish"),
    ("state-water-iron", "discrete",
     "Fact: At room temperature, {} is a ", "water", "liquid", "iron", "solid"),

    # ---- graded categorical ----------------------------------------------
    ("planet-colour", "graded",
     "Fact: The color of the planet {} from the Sun is ", "fourth", "red", "third", "blue"),
    ("fruit-colour", "graded",
     "Fact: The color of a ripe {} is ", "banana", "yellow", "lime", "green"),
    ("metal-colour", "graded",
     "Fact: The color of pure {} is ", "gold", "yellow", "copper", "orange"),
    ("bird-colour", "graded",
     "Fact: The color of a {} is ", "swan", "white", "raven", "black"),
    ("capital-language", "graded",
     "Fact: The main language spoken in {} is ", "France", "French", "Spain", "Spanish"),
    ("capital-language-2", "graded",
     "Fact: The main language spoken in {} is ", "Italy", "Italian", "Germany", "German"),
]


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
    parser.add_argument("--min-swing", type=float, default=4.0)
    parser.add_argument("--out", default="effective_gate_test.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    alphas = torch.linspace(0.0, 1.0, args.alpha_steps)
    gain = 1.0 + model.final_norm.weight.float()

    def single(word):
        for candidate in (word, " " + word):
            pieces = tokenizer.encode(candidate, add_special_tokens=False)
            if len(pieces) == 1:
                return pieces[0]
        return None

    rows, skipped = [], []
    for name, kind, template, filler_a, answer_a, filler_b, answer_b in ITEMS:
        token_a, token_b = single(answer_a), single(answer_b)
        if token_a is None or token_b is None:
            skipped.append((name, "multi-token answer")); continue

        texts = {"a": template.format(filler_a), "b": template.format(filler_b)}
        encodings = {k: tokenizer(v, return_tensors="pt").to(model.device) for k, v in texts.items()}
        if encodings["a"].input_ids.shape[1] != encodings["b"].input_ids.shape[1]:
            skipped.append((name, "prompts differ in token length")); continue

        behaviour = {}
        for key in ("a", "b"):
            with torch.inference_mode():
                generated = hf_model.generate(**encodings[key], max_new_tokens=6, do_sample=False,
                                              pad_token_id=tokenizer.eos_token_id)
            behaviour[key] = tokenizer.decode(generated[0, encodings[key].input_ids.shape[1]:])
        if not (behaviour["a"].strip().lower().startswith(answer_a.lower())
                and behaviour["b"].strip().lower().startswith(answer_b.lower())):
            skipped.append((name, f"behaviour {behaviour}"))
            print(f"skip {name:24s} behaviour {behaviour}", flush=True); continue

        stack_a = jc.residual_stack(model, encodings["a"].input_ids)
        stack_b = jc.residual_stack(model, encodings["b"].input_ids)
        seq_len = encodings["a"].input_ids.shape[1]
        position = seq_len - 1
        w_diff = (model.lm_head.weight[token_b].float()
                  - model.lm_head.weight[token_a].float()) * gain

        record = {"name": name, "kind": kind, "behaviour": behaviour,
                  "answer_a": answer_a, "answer_b": answer_b, "layers": {}}
        for role, layer in (("workspace", args.layer), ("control", args.control_layer)):
            start = stack_a[layer + 1][0].float()      # [seq, d] -- every position
            end = stack_b[layer + 1][0].float()
            captured: list[torch.Tensor] = []

            def grab(_m, _a, out):
                captured.append(jc._block_output(out)[:, position].float())

            for chunk in alphas.split(16):
                blend = chunk.to(model.device)[:, None, None]
                states = (1 - blend) * start[None] + blend * end[None]   # [n, seq, d]

                def inject(_m, _a, out, states=states):
                    hidden = jc._block_output(out)
                    return jc._rewrap(out, states.to(hidden.dtype))

                handles = [model.layers[layer].register_forward_hook(inject),
                           model.layers[-1].register_forward_hook(grab)]
                try:
                    with torch.inference_mode():
                        model.forward(encodings["a"].input_ids.expand(chunk.shape[0], -1))
                finally:
                    for handle in handles:
                        handle.remove()
            finals = torch.cat(captured)
            logits = model.unembed(finals)
            response = (logits[:, token_b] - logits[:, token_a]).cpu().numpy()
            unnormalised = (finals @ w_diff).cpu().numpy()
            tangent = float((unnormalised[1] - unnormalised[0]) / (alphas[1] - alphas[0]))
            record["layers"][role] = {
                "width": transition_width(alphas.tolist(), response.tolist()),
                "swing": float(abs(response[-1] - response[0])),
                "crosses_zero": bool(response[0] * response[-1] < 0),
                "endpoints": [float(response[0]), float(response[-1])],
                "gate_strength": abs(float(unnormalised[-1] - unnormalised[0])) / max(abs(tangent), 1e-9),
            }
            if role == "workspace":
                record["response"] = response.tolist()

        work = record["layers"]["workspace"]
        effective = work["swing"] >= args.min_swing and work["crosses_zero"]
        record["effective"] = effective
        if not effective:
            skipped.append((name, f"ineffective: swing {work['swing']:.2f}, "
                                  f"crosses {work['crosses_zero']}"))
            print(f"drop {name:24s} {kind:9s} swing {work['swing']:6.2f}  "
                  f"crosses {str(work['crosses_zero']):5s}  -> not scored", flush=True)
            continue
        rows.append(record)
        print(f"{name:24s} {kind:9s} swing {work['swing']:6.2f}  width "
              f"{work['width']:.3f} (ctrl {record['layers']['control']['width']:.3f})  "
              f"gate x{work['gate_strength']:.1f}", flush=True)

    summary = {}
    for kind in ("numeric", "discrete", "graded"):
        members = [r for r in rows if r["kind"] == kind]
        if members:
            summary[kind] = {
                "n": len(members),
                "median_width": float(np.median([r["layers"]["workspace"]["width"] for r in members])),
                "median_control_width": float(np.median([r["layers"]["control"]["width"] for r in members])),
                "median_swing": float(np.median([r["layers"]["workspace"]["swing"] for r in members])),
                "widths": {r["name"]: round(r["layers"]["workspace"]["width"], 3) for r in members},
            }
    print("\n" + json.dumps(summary, indent=1))
    print(f"\nscored {len(rows)} of {len(ITEMS)}; attrition:")
    for name, reason in skipped:
        print(f"  {name:24s} {reason}")
    (HERE / args.out).write_text(json.dumps(
        {"model": args.model, "layer": args.layer, "min_swing": args.min_swing,
         "intervention": "all positions, token-length-matched pairs",
         "summary": summary, "rows": rows, "skipped": skipped}, indent=1))
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
