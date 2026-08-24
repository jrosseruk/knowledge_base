"""Does the *answer type* decide whether the computation gates?

The survey found numeric answers gating and non-numeric ones not, but each
family also differed in entity, phrasing and relation, so answer type was
confounded with everything else.

This isolates it. Hold the **bridge entity pair fixed** and vary only the
property asked about. `Mars` versus `Earth` is the same two-hop retrieval in
every case; only the answer changes from a colour to a count:

    "The color of the planet fourth from the Sun is "            -> red / blue
    "The number of moons orbiting the planet fourth ... is "     -> 2 / 1
    "The number of letters in the name of the planet fourth ..." -> 4 / 5

If the count versions gate and the colour version does not, answer type is
doing the work, because nothing else differs.

Part B adds few-shot framing to the discrete non-numeric families, which the
survey lost to Gemma answering in multiple-choice format, with an emoji, or by
echoing the entity.

Width only, so this is cheap: no lens fitting.
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

# (name, group, kind, base prompt, answer, counter prompt, counter answer)
ITEMS = [
    # ---- Part A: one bridge pair, three different properties ---------------
    ("planet-colour", "same-bridge", "graded",
     "Fact: The color of the planet fourth from the Sun is ", "red",
     "Fact: The color of the planet third from the Sun is ", "blue"),
    ("planet-moons", "same-bridge", "numeric",
     "Fact: The number of moons orbiting the planet fourth from the Sun is ", "2",
     "Fact: The number of moons orbiting the planet third from the Sun is ", "1"),
    ("planet-letters", "same-bridge", "numeric",
     "Fact: The number of letters in the name of the planet fourth from the Sun is ", "4",
     "Fact: The number of letters in the name of the planet third from the Sun is ", "5"),
    # same trick on the animal axis
    ("animal-legs", "same-bridge", "numeric",
     "Fact: The number of legs on the animal that spins webs is ", "8",
     "Fact: The number of legs on the animal that barks and fetches sticks is ", "4"),
    ("animal-letters", "same-bridge", "numeric",
     "Fact: The number of letters in the name of the animal that spins webs is ", "6",
     "Fact: The number of letters in the name of the animal that barks and fetches sticks is ", "3"),

    # ---- Part B: discrete non-numeric, few-shot framed ---------------------
    ("parity-fs", "discrete", "discrete",
     "Two is even. Five is odd. Ten is even. Seven is ", "odd",
     "Two is even. Five is odd. Ten is even. Eight is ", "even"),
    ("state-fs", "discrete", "discrete",
     "Iron is solid. Oxygen is gas. Mercury is liquid. Water is ", "liquid",
     "Iron is solid. Oxygen is gas. Mercury is liquid. Helium is ", "gas"),
    ("class-fs", "discrete", "discrete",
     "A dog is a mammal. A trout is a fish. An eagle is a bird. A snake is a ", "reptile",
     "A dog is a mammal. A trout is a fish. An eagle is a bird. A salmon is a ", "fish"),
    ("class-twohop-fs", "discrete", "discrete",
     "The animal that barks is a mammal. The animal with gills is a fish. "
     "The animal that slithers is a ", "reptile",
     "The animal that barks is a mammal. The animal with gills is a fish. "
     "The animal that flies and has feathers is a ", "bird"),
    ("antonym-fs", "discrete", "discrete",
     "The opposite of big is small. The opposite of fast is slow. The opposite of hot is ", "cold",
     "The opposite of big is small. The opposite of fast is slow. The opposite of wet is ", "dry"),
    ("direction-fs", "discrete", "discrete",
     "The opposite of left is right. The opposite of up is down. The opposite of north is ", "south",
     "The opposite of left is right. The opposite of up is down. The opposite of east is ", "west"),

    # ---- extra numeric, few-shot framed, for a matched comparison ----------
    ("legs-fs", "numeric-fs", "numeric",
     "A dog has 4 legs. A bird has 2 legs. A spider has ", "8",
     "A dog has 4 legs. A bird has 2 legs. An ant has ", "6"),
    ("sides-fs", "numeric-fs", "numeric",
     "A triangle has 3 sides. A hexagon has 6 sides. A square has ", "4",
     "A triangle has 3 sides. A hexagon has 6 sides. A pentagon has ", "5"),
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
    parser.add_argument("--out", default="controlled_gate_test.json")
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
    for name, group, kind, base, answer, counter, counter_answer in ITEMS:
        token_a, token_b = single(answer), single(counter_answer)
        if token_a is None or token_b is None:
            skipped.append((name, "multi-token")); continue

        encodings, behaviour = {}, {}
        for role, text in (("base", base), ("counter", counter)):
            enc = tokenizer(text, return_tensors="pt").to(model.device)
            encodings[role] = enc
            with torch.inference_mode():
                generated = hf_model.generate(**enc, max_new_tokens=6, do_sample=False,
                                              pad_token_id=tokenizer.eos_token_id)
            behaviour[role] = tokenizer.decode(generated[0, enc.input_ids.shape[1]:])
        if not (behaviour["base"].strip().lower().startswith(answer.lower())
                and behaviour["counter"].strip().lower().startswith(counter_answer.lower())):
            skipped.append((name, f"behaviour {behaviour}"))
            print(f"skip {name:18s} {behaviour}", flush=True); continue

        stack_a = jc.residual_stack(model, encodings["base"].input_ids)
        stack_b = jc.residual_stack(model, encodings["counter"].input_ids)
        position = encodings["base"].input_ids.shape[1] - 1
        record = {"name": name, "group": group, "kind": kind,
                  "answer": answer, "counter_answer": counter_answer,
                  "behaviour": behaviour, "widths": {}}

        for role, layer in (("workspace", args.layer), ("control", args.control_layer)):
            start = stack_a[layer + 1][0, -1].float()
            end = stack_b[layer + 1][0, -1].float()
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
            unnormalised = (finals @ ((model.lm_head.weight[token_b].float()
                                       - model.lm_head.weight[token_a].float()) * gain)).cpu().numpy()
            tangent = float((unnormalised[1] - unnormalised[0]) / (alphas[1] - alphas[0]))
            record["widths"][role] = transition_width(alphas.tolist(), response.tolist())
            if role == "workspace":
                record["gate_strength"] = abs(float(unnormalised[-1] - unnormalised[0])) / max(abs(tangent), 1e-9)
                record["response"] = response.tolist()
        rows.append(record)
        print(f"{name:18s} {kind:8s} width {record['widths']['workspace']:.3f} "
              f"(ctrl {record['widths']['control']:.3f})  gate x{record['gate_strength']:.1f}",
              flush=True)

    summary = {}
    for kind in ("numeric", "discrete", "graded"):
        members = [r for r in rows if r["kind"] == kind]
        if members:
            summary[kind] = {
                "n": len(members),
                "median_width": float(np.median([r["widths"]["workspace"] for r in members])),
                "widths": {r["name"]: round(r["widths"]["workspace"], 3) for r in members},
            }
    same_bridge = [r for r in rows if r["group"] == "same-bridge"]
    summary["same_bridge_control"] = {
        r["name"]: {"kind": r["kind"], "width": round(r["widths"]["workspace"], 3)}
        for r in same_bridge
    }
    print("\n" + json.dumps(summary, indent=1))
    (HERE / args.out).write_text(json.dumps(
        {"model": args.model, "layer": args.layer, "summary": summary,
         "rows": rows, "skipped": skipped}, indent=1))
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
