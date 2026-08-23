"""Is the transition sharper at the workspace layer than near the output?

CLAIMS.md Addendum 3. A within-item contrast, so it needs no absolute
threshold: for each two-hop prompt pair, interpolate the layer-`l` residual
between the two real internal states and measure how concentrated the resulting
change in `logit(B') - logit(B)` is, at a workspace layer and at a late control
layer. If gate-like circuits live in the workspace band, the workspace
transition should be the sharper of the two, item by item.

Sharpness is the 10%-to-90% transition width as a fraction of the sweep.
Linear-fit `R^2` is also recorded, but Addendum 3 explains why it is a poor
detector here and it is not used for the verdict.
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

PREFIX = "Fact: "
PAIRS = [
    ("legs-spider-bee", "The number of legs on the animal that spins webs is ", "8",
     "The number of legs on the animal that buzzes and makes honey is ", "6", "spider", "bee"),
    ("legs-spider-dog", "The number of legs on the animal that spins webs is ", "8",
     "The number of legs on the animal that barks and fetches sticks is ", "4", "spider", "dog"),
    ("legs-ant-bird", "The number of legs on the tiny insect that marches in colonies is ", "6",
     "The number of legs on the animal that lays eggs and has feathers is ", "2", "ant", "bird"),
    ("planet-color", "The color of the planet fourth from the Sun is ", "red",
     "The color of the planet third from the Sun is ", "blue", "Mars", "Earth"),
    ("capital-language", "The language spoken in the country whose capital is Paris is ", "French",
     "The language spoken in the country whose capital is Madrid is ", "Spanish", "France", "Spain"),
    ("capital-language-2", "The language spoken in the country whose capital is Rome is ", "Italian",
     "The language spoken in the country whose capital is Berlin is ", "German", "Italy", "Germany"),
    ("currency", "The currency used in the country shaped like a boot is the ", "euro",
     "The currency used in the country where the Shinkansen runs is the ", "yen", "Italy", "Japan"),
    ("continent", "The continent containing the country where the Eiffel Tower stands is ", "Europe",
     "The continent containing the country where the Pyramids of Giza stand is ", "Africa", "France", "Egypt"),
    ("ocean", "The ocean on the west coast of the country where Hollywood is is the ", "Pacific",
     "The ocean on the east coast of the country where Lisbon is is the ", "Atlantic", "USA", "Portugal"),
    ("cover-turtle", "The hard covering on the back of the slow reptile that hides its head is called a ", "shell",
     "The soft covering on the body of the animal that gives us wool is called ", "fleece", "turtle", "sheep"),
    ("sound-dog-cat", "The sound made by the animal that barks and fetches sticks is a ", "bark",
     "The sound made by the animal that purrs and chases mice is a ", "meow", "dog", "cat"),
    ("season", "The season that follows the coldest months of the year is ", "spring",
     "The season that follows the hottest months of the year is ", "autumn", "winter", "summer"),
    # Second round: items added to power the paired test. Chosen for simple
    # phrasing and single-token answers across *diverse* relations -- deliberately
    # not more counting items, since that is the family already known to show the
    # effect and stacking it would bias the test.
    ("wings-bird-insect", "The number of wings on a common housefly is ", "2",
     "The number of wings on a butterfly is ", "4", "fly", "butterfly"),
    ("horns", "The number of horns on a rhinoceros is ", "1",
     "The number of horns on a bull is ", "2", "rhino", "bull"),
    ("sides-triangle", "The number of sides on a triangle is ", "3",
     "The number of sides on a square is ", "4", "triangle", "square"),
    ("planet-moons", "The number of moons orbiting Mars is ", "2",
     "The number of moons orbiting Earth is ", "1", "Mars", "Earth"),
    ("bird-color-swan", "The color of a swan is ", "white",
     "The color of a raven is ", "black", "swan", "raven"),
    ("metal-color", "The color of pure gold is ", "yellow",
     "The color of pure copper is ", "orange", "gold", "copper"),
    ("fruit-color", "The color of a ripe banana is ", "yellow",
     "The color of a ripe lime is ", "green", "banana", "lime"),
    ("water-state", "At room temperature, the state of water is ", "liquid",
     "At room temperature, the state of iron is ", "solid", "water", "iron"),
    ("capital-france", "The capital city of France is ", "Paris",
     "The capital city of Japan is ", "Tokyo", "France", "Japan"),
]


def transition_width(alphas, response) -> float:
    low, high = response[0], response[-1]
    if abs(high - low) < 1e-6:
        return float("nan")
    normalised = [(value - low) / (high - low) for value in response]
    inside = [a for a, v in zip(alphas, normalised) if 0.1 <= v <= 0.9]
    return float((max(inside) - min(inside)) / (alphas[-1] - alphas[0])) if inside else float("nan")


def linear_r2(x, y) -> float:
    design = np.stack([x, np.ones_like(x)], 1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = y - y.mean()
    return float(1 - residual.dot(residual) / total.dot(total))


def run_path(model, input_ids, layer, position, states):
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--workspace-layer", type=int, default=34)
    parser.add_argument("--control-layer", type=int, default=46)
    parser.add_argument("--alpha-steps", type=int, default=61)
    parser.add_argument("--out", default="shape_paired_test.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    alphas = torch.linspace(0.0, 1.0, args.alpha_steps)

    def single(word):
        for candidate in (word, " " + word):
            ids = tokenizer.encode(candidate, add_special_tokens=False)
            if len(ids) == 1:
                return ids[0]
        return None

    rows, skipped = [], []
    for name, base, answer, counter, counter_answer, bridge, counter_bridge in PAIRS:
        token_b, token_b2 = single(answer), single(counter_answer)
        if token_b is None or token_b2 is None:
            skipped.append((name, "multi-token answer"))
            continue

        encodings, behaviour = {}, {}
        for role, text in (("base", PREFIX + base), ("counter", PREFIX + counter)):
            enc = tokenizer(text, return_tensors="pt").to(model.device)
            encodings[role] = enc
            with torch.inference_mode():
                generated = hf_model.generate(
                    **enc, max_new_tokens=6, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            behaviour[role] = tokenizer.decode(generated[0, enc.input_ids.shape[1]:])
        if not (behaviour["base"].strip().lower().startswith(answer.lower())
                and behaviour["counter"].strip().lower().startswith(counter_answer.lower())):
            skipped.append((name, f"behaviour {behaviour}"))
            print(f"skip {name}: {behaviour}", flush=True)
            continue

        stack_base = jc.residual_stack(model, encodings["base"].input_ids)
        stack_counter = jc.residual_stack(model, encodings["counter"].input_ids)
        position = encodings["base"].input_ids.shape[1] - 1

        row = {"name": name, "bridge": bridge, "counter_bridge": counter_bridge,
               "answer": answer, "counter_answer": counter_answer, "widths": {}, "r2": {}}
        for role, layer in (("workspace", args.workspace_layer), ("control", args.control_layer)):
            start = stack_base[layer + 1][0, -1].float()
            end = stack_counter[layer + 1][0, -1].float()
            grid = alphas.to(model.device)[:, None]
            path = (1 - grid) * start + grid * end
            finals = torch.cat([
                run_path(model, encodings["base"].input_ids, layer, position, chunk)
                for chunk in path.split(16)
            ])
            logits = model.unembed(finals)
            response = (logits[:, token_b2] - logits[:, token_b]).cpu().tolist()
            row["widths"][role] = transition_width(alphas.tolist(), response)
            row["r2"][role] = linear_r2(np.array(alphas.tolist()), np.array(response))
            row[f"response_{role}"] = response
        row["alphas"] = alphas.tolist()
        rows.append(row)
        print(f"{name:20s} width  workspace {row['widths']['workspace']:.3f}   "
              f"control {row['widths']['control']:.3f}", flush=True)

    valid = [r for r in rows if not np.isnan(r["widths"]["workspace"])
             and not np.isnan(r["widths"]["control"])]
    differences = [r["widths"]["workspace"] - r["widths"]["control"] for r in valid]
    sharper = sum(d < 0 for d in differences)
    summary = {
        "n_items": len(valid),
        "n_workspace_sharper": sharper,
        "median_difference": float(np.median(differences)) if differences else None,
        "median_workspace_width": float(np.median([r["widths"]["workspace"] for r in valid])) if valid else None,
        "median_control_width": float(np.median([r["widths"]["control"] for r in valid])) if valid else None,
        "fraction_workspace_below_0.5": (
            float(np.mean([r["widths"]["workspace"] < 0.5 for r in valid])) if valid else None
        ),
        "skipped": skipped,
    }
    print(json.dumps(summary, indent=1))
    (HERE / args.out).write_text(
        json.dumps({"model": args.model, "workspace_layer": args.workspace_layer,
                    "control_layer": args.control_layer, "summary": summary, "rows": rows}, indent=1)
    )
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
