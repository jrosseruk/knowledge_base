"""The integrated lens: one knob that interpolates J-lens -> Tuned Lens.

A lens is a linear map `A` standing in for the remaining computation
`F_l : h -> h_final`. Different lenses are the best linear map *at a different
intervention scale*:

    A_{l,sigma} = argmin_A  E_{h, delta ~ q_sigma} || F_l(h + delta) - F_l(h) - A delta ||^2

* `sigma -> 0` recovers the J-lens, `E_h[J_l(h)]`;
* `delta = h' - h` for independent natural `h, h'` recovers the least-squares
  (tuned-lens-like) map `Cov(F(h), h) Cov(h)^{-1}`.

The implementation trick that makes the whole family as cheap as the J-lens.
By the fundamental theorem of calculus the finite effect is exactly
`F(h+delta) - F(h) = J_bar(h, delta) delta` with
`J_bar(h,delta) = int_0^1 J(h + alpha delta) d alpha`. For isotropic `delta`
independent of `h`, the least-squares solution of the objective above is the
expected integrated Jacobian, and

    E_{h,delta}[ J_bar(h,delta) ]  =  E_{h,delta,alpha}[ J(h + alpha delta) ]
                                   =  E_{h' ~ p_sigma}[ J(h') ]

where `p_sigma` is the activation distribution *smeared at scale sigma*. So the
scale-sigma lens is just the ordinary Jacobian-lens estimator run over
perturbed activations. No extra forward passes, no matched dataset, one extra
line in the injection hook -- and `sigma = 0` reduces to the existing estimator
exactly, which is the correctness check.

Perturbations are scaled relative to the corpus residual norm at that layer, so
`sigma` is dimensionless and comparable across layers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Callable, Sequence

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


def transport_at_scale(
    model: jc.LensModel,
    layer: int,
    directions: torch.Tensor,
    batches: Sequence[torch.Tensor],
    *,
    sigma: float,
    reference_norm: float,
    seed: int = 0,
    skip_first: int = jc.SKIP_FIRST_N_POSITIONS,
) -> torch.Tensor:
    """`A_{l,sigma} @ d` for each row `d` of `directions`.

    Identical to `jlens_core.transport` except that, before the Jacobian is
    taken, every valid source activation is displaced by an independent
    Gaussian of scale `sigma * reference_norm`. At `sigma = 0` this is exactly
    `jlens_core.transport`.
    """
    if directions.ndim == 1:
        directions = directions[None]
    directions = directions.to(model.device, torch.float32)
    n_directions = directions.shape[0]

    source_module = model.layers[layer]
    target_module = model.layers[-1]
    totals = torch.zeros(n_directions, model.d_model, dtype=torch.float64, device=model.device)
    n_sequences = 0
    generator = torch.Generator(device=model.device).manual_seed(seed)
    scale = sigma * reference_norm / (model.d_model ** 0.5)

    for input_ids in batches:
        n_batch, seq_len = input_ids.shape
        positions = jc.valid_positions(seq_len, skip_first).to(model.device)
        expanded = input_ids.repeat_interleave(n_directions, dim=0)
        pattern = directions.repeat(n_batch, 1)
        smear = (
            torch.randn(expanded.shape[0], len(positions), model.d_model,
                        generator=generator, device=model.device) * scale
            if sigma > 0 else None
        )

        def run(alpha: torch.Tensor) -> torch.Tensor:
            captured: dict[str, torch.Tensor] = {}

            def inject(_module, _args, module_output):
                hidden = jc._block_output(module_output)
                delta = torch.zeros_like(hidden)
                delta[:, positions] = (alpha * pattern).to(hidden.dtype)[:, None, :]
                if smear is not None:
                    delta[:, positions] += smear.to(hidden.dtype)
                return jc._rewrap(module_output, hidden + delta)

            def capture(_module, _args, module_output):
                captured["h"] = jc._block_output(module_output)

            handles = [
                source_module.register_forward_hook(inject),
                target_module.register_forward_hook(capture),
            ]
            try:
                model.forward(expanded)
            finally:
                for handle in handles:
                    handle.remove()
            return captured["h"][:, positions].float().sum(dim=1)

        zero = torch.zeros((), device=model.device, dtype=torch.float32)
        one = torch.ones((), device=model.device, dtype=torch.float32)
        with torch.no_grad():
            _, tangent = torch.func.jvp(run, (zero,), (one,))
        totals += tangent.double().view(n_batch, n_directions, model.d_model).sum(dim=0) / len(positions)
        n_sequences += n_batch

    return (totals / n_sequences).float()


def reference_norm(model, batches, layer) -> float:
    """Mean residual-stream norm at `layer` over the corpus, so sigma is dimensionless."""
    norms = []
    for input_ids in batches:
        positions = jc.valid_positions(input_ids.shape[1]).to(model.device)
        stack = jc.residual_stack(model, input_ids)[layer + 1][:, positions]
        norms.append(stack.float().norm(dim=-1).flatten())
    return float(torch.cat(norms).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--sigmas", type=float, nargs="*",
                        default=[0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5])
    parser.add_argument("--corpus-sequences", type=int, default=48)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--out", default="integrated_lens.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    layer = args.layer

    PROBES = [
        ("spider-legs", "Fact: The number of legs on the animal that spins webs is ",
         "spider", "8"),
        ("mars-colour", "Fact: The color of the planet fourth from the Sun is ",
         "Mars", "red"),
        ("basketball", "Fact: The number of players from one team on a basketball court is ",
         "basketball", "5"),
    ]

    def single(word):
        for candidate in (word, " " + word):
            pieces = tokenizer.encode(candidate, add_special_tokens=False)
            if len(pieces) == 1:
                return pieces[0]
        return None

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    corpus = jc.corpus_batches(tokenizer, dataset["text"], n_sequences=args.corpus_sequences,
                               seq_len=args.seq_len, batch_size=2, seed=5)
    norm = reference_norm(model, corpus, layer)
    print(f"mean residual norm at layer {layer}: {norm:.1f}")

    probes = []
    for name, prompt, bridge, answer in PROBES:
        bridge_id, answer_id = single(bridge), single(answer)
        if bridge_id is None or answer_id is None:
            print(f"skip {name}: multi-token"); continue
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        hidden = jc.residual_stack(model, enc.input_ids)[layer + 1][0, -1].float()
        probes.append({"name": name, "bridge": bridge, "answer": answer,
                       "bridge_id": bridge_id, "answer_id": answer_id, "hidden": hidden})
    stacked = torch.stack([p["hidden"] for p in probes])

    rows = []
    for sigma in args.sigmas:
        transported = transport_at_scale(model, layer, stacked, corpus,
                                         sigma=sigma, reference_norm=norm)
        entry = {"sigma": sigma, "probes": {}}
        for index, probe in enumerate(probes):
            scores = model.unembed(transported[index]).squeeze()
            entry["probes"][probe["name"]] = {
                "bridge_rank": jc.rank_of(scores, probe["bridge_id"]),
                "answer_rank": jc.rank_of(scores, probe["answer_id"]),
                "top": jc.top_tokens(tokenizer, scores, 6),
            }
        rows.append(entry)
        summary = "   ".join(
            f"{p['name']}: {p['bridge']} {entry['probes'][p['name']]['bridge_rank']:>6} / "
            f"{p['answer']} {entry['probes'][p['name']]['answer_rank']:>6}"
            for p in probes
        )
        print(f"sigma {sigma:5.2f}   {summary}", flush=True)
        print(f"              top@{sigma:.2f}: "
              f"{entry['probes'][probes[0]['name']]['top'][:5]}", flush=True)

    (HERE / args.out).write_text(json.dumps(
        {"model": args.model, "layer": layer, "reference_norm": norm,
         "probes": [{k: v for k, v in p.items() if k != "hidden"} for p in probes],
         "rows": rows}, indent=1))
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
