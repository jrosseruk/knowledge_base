"""Claim 2, population test: is the J/Tuned divergence *caused* by nonlinearity?

The whole of Neel's explanation rests on one identity.  Write the remaining
computation as `F_l : h_l -> h_final`.  If `F_l` were globally affine,
`F_l(h) = M h + c`, then

  * its Jacobian is `M` everywhere, so the corpus-averaged Jacobian `J_l = M`;
  * least squares recovers it exactly, so the regression matrix `A_l = M`.

So `J_l = A_l` whenever `F_l` is affine, and **any** measured gap between the
two is a measurement of how nonlinear `F_l` is.  That makes the following a
direct test rather than an analogy:

    for each layer, measure  cos( J_l d , A_l d )  over probe directions d,
    against a noise ceiling from splitting the corpus in half.

Neel's hypothesis predicts that the layers where the two lenses *read out*
different things are the layers where the two *maps* disagree -- and that the
disagreement is real, i.e. far below the split-half ceiling.

Probe directions are drawn from the empirical distribution of centred
residuals at each layer, so the maps are compared where the data actually
lives rather than on isotropic noise.

Writes `nonlinearity_vs_divergence.json`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

import jlens_core as jc

hf_logging.set_verbosity_error()
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

HERE = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get("JLENS_ARTIFACTS", "/dev/shm/jlens-artifacts"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--corpus-sequences", type=int, default=64)
    parser.add_argument("--probe-sequences", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--probes", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--tuned", default="tuned_lens_ols.pt")
    parser.add_argument("--out", default="nonlinearity_vs_divergence.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    n_layers = model.n_layers

    tuned = torch.load(ARTIFACTS / args.tuned, map_location="cuda", weights_only=False)
    tuned_state = tuned["state"]
    tuned_form = tuned.get("form", "affine")
    if tuned_form != "affine":
        raise SystemExit(
            "this test compares the *least-squares* map A_l against J_l, because "
            "OLS is what provably equals the Jacobian when F_l is affine; pass "
            "--tuned tuned_lens_ols.pt"
        )

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    batches = jc.corpus_batches(
        tokenizer,
        dataset["text"],
        n_sequences=args.corpus_sequences,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=7,
    )
    half = len(batches) // 2
    first_half, second_half = batches[:half], batches[half:]

    # Probe directions: centred residual differences from held-out text, so the
    # maps are compared on the manifold the model actually visits.
    probe_batches = jc.corpus_batches(
        tokenizer,
        dataset["text"],
        n_sequences=args.probe_sequences,
        seq_len=args.seq_len,
        batch_size=8,
        seed=99,
    )
    probe_pool = []
    for input_ids in probe_batches:
        positions = jc.valid_positions(input_ids.shape[1]).to(model.device)
        probe_pool.append(
            jc.residual_stack(model, input_ids)[:, :, positions].flatten(1, 2).float()
        )
    probe_pool = torch.cat(probe_pool, dim=1)  # [n_layers + 1, n_tokens, d]

    generator = torch.Generator(device="cpu").manual_seed(3)
    left = torch.randint(0, probe_pool.shape[1], (args.probes,), generator=generator)
    right = torch.randint(0, probe_pool.shape[1], (args.probes,), generator=generator)

    rows = []
    for layer in range(n_layers):
        started = time.time()
        source = probe_pool[layer + 1]
        directions = (source[left.to(source.device)] - source[right.to(source.device)]).contiguous()
        directions = directions / directions.norm(dim=1, keepdim=True)

        # The estimator is a mean over sequences, so the full-corpus J is the
        # size-weighted mean of the two half-corpus estimates.  Computing it
        # that way makes the split-half noise ceiling free.
        j_first = jc.transport(model, layer, directions, first_half)
        j_second = jc.transport(model, layer, directions, second_half)
        n_first = sum(b.shape[0] for b in first_half)
        n_second = sum(b.shape[0] for b in second_half)
        j_full = (n_first * j_first + n_second * j_second) / (n_first + n_second)
        weight = tuned_state[layer]["A"].to(model.device, torch.float32)
        regressed = directions @ weight.T

        row = {
            "layer": layer,
            # the quantity of interest: do the two linear maps agree?
            "cos_J_vs_regression": float(F.cosine_similarity(j_full, regressed, dim=1).mean()),
            # noise ceiling: how well does J agree with an independent estimate
            # of itself on disjoint corpus halves?
            "cos_J_splithalf": float(F.cosine_similarity(j_first, j_second, dim=1).mean()),
            "cos_J_vs_identity": float(F.cosine_similarity(j_full, directions, dim=1).mean()),
            "cos_regression_vs_identity": float(F.cosine_similarity(regressed, directions, dim=1).mean()),
            "norm_ratio_J_over_regression": float(
                (j_full.norm(dim=1) / regressed.norm(dim=1)).mean()
            ),
            "seconds": round(time.time() - started, 1),
        }
        rows.append(row)
        print(
            f"L{layer:02d}  cos(J, A_reg) = {row['cos_J_vs_regression']:+.3f}   "
            f"split-half ceiling = {row['cos_J_splithalf']:+.3f}   "
            f"cos(J, I) = {row['cos_J_vs_identity']:+.3f}",
            flush=True,
        )

    (HERE / args.out).write_text(
        json.dumps(
            {
                "model": args.model,
                "corpus_sequences": args.corpus_sequences,
                "probes": args.probes,
                "tuned_lens": args.tuned,
                "note": "J_l == A_l identically whenever F_l is globally affine; "
                "the gap between cos_J_vs_regression and cos_J_splithalf is the "
                "measured departure from affineness.",
                "rows": rows,
            },
            indent=1,
        )
    )
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
