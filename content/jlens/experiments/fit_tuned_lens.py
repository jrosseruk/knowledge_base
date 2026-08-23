"""Fit Tuned Lenses at every layer of Gemma 3 12B IT.

Two objectives are fitted, because Claim 1 and Claim 2 name different objects.

`kl` -- the **published** Tuned Lens (Belrose et al. 2023, as implemented in
    AlignmentResearch/tuned-lens).  The translator is affine with an identity
    residual, `h -> h + W h + b`, with `W` and `b` zero-initialised, and the
    readout is `W_U norm(h + W h + b)` using the model's own final norm.  It is
    trained by distillation: minimise the forward KL from the model's true
    final-layer distribution to the lens distribution.  This is the lens the
    J-lens paper compares against in Figure 51, so it is what Claim 1 needs.

`ols` -- least squares between residual streams,
    `argmin_{A,b} E || h_final - (A h_l + b) ||^2`.
    This is the object Neel's review names when it calls the Tuned Lens
    "linear regression between residual streams", and it is what CLAIMS.md
    preregisters for Claim 2.  Being ordinary least squares it has a closed
    form, so we accumulate second moments and solve the normal equations --
    no optimiser, no learning rate, nothing to tune but the ridge term.

Both see the same corpus and the same position convention as the Jacobian
estimator (`skip_first = 16`, final position dropped), so the three lenses are
fitted on matched data.

Writes `tuned_lens_<objective>.pt` and prints held-out KL against the model.
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
torch.backends.cuda.matmul.allow_tf32 = False
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OUT_DIR = Path(__file__).resolve().parent
# The translators are 48 x 3840 x 3840 each; they are working artefacts, not
# repository content, so they live on the RAM disk and only the held-out
# evaluation JSON is written next to the code.
ARTIFACTS = Path(os.environ.get("JLENS_ARTIFACTS", "/dev/shm/jlens-artifacts"))
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def cache_residuals(hf_model, model, batches, keep_final_logits=False):
    """Run the corpus and stack residuals at the valid positions.

    Returns `[n_layers + 1, n_tokens, d_model]` in bfloat16 on GPU.
    """
    chunks = []
    for input_ids in batches:
        positions = jc.valid_positions(input_ids.shape[1]).to(model.device)
        stacked = jc.residual_stack(model, input_ids)[:, :, positions]
        chunks.append(stacked.flatten(1, 2).to(torch.bfloat16).clone())
    return torch.cat(chunks, dim=1)


def fit_ols(cache, n_layers, ridge):
    """Closed-form `argmin_{A,b} E ||y - (A x + b)||^2` per layer."""
    target = cache[n_layers].float()
    mean_y = target.mean(dim=0).double()
    device = cache.device
    d_model = cache.shape[-1]
    identity = torch.eye(d_model, dtype=torch.float64, device=device)

    state = {}
    for layer in range(n_layers):
        source = cache[layer + 1].float()
        mean_x = source.mean(dim=0).double()
        centered_x = (source - mean_x.float()).double()
        centered_y = (target - mean_y.float()).double()
        cov_xx = centered_x.T @ centered_x / centered_x.shape[0]
        cov_yx = centered_y.T @ centered_x / centered_x.shape[0]
        scale = float(torch.diagonal(cov_xx).mean())
        weight = torch.linalg.solve(cov_xx + ridge * scale * identity, cov_yx.T).T
        bias = mean_y - weight @ mean_x
        state[layer] = {"A": weight.half().cpu(), "b": bias.float().cpu()}
        residual = centered_y - centered_x @ weight.T
        r2 = 1.0 - float(residual.pow(2).sum() / centered_y.pow(2).sum())
        print(f"  ols layer {layer:2d}: R^2 = {r2:.4f}", flush=True)
    return state


def fit_kl(model, cache, n_layers, unembed, *, steps, batch_tokens, lr, weight_decay, seed,
           layers=None):
    """Distil the model's final distribution into a per-layer affine translator.

    Two details matter for correctness at the deep layers, where the right
    answer is `W = 0`:

    * the readout runs in float32.  With a bfloat16 unembedding the KL at the
      last layer is not exactly zero but a few units in the last place, and
      Adam -- being scale invariant -- happily walks a translator away from
      zero on that rounding noise alone.
    * AdamW with the published weight decay pulls the translator back toward
      the identity, which is what the residual parameterisation is for.
    """
    # `layers` lets a notebook fit only the depths it displays; the full sweep
    # fits every layer *except* the last, whose translator is known in closed
    # form (its source is its target, so W = 0, b = 0).  Fitting it is not
    # merely wasteful: the true gradient there is exactly zero, float32
    # non-determinism between the no-grad target path and the autograd
    # prediction path leaves a few-ULP residue, and Adam -- being scale
    # invariant -- promotes that rounding noise to full-size updates.
    # See LOG.md.
    layers = list(range(n_layers - 1)) if layers is None else list(layers)
    d_model = cache.shape[-1]
    device = cache.device
    weight = torch.zeros(len(layers), d_model, d_model, device=device, requires_grad=True)
    bias = torch.zeros(len(layers), d_model, device=device, requires_grad=True)
    optimizer = torch.optim.AdamW([weight, bias], lr=lr, weight_decay=weight_decay)
    schedule = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.05, total_iters=steps
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_tokens = cache.shape[1]

    started = time.time()
    for step in range(steps):
        index = torch.randint(0, n_tokens, (batch_tokens,), generator=generator).to(device)
        with torch.no_grad():
            target = F.log_softmax(unembed(cache[n_layers, index].float()), dim=-1)
            probabilities = target.exp()
        optimizer.zero_grad(set_to_none=True)
        # One backward per layer: the translators are independent, so this
        # keeps only a single [batch, vocab] logit graph alive at a time.
        total = 0.0
        for slot, layer in enumerate(layers):
            source = cache[layer + 1, index].float()
            translated = source + source @ weight[slot].T + bias[slot]
            logits = unembed(translated)
            kl = (probabilities * (target - F.log_softmax(logits, dim=-1))).sum(-1).mean()
            kl.backward()
            total += kl.item()
        optimizer.step()
        schedule.step()
        if step % 25 == 0 or step == steps - 1:
            # The last block's translator has an exactly-known answer (W = 0,
            # since its source *is* the target), so its norm is a live check
            # that nothing is driving the optimiser off that fixed point.
            print(
                f"  kl step {step:4d}/{steps}  mean KL = "
                f"{total / len(layers):.4f}  |W[L{layers[-1]}]| = {weight[-1].norm().item():.3f}"
                f"  ({time.time() - started:.0f}s)",
                flush=True,
            )

    state = {
        layer: {"A": weight[slot].detach().half().cpu(), "b": bias[slot].detach().float().cpu()}
        for slot, layer in enumerate(layers)
    }
    if n_layers - 1 not in state:
        state[n_layers - 1] = {
            "A": torch.zeros(d_model, d_model, dtype=torch.half),
            "b": torch.zeros(d_model),
        }
    return state


def apply_lens(source, entry, form):
    """`A x + b` for the affine (OLS) form, `x + W x + b` for the residual form."""
    weight = entry["A"].to(source.device, torch.float32)
    bias = entry["b"].to(source.device, torch.float32)
    translated = source @ weight.T + bias
    return translated + source if form == "residual" else translated


def evaluate(model, cache, state, n_layers, unembed, form):
    """Held-out KL(model || lens) per layer, for the tuned and logit lenses."""
    target = F.log_softmax(unembed(cache[n_layers].float()), dim=-1)
    report = []
    for layer in sorted(state):
        source = cache[layer + 1].float()
        tuned = unembed(apply_lens(source, state[layer], form))
        logit = unembed(source)
        row = {
            "layer": layer,
            "tuned_kl": float((target.exp() * (target - F.log_softmax(tuned, -1))).sum(-1).mean()),
            "logit_kl": float((target.exp() * (target - F.log_softmax(logit, -1))).sum(-1).mean()),
        }
        report.append(row)
        print(f"  eval layer {layer:2d}: tuned KL {row['tuned_kl']:.3f}  logit KL {row['logit_kl']:.3f}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--objective", choices=["kl", "ols", "both"], default="both")
    parser.add_argument("--sequences", type=int, default=512)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-tokens", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    n_layers = model.n_layers

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    total = args.sequences + args.eval_sequences
    batches = jc.corpus_batches(
        tokenizer,
        dataset["text"],
        n_sequences=total,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    split = args.sequences // args.batch_size
    print(f"caching residuals for {total} sequences ...", flush=True)
    train_cache = cache_residuals(hf_model, model, batches[:split])
    eval_cache = cache_residuals(hf_model, model, batches[split:])
    print("train cache", tuple(train_cache.shape), "eval cache", tuple(eval_cache.shape), flush=True)

    # A float32 copy of the readout head.  See fit_kl's docstring.
    head_weight = model.lm_head.weight.float()
    norm_weight = model.final_norm.weight.float()
    gemma = type(model.final_norm).__name__.startswith("Gemma")
    eps = getattr(model.final_norm, "eps", 1e-6)

    def unembed(residual: torch.Tensor) -> torch.Tensor:
        normed = residual * torch.rsqrt(residual.pow(2).mean(-1, keepdim=True) + eps)
        gain = (1.0 + norm_weight) if gemma else norm_weight
        return (normed * gain) @ head_weight.T

    objectives = ["ols", "kl"] if args.objective == "both" else [args.objective]
    summary = {}
    for objective in objectives:
        print(f"=== fitting {objective} ===", flush=True)
        if objective == "ols":
            state = fit_ols(train_cache, n_layers, args.ridge)
        else:
            state = fit_kl(
                model,
                train_cache,
                n_layers,
                unembed,
                steps=args.steps,
                weight_decay=args.weight_decay,
                batch_tokens=args.batch_tokens,
                lr=args.lr,
                seed=args.seed,
            )
        print(f"--- held-out evaluation ({objective}) ---", flush=True)
        form = "affine" if objective == "ols" else "residual"
        report = evaluate(model, eval_cache, state, n_layers, unembed, form)
        summary[objective] = report
        torch.save(
            {
                "state": state,
                "objective": objective,
                "form": form,
                "model": args.model,
                "n_train_tokens": int(train_cache.shape[1]),
                "seq_len": args.seq_len,
                "ridge": args.ridge,
                "steps": args.steps,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "heldout": report,
            },
            ARTIFACTS / f"tuned_lens_{objective}.pt",
        )
        print("wrote", ARTIFACTS / f"tuned_lens_{objective}.pt", flush=True)

    (OUT_DIR / "tuned_lens_heldout.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
