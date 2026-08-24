"""A gated lens: linear map plus K sigmoid units, fitted on cached activations.

CLAIMS.md Addendum 7. Every lens so far is one linear map, so none can
represent the structure we actually measured -- a response that is flat, rises
sharply, then saturates. That is one sigmoid. The minimal extension is

    F_l(h)  ~=  A h + b + sum_k v_k * sigmoid(w_k . h + c_k)

`A, b` is the ordinary least-squares lens, solved in closed form; the `K` units
are then fitted to the *residual* it leaves behind. No model forward passes are
needed beyond the single caching pass the tuned lens already requires, and the
nonlinear part adds `2 K d` parameters -- 3% of the linear map at `K = 64`.

Each unit is interpretable in the same currency as the lens itself: `w_k` is a
detector direction and `v_k` a write direction, both decodable through the
unembedding as "when this is present strongly enough, the model becomes
disposed to say that".
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
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

# Length-matched probe pairs, reusing the items that passed the effectiveness
# precondition: two that gate, two that do not.
PROBES = [
    ("legs-spider-dog", True, "Fact: The number of legs on a {} is ", "spider", "8", "dog", "4"),
    ("sides-triangle-square", True, "Fact: The number of sides on a {} is ",
     "triangle", "3", "square", "4"),
    ("antonym-up-wet", False, "Fact: In one word, the opposite of {} is ",
     "up", "down", "wet", "dry"),
    ("capital-language", False, "Fact: The main language spoken in {} is ",
     "France", "French", "Spain", "Spanish"),
]


def cache_pairs(model, batches, layer):
    """Return `(h_l, h_final)` at every valid position of the corpus."""
    sources, targets = [], []
    for input_ids in batches:
        positions = jc.valid_positions(input_ids.shape[1]).to(model.device)
        stack = jc.residual_stack(model, input_ids)[:, :, positions]
        sources.append(stack[layer + 1].flatten(0, 1).to(torch.float32))
        targets.append(stack[model.n_layers].flatten(0, 1).to(torch.float32))
    return torch.cat(sources), torch.cat(targets)


def fit_linear(source, target, ridge, d_model, device):
    mean_x, mean_y = source.mean(0).double(), target.mean(0).double()
    cx = (source - mean_x.float()).double()
    cy = (target - mean_y.float()).double()
    cov_xx = cx.T @ cx / cx.shape[0]
    cov_yx = cy.T @ cx / cx.shape[0]
    scale = torch.diagonal(cov_xx).mean()
    weight = torch.linalg.solve(
        cov_xx + ridge * scale * torch.eye(d_model, dtype=torch.float64, device=device),
        cov_yx.T,
    ).T.float()
    bias = (mean_y - weight.double() @ mean_x).float()
    return weight, bias


class GatedUnits(torch.nn.Module):
    """`sum_k v_k * sigmoid(w_k . h + c_k)`, fitted to a linear lens's residual.

    Detectors are initialised from the leading principal directions of the
    (standardised) activations rather than at random. A first attempt with
    random init collapsed every unit onto one shared direction, so the units
    are also kept apart by an orthogonality penalty during fitting.
    """

    def __init__(self, d_model: int, k: int, scale: float, init=None):
        super().__init__()
        detector = (init if init is not None
                    else torch.randn(k, d_model) / (d_model ** 0.5))
        self.detector = torch.nn.Parameter(detector.clone())
        self.threshold = torch.nn.Parameter(torch.zeros(k))
        self.write = torch.nn.Parameter(torch.randn(k, d_model) * scale / (k ** 0.5))

    def orthogonality(self) -> torch.Tensor:
        normed = F.normalize(self.detector, dim=1)
        gram = normed @ normed.T
        off_diagonal = gram - torch.diag(torch.diagonal(gram))
        return off_diagonal.pow(2).mean()

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(hidden @ self.detector.T + self.threshold)
        return gate @ self.write


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--train-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--diversity", type=float, default=1e4)
    parser.add_argument("--corpus", choices=["web", "domain"], default="web")
    parser.add_argument("--out", default="gated_lens.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    layer, device = args.layer, model.device
    gain = 1.0 + model.final_norm.weight.float()

    if args.corpus == "web":
        dataset = load_dataset("NeelNanda/pile-10k", split="train")
        batches = jc.corpus_batches(
            tokenizer, dataset["text"], n_sequences=args.train_sequences + args.eval_sequences,
            seq_len=args.seq_len, batch_size=8, seed=17,
        )
    else:
        # a counting-relation corpus: if the gate is absent from web text, the
        # units have no chance of finding it there.
        entities = ("spider", "ant", "dog", "cat", "bird", "horse", "cow", "bee", "beetle",
                    "octopus", "crab", "sheep", "pig", "frog", "chicken", "triangle",
                    "square", "pentagon", "hexagon", "octagon", "bicycle", "car", "tricycle",
                    "truck", "basketball", "volleyball", "soccer", "hockey")
        frames = ("The number of legs on a {} is .", "A {} has this many sides: .",
                  "Counting the parts of a {} gives .", "The number of wheels on a {} is .",
                  "How many players are on a {} team? .", "A {} has legs.")
        lines = [f.format(e) for e in entities for f in frames]
        generator = torch.Generator().manual_seed(0)
        ids = []
        need = (args.train_sequences + args.eval_sequences) * args.seq_len
        while len(ids) < need:
            for i in torch.randperm(len(lines), generator=generator).tolist():
                ids.extend(tokenizer.encode(lines[i] + "\n", add_special_tokens=False))
        usable = (len(ids) // (args.seq_len - 1)) * (args.seq_len - 1)
        rows_t = torch.tensor(ids[:usable]).view(-1, args.seq_len - 1)
        rows_t = torch.cat([torch.full((rows_t.shape[0], 1), tokenizer.bos_token_id), rows_t], 1)
        batches = list(rows_t[: args.train_sequences + args.eval_sequences].to(device).split(8))
    split = args.train_sequences // 8
    print("caching activations ...", flush=True)
    train_x, train_y = cache_pairs(model, batches[:split], layer)
    eval_x, eval_y = cache_pairs(model, batches[split:], layer)
    print(f"train {tuple(train_x.shape)}   eval {tuple(eval_x.shape)}", flush=True)

    weight, bias = fit_linear(train_x, train_y, args.ridge, model.d_model, device)

    def linear(h):
        return h @ weight.T + bias

    train_residual = train_y - linear(train_x)
    eval_residual = eval_y - linear(eval_x)
    total_train = (train_y - train_y.mean(0)).pow(2).sum()
    total_eval = (eval_y - eval_y.mean(0)).pow(2).sum()
    print(f"linear lens: train R2 = {1 - train_residual.pow(2).sum() / total_train:.4f}   "
          f"held-out R2 = {1 - eval_residual.pow(2).sum() / total_eval:.4f}", flush=True)

    centre_init, spread_init = train_x.mean(0), train_x.std(0) + 1e-6
    standardised = (train_x[:20000] - centre_init) / spread_init
    _, _, components = torch.pca_lowrank(standardised, q=min(args.k, 256))
    units = GatedUnits(model.d_model, args.k, float(train_residual.std()),
                       init=components.T[: args.k].contiguous()).to(device)
    optimizer = torch.optim.Adam(units.parameters(), lr=args.lr)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps)
    generator = torch.Generator(device="cpu").manual_seed(0)
    # standardise the input to the gates so thresholds start in a sensible range
    centre, spread = centre_init, spread_init
    best = {"r2": -float("inf"), "step": -1, "state": None}

    for step in range(args.steps):
        index = torch.randint(0, train_x.shape[0], (args.batch,), generator=generator).to(device)
        prediction = units((train_x[index] - centre) / spread)
        loss = F.mse_loss(prediction, train_residual[index]) + args.diversity * units.orthogonality()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        schedule.step()
        if step % 250 == 0 or step == args.steps - 1:
            with torch.no_grad():
                held = units((eval_x - centre) / spread)
                r2 = float(1 - (eval_residual - held).pow(2).sum() / total_eval)
                r2_linear = float(1 - eval_residual.pow(2).sum() / total_eval)
            if r2 > best["r2"]:
                best = {"r2": r2, "step": step,
                        "state": {k: v.detach().clone() for k, v in units.state_dict().items()}}
            print(f"  step {step:5d}  train mse {loss.item():.4f}   "
                  f"held-out R2 {r2:.4f}  (linear {r2_linear:.4f})", flush=True)

    # early stopping: keep the held-out best rather than the last step
    if best["state"] is not None:
        units.load_state_dict(best["state"])
    print(f"kept step {best['step']} (held-out R2 {best['r2']:.4f})", flush=True)
    with torch.no_grad():
        held = units((eval_x - centre) / spread)
        r2_gated = float(1 - (eval_residual - held).pow(2).sum() / total_eval)
        r2_linear = float(1 - eval_residual.pow(2).sum() / total_eval)

    def gated(h):
        return linear(h) + units((h - centre) / spread)

    # ---- the decisive test: can either lens follow a gate? -----------------
    probe_results = []
    alphas = torch.linspace(0, 1, 61)
    for name, gates, template, filler_a, answer_a, filler_b, answer_b in PROBES:
        def token(word):
            for candidate in (word, " " + word):
                pieces = tokenizer.encode(candidate, add_special_tokens=False)
                if len(pieces) == 1:
                    return pieces[0]
            return None
        token_a, token_b = token(answer_a), token(answer_b)
        enc_a = tokenizer(template.format(filler_a), return_tensors="pt").to(device)
        enc_b = tokenizer(template.format(filler_b), return_tensors="pt").to(device)
        if enc_a.input_ids.shape[1] != enc_b.input_ids.shape[1]:
            continue
        stack_a = jc.residual_stack(model, enc_a.input_ids)
        stack_b = jc.residual_stack(model, enc_b.input_ids)
        position = enc_a.input_ids.shape[1] - 1
        start_full = stack_a[layer + 1][0].float()
        end_full = stack_b[layer + 1][0].float()
        w_diff = (model.lm_head.weight[token_b].float()
                  - model.lm_head.weight[token_a].float()) * gain

        captured: list[torch.Tensor] = []

        def grab(_m, _a, out):
            captured.append(jc._block_output(out)[:, position].float())

        for chunk in alphas.split(16):
            blend = chunk.to(device)[:, None, None]
            states = (1 - blend) * start_full[None] + blend * end_full[None]

            def inject(_m, _a, out, states=states):
                hidden = jc._block_output(out)
                return jc._rewrap(out, states.to(hidden.dtype))

            handles = [model.layers[layer].register_forward_hook(inject),
                       model.layers[-1].register_forward_hook(grab)]
            try:
                with torch.inference_mode():
                    model.forward(enc_a.input_ids.expand(chunk.shape[0], -1))
            finally:
                for handle in handles:
                    handle.remove()
        actual = (torch.cat(captured) @ w_diff).cpu().numpy()

        path = ((1 - alphas.to(device)[:, None]) * start_full[position]
                + alphas.to(device)[:, None] * end_full[position])
        with torch.no_grad():
            predicted_linear = (linear(path) @ w_diff).cpu().numpy()
            predicted_gated = (gated(path) @ w_diff).cpu().numpy()

        def fit_quality(prediction):
            # best affine rescaling of the prediction, so only *shape* is scored
            design = np.stack([prediction, np.ones_like(prediction)], 1)
            coefficients, *_ = np.linalg.lstsq(design, actual, rcond=None)
            residual = actual - design @ coefficients
            return float(1 - residual.dot(residual) / ((actual - actual.mean()) ** 2).sum())

        entry = {"name": name, "gates": gates,
                 "shape_r2_linear": fit_quality(predicted_linear),
                 "shape_r2_gated": fit_quality(predicted_gated)}
        probe_results.append(entry)
        print(f"{name:24s} gates={str(gates):5s}  shape R2: linear "
              f"{entry['shape_r2_linear']:+.3f}  gated {entry['shape_r2_gated']:+.3f}", flush=True)

    # ---- what did the units learn? -----------------------------------------
    with torch.no_grad():
        activity = torch.sigmoid((eval_x - centre) / spread @ units.detector.T + units.threshold)
        variance = activity.var(0) * units.write.norm(dim=1) ** 2
        top = torch.topk(variance, min(8, args.k)).indices.tolist()
        described = []
        for k in top:
            detector_scores = model.unembed((units.detector[k] / spread) * spread).squeeze()
            write_scores = model.unembed(units.write[k]).squeeze()
            described.append({
                "unit": k,
                "fires_on": jc.top_tokens(tokenizer, detector_scores, 6),
                "writes": jc.top_tokens(tokenizer, write_scores, 6),
                "activity_mean": float(activity[:, k].mean()),
            })
            print(f"  unit {k:3d} fires on {described[-1]['fires_on'][:4]}"
                  f"  -> writes {described[-1]['writes'][:4]}", flush=True)

    payload = {"model": args.model, "layer": layer, "k": args.k,
               "heldout_r2_linear": r2_linear, "heldout_r2_gated": r2_gated,
               "probes": probe_results, "units": described,
               "n_train_tokens": int(train_x.shape[0])}
    (HERE / args.out).write_text(json.dumps(payload, indent=1))
    torch.save({"weight": weight.cpu(), "bias": bias.cpu(),
                "units": units.state_dict(), "centre": centre.cpu(), "spread": spread.cpu()},
               ARTIFACTS / "gated_lens.pt")
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
