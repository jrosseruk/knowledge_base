"""Claim 2, mechanism test: sweep an internal representation and watch F_l bend.

The intervention is the paper's lens-coordinate swap (Figure 4C), applied to a
single layer and a single token position so that the object being perturbed is
one honest vector `h` and the object being probed is one honest function

    F_l : h_{l, pos}  ->  h_{final, pos}     (everything else held fixed).

Let `v_A`, `v_A'` be the J-lens vectors of the bridge entity and its
counterfactual, `V = [v_A; v_A']`, and `a = (V V^T)^{-1} V h` the activation's
coordinates in that two-dimensional frame.  Exchanging the two coordinates
leaves the orthogonal component untouched and moves `h` by

    delta = (a_2 - a_1) (v_A - v_A'),        h(alpha) = h + alpha * delta,

so `alpha = 0` is the unmodified model and `alpha = 1` is the full swap.

**Everything is read out through one fixed linear functional** so that the
model and the three lenses are directly comparable:

    s(x) = < w_B' - w_B , x >,      w_v = W_U[v] * (final-norm gain)

which is the unnormalised logit difference between the counterfactual answer
and the true answer.  Then

    actual(alpha) = s( F_l(h(alpha)) )        <- full forward pass, nonlinear
    jlens(alpha)  = s( J_l h(alpha) )         <- averaged local Jacobian
    tuned(alpha)  = s( A_l h(alpha) + b_l )   <- global affine regression
    logit(alpha)  = s( h(alpha) )             <- identity

The three lens curves are straight lines by construction; the point of the
figure is where each line sits relative to the model's real response.  We also
compute the *pointwise* tangent `s(dF_l/dh . delta)` by a JVP at `alpha = 0`,
which is the local slope the J-lens is supposed to be approximating, and the
secant slope over a shrinking window, which interpolates between that tangent
and the chord.

Controls, both preregistered in CLAIMS.md:

  * timing -- the same swap applied to the *answer* pair (`B <-> B'`) at the
    same layer should move the output less than the bridge swap, showing `A`
    is present and used before `B` is represented.
  * affineness -- the same sweep run at a late layer, where `J_l` and `A_l`
    agree, should show a straight `actual` curve and coincident lens lines.

Writes `causal_sweep.json` (+ `.png`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
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
DATA = HERE.parent.parent.parent / "jacobian-lens" / "data"


def single_token(tokenizer, word: str) -> int | None:
    for candidate in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        pieces = tokenizer.encode(candidate, add_special_tokens=False)
        if len(pieces) == 1:
            return pieces[0]
    return None


def norm_gain(model) -> torch.Tensor:
    weight = model.final_norm.weight.float()
    return 1.0 + weight if type(model.final_norm).__name__.startswith("Gemma") else weight


def swap_delta(hidden: torch.Tensor, vectors: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Move `hidden` by the coordinate exchange in the frame spanned by `vectors`.

    `vectors` is `[2, d]`.  Coordinates are the least-squares (dual) ones, so
    the component orthogonal to the two vectors is untouched.

    Two J-lens vectors for related entities can be strongly correlated, which
    makes the 2x2 Gram matrix ill-conditioned and the dual coordinates huge.
    The conditioning is reported alongside the delta so a swap that is really
    an extrapolation cannot be mistaken for a clean exchange.
    """
    gram = vectors @ vectors.T
    coordinates = torch.linalg.solve(gram, vectors @ hidden)
    swapped = coordinates.flip(0)
    delta = (swapped - coordinates) @ vectors
    cosine = float(
        gram[0, 1] / (vectors[0].norm() * vectors[1].norm())
    )
    diagnostics = {
        "vector_cosine": cosine,
        "gram_condition": float(torch.linalg.cond(gram)),
        "coordinates": coordinates.tolist(),
    }
    return delta, diagnostics


def run_from_layer(hf_model, model, input_ids, layer, sites, position, offsets):
    """Forward with `alpha * delta` added at `sites`, batched over alphas.

    `offsets` is `[n_alphas, d_model]`.  Returns the final residual at
    `position`, shape `[n_alphas, d_model]`.
    """
    n = offsets.shape[0]
    expanded = input_ids.expand(n, -1)
    captured: dict[str, torch.Tensor] = {}

    def inject(_module, _args, module_output):
        hidden = jc._block_output(module_output)
        edited = hidden.clone()
        edited[:, sites] += offsets[:, None, :].to(hidden.dtype)
        return jc._rewrap(module_output, edited)

    def capture(_module, _args, module_output):
        captured["h"] = jc._block_output(module_output)

    handles = [
        model.layers[layer].register_forward_hook(inject),
        model.layers[-1].register_forward_hook(capture),
    ]
    try:
        with torch.inference_mode():
            model.forward(expanded)
    finally:
        for handle in handles:
            handle.remove()
    return captured["h"][:, position].float()


def local_tangent(hf_model, model, input_ids, layer, sites, position, delta):
    """`dF_l/dh . delta` at this exact activation, by forward-mode AD."""
    captured: dict[str, torch.Tensor] = {}

    def run(alpha: torch.Tensor) -> torch.Tensor:
        def inject(_module, _args, module_output):
            block = jc._block_output(module_output)
            edited = block.clone()
            edited[:, sites] = block[:, sites] + (alpha * delta).to(block.dtype)
            return jc._rewrap(module_output, edited)

        def capture(_module, _args, module_output):
            captured["h"] = jc._block_output(module_output)

        handles = [
            model.layers[layer].register_forward_hook(inject),
            model.layers[-1].register_forward_hook(capture),
        ]
        try:
            model.forward(input_ids)
        finally:
            for handle in handles:
                handle.remove()
        return captured["h"][:, position].float()

    zero = torch.zeros((), device=model.device)
    one = torch.ones((), device=model.device)
    with torch.no_grad():
        _, tangent = torch.func.jvp(run, (zero,), (one,))
    return tangent[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--items", nargs="*", default=None, help="probe-swap item names")
    parser.add_argument("--layer", type=int, default=None, help="override the swap layer")
    parser.add_argument("--control-layer", type=int, default=45)
    parser.add_argument("--sites", choices=["last", "all"], default="last",
                        help="token positions the swap is written to")
    parser.add_argument("--corpus-sequences", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--alpha-min", type=float, default=-0.5)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-steps", type=int, default=51)
    parser.add_argument("--tuned-kl", default="tuned_lens_kl.pt")
    parser.add_argument("--tuned-ols", default="tuned_lens_ols.pt")
    parser.add_argument("--screen", default="multihop_three_lens_screen.json")
    parser.add_argument("--out", default="causal_sweep.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    gain = norm_gain(model)

    loaded = {
        "kl": torch.load(ARTIFACTS / args.tuned_kl, map_location="cuda", weights_only=False),
        "ols": torch.load(ARTIFACTS / args.tuned_ols, map_location="cuda", weights_only=False),
    }
    lenses = {k: v["state"] for k, v in loaded.items()}
    forms = {k: v.get("form", "residual" if k == "kl" else "affine") for k, v in loaded.items()}

    all_items = json.loads((DATA / "experiments" / "probe-swap.json").read_text())["items"]
    by_name = {i["name"]: i for i in all_items}
    names = args.items or ["mars-color"]

    # The pre-switch layer is taken from the Claim 1 screen: the layer at which
    # the J-lens best recovers the bridge entity.  It is chosen before any
    # causal measurement is made.
    chosen_layers = {}
    screen_path = HERE / args.screen
    if screen_path.exists():
        screen = json.loads(screen_path.read_text())
        for entry in screen["aligned"]:
            chosen_layers[entry["name"]] = entry["layer"]

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    batches = jc.corpus_batches(
        tokenizer, dataset["text"], n_sequences=args.corpus_sequences,
        seq_len=args.seq_len, batch_size=1, seed=11,
    )

    alphas = torch.linspace(args.alpha_min, args.alpha_max, args.alpha_steps)
    results = []

    for name in names:
        item = by_name[name]
        ids = {
            key: single_token(tokenizer, item[key])
            for key in ("intermediate", "swap_to", "answer", "swap_answer")
        }
        if any(v is None for v in ids.values()):
            print(f"skipping {name}: not all of {item} are single tokens -> {ids}")
            continue

        encoded = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        input_ids = encoded.input_ids
        position = input_ids.shape[1] - 1
        hidden_states = jc.residual_stack(model, input_ids)
        with torch.inference_mode():
            generated = hf_model.generate(
                **encoded, max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        baseline_continuation = tokenizer.decode(generated[0, input_ids.shape[1]:])

        sites = (
            torch.tensor([position], device=model.device)
            if args.sites == "last"
            else torch.arange(1, input_ids.shape[1], device=model.device)
        )

        for role, layer in (
            ("preswitch", args.layer if args.layer is not None else chosen_layers.get(name, 34)),
            ("late_control", args.control_layer),
        ):
            hidden = hidden_states[layer + 1][0, position].float()

            # J-lens vectors for the four tokens, at this layer.
            token_list = [ids["intermediate"], ids["swap_to"], ids["answer"], ids["swap_answer"]]
            vectors = jc.lens_vectors(model, layer, token_list, batches)
            bridge_frame = vectors[:2]
            answer_frame = vectors[2:]

            # readout functional: unnormalised logit(B') - logit(B)
            w_diff = (
                model.lm_head.weight[ids["swap_answer"]].float()
                - model.lm_head.weight[ids["answer"]].float()
            ) * gain

            sweeps = {}
            for swap_name, frame in (("bridge", bridge_frame), ("answer", answer_frame)):
                delta, frame_diagnostics = swap_delta(hidden, frame)

                offsets = alphas.to(model.device)[:, None] * delta[None]
                path = hidden[None] + offsets

                actual_final = []
                for chunk in offsets.split(16):
                    actual_final.append(
                        run_from_layer(hf_model, model, input_ids, layer, sites, position, chunk)
                    )
                actual_final = torch.cat(actual_final)
                actual = (actual_final @ w_diff).cpu()
                final_logits = model.unembed(actual_final)
                true_logit_difference = (
                    final_logits[:, ids["swap_answer"]] - final_logits[:, ids["answer"]]
                ).cpu()
                top_at_alpha = [
                    tokenizer.decode([int(i)]) for i in final_logits.argmax(dim=-1).tolist()
                ]

                transported = jc.transport(model, layer, path, batches)
                jlens = (transported @ w_diff).cpu()

                lines = {}
                for objective, state in lenses.items():
                    weight = state[layer]["A"].to(model.device, torch.float32)
                    bias = state[layer]["b"].to(model.device, torch.float32)
                    translated = path @ weight.T + bias
                    if forms[objective] == "residual":
                        translated = translated + path
                    lines[objective] = (translated @ w_diff).cpu()

                tangent_vector = local_tangent(
                    hf_model, model, input_ids, layer, sites, position, delta
                )
                tangent_slope = float(tangent_vector @ w_diff)

                sweeps[swap_name] = {
                    "alphas": alphas.tolist(),
                    "actual": actual.tolist(),
                    "true_logit_difference": true_logit_difference.tolist(),
                    "top_token": top_at_alpha,
                    "jlens": jlens.tolist(),
                    "tuned_kl": lines["kl"].tolist(),
                    "tuned_ols": lines["ols"].tolist(),
                    "logit": (path @ w_diff).cpu().tolist(),
                    "pointwise_tangent_slope": tangent_slope,
                    "delta_norm": float(delta.norm()),
                    "hidden_norm": float(hidden.norm()),
                    **frame_diagnostics,
                }
                print(
                    f"{name} L{layer} {swap_name}: actual(0)={actual[alphas.abs().argmin()]:.1f} "
                    f"tangent slope={tangent_slope:.1f}  |delta|/|h|="
                    f"{delta.norm() / hidden.norm():.3f}  "
                    f"cos(v_A, v_A')={frame_diagnostics['vector_cosine']:+.3f}",
                    flush=True,
                )

            results.append(
                {
                    "name": name,
                    "item": item,
                    "role": role,
                    "layer": layer,
                    "position": position,
                    "baseline_continuation": baseline_continuation,
                    "sites": args.sites,
                    "token_ids": ids,
                    "sweeps": sweeps,
                }
            )

    (HERE / args.out).write_text(
        json.dumps(
            {
                "model": args.model,
                "corpus_sequences": args.corpus_sequences,
                "alpha_range": [args.alpha_min, args.alpha_max, args.alpha_steps],
                "results": results,
            },
            indent=1,
        )
    )
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
