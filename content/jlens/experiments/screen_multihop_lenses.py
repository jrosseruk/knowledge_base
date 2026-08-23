"""Claim 1 at scale: do the three lenses answer different questions?

Runs the J-lens, logit lens and Tuned Lens at *every* layer on Anthropic's
released two-hop prompt sets (`jacobian-lens/data/`, Apache-2.0), and records
full-vocabulary ranks for both the bridge entity `A` and the final answer `B`.

Preregistered readouts (CLAIMS.md, Claim 1):

  * `A` recovery      -- min-over-layers full-vocabulary rank of the
                         intermediate, per lens.  This is the paper's pass@k
                         statistic.
  * skip-ahead        -- at the layer where each lens best recovers `A`, the
                         rank of `B`.  The Tuned Lens is predicted to rank `B`
                         far above `A` at layers where the J-lens still reads
                         `A`.

Only items the model actually answers correctly are scored; the behavioural
filter runs first and its verdict is recorded for every item, pass or fail.
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


def token_forms(tokenizer, word: str) -> list[int]:
    """Single-token ids for a word, trying the bare and space-prefixed forms."""
    ids = []
    for candidate in (word, " " + word, word.capitalize(), " " + word.capitalize()):
        pieces = tokenizer.encode(candidate, add_special_tokens=False)
        if len(pieces) == 1 and pieces[0] not in ids:
            ids.append(pieces[0])
    return ids


def apply_lens(source, entry, form):
    """`A x + b` for the affine (OLS) form, `x + W x + b` for the residual form."""
    weight = entry["A"].to(source.device, torch.float32)
    bias = entry["b"].to(source.device, torch.float32)
    translated = source @ weight.T + bias
    return translated + source if form == "residual" else translated


def best_rank(scores: torch.Tensor, ids: list[int]) -> int:
    return min(jc.rank_of(scores, i) for i in ids)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="google/gemma-3-12b-it")
    parser.add_argument("--corpus-sequences", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--tuned", default="tuned_lens_kl.pt")
    parser.add_argument("--out", default="multihop_three_lens_screen.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)
    n_layers = model.n_layers

    tuned = torch.load(ARTIFACTS / args.tuned, map_location="cuda", weights_only=False)
    tuned_state = tuned["state"]
    tuned_form = tuned.get("form", "residual")

    items = json.loads((DATA / "experiments" / "probe-swap.json").read_text())["items"]
    if args.max_items:
        items = items[: args.max_items]

    # ---- behavioural filter ------------------------------------------------
    kept = []
    for item in items:
        encoded = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = hf_model.generate(
                **encoded, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.eos_token_id
            )
        continuation = tokenizer.decode(generated[0, encoded.input_ids.shape[1] :])
        correct = continuation.strip().lower().startswith(item["answer"].lower())
        intermediate_ids = token_forms(tokenizer, item["intermediate"])
        answer_ids = token_forms(tokenizer, item["answer"])
        item = dict(item)
        item["continuation"] = continuation
        item["behaviourally_correct"] = bool(correct)
        item["intermediate_ids"] = intermediate_ids
        item["answer_ids"] = answer_ids
        item["single_token_ok"] = bool(intermediate_ids) and bool(answer_ids)
        kept.append(item)

    scored = [i for i in kept if i["behaviourally_correct"] and i["single_token_ok"]]
    print(
        f"{len(scored)}/{len(kept)} items pass the behavioural + single-token filter",
        flush=True,
    )
    if not scored:
        raise SystemExit("no items to score")

    # ---- activations at the readout position -------------------------------
    activations = []
    for item in scored:
        encoded = tokenizer(item["prompt"], return_tensors="pt").to(model.device)
        activations.append(jc.residual_stack(model, encoded.input_ids)[:, 0, -1].float())
    activations = torch.stack(activations)  # [n_items, n_layers + 1, d_model]
    print("activations", tuple(activations.shape), flush=True)

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    batches = jc.corpus_batches(
        tokenizer,
        dataset["text"],
        n_sequences=args.corpus_sequences,
        seq_len=args.seq_len,
        batch_size=1,
    )

    per_item = {item["name"]: {"item": item, "layers": []} for item in scored}
    for layer in range(n_layers):
        started = time.time()
        directions = activations[:, layer + 1].to(model.device)
        transported = jc.transport(model, layer, directions, batches)
        tuned_hidden = apply_lens(directions, tuned_state[layer], tuned_form)

        j_scores = model.unembed(transported)
        logit_scores = model.unembed(directions)
        tuned_scores = model.unembed(tuned_hidden)

        for index, item in enumerate(scored):
            row = {"layer": layer}
            for lens_name, scores in (
                ("j", j_scores[index]),
                ("logit", logit_scores[index]),
                ("tuned", tuned_scores[index]),
            ):
                row[f"{lens_name}_A"] = best_rank(scores, item["intermediate_ids"])
                row[f"{lens_name}_B"] = best_rank(scores, item["answer_ids"])
            row["j_top"] = jc.top_tokens(tokenizer, j_scores[index], 5)
            row["tuned_top"] = jc.top_tokens(tokenizer, tuned_scores[index], 5)
            row["logit_top"] = jc.top_tokens(tokenizer, logit_scores[index], 5)
            per_item[item["name"]]["layers"].append(row)
        print(f"layer {layer:2d} done in {time.time() - started:.1f}s", flush=True)

    # ---- preregistered summary --------------------------------------------
    summary = {"n_items": len(scored), "lenses": {}}
    for lens_name in ("j", "logit", "tuned"):
        best_a_ranks, skip_b_ranks, skip_a_ranks = [], [], []
        for record in per_item.values():
            rows = record["layers"]
            best = min(rows, key=lambda r: r[f"{lens_name}_A"])
            best_a_ranks.append(best[f"{lens_name}_A"])
            skip_b_ranks.append(best[f"{lens_name}_B"])
            skip_a_ranks.append(best[f"{lens_name}_A"])
        summary["lenses"][lens_name] = {
            "pass@1": sum(r <= 1 for r in best_a_ranks) / len(best_a_ranks),
            "pass@5": sum(r <= 5 for r in best_a_ranks) / len(best_a_ranks),
            "pass@10": sum(r <= 10 for r in best_a_ranks) / len(best_a_ranks),
            "median_best_A_rank": sorted(best_a_ranks)[len(best_a_ranks) // 2],
        }

    # At the layer where the J-lens best reads A, what do the other lenses say?
    aligned = []
    for record in per_item.values():
        rows = record["layers"]
        best = min(rows, key=lambda r: r["j_A"])
        aligned.append(
            {
                "name": record["item"]["name"],
                "layer": best["layer"],
                "j_A": best["j_A"],
                "j_B": best["j_B"],
                "tuned_A": best["tuned_A"],
                "tuned_B": best["tuned_B"],
                "logit_A": best["logit_A"],
                "logit_B": best["logit_B"],
            }
        )
    summary["at_j_best_layer"] = {
        "median_tuned_B": sorted(a["tuned_B"] for a in aligned)[len(aligned) // 2],
        "median_tuned_A": sorted(a["tuned_A"] for a in aligned)[len(aligned) // 2],
        "fraction_tuned_prefers_B": sum(a["tuned_B"] < a["tuned_A"] for a in aligned) / len(aligned),
        "fraction_j_prefers_A": sum(a["j_A"] < a["j_B"] for a in aligned) / len(aligned),
        "fraction_logit_prefers_A": sum(a["logit_A"] < a["logit_B"] for a in aligned) / len(aligned),
    }
    print(json.dumps(summary, indent=1))

    (HERE / args.out).write_text(
        json.dumps(
            {
                "model": args.model,
                "corpus_sequences": args.corpus_sequences,
                "seq_len": args.seq_len,
                "tuned_lens": args.tuned,
                "behavioural": [
                    {k: v for k, v in i.items() if k != "layers"} for i in kept
                ],
                "summary": summary,
                "aligned": aligned,
                "per_item": {k: v["layers"] for k, v in per_item.items()},
            },
            indent=1,
        )
    )
    print("wrote", HERE / args.out)


if __name__ == "__main__":
    main()
