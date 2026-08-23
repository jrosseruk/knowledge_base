"""Pilot: does a *faithfully estimated* J-lens recover `Mars` in Gemma 3 12B IT?

Earlier screens in this directory read the lens at four hand-picked layers and
injected the probe direction at a single early source position (index 8), which
sits inside the attention-sink band that the reference estimator explicitly
excludes.  This script uses the estimator from `jlens_core` -- perturbation at
every valid source position, cotangent summed over every valid target position,
16 leading positions skipped -- and sweeps *all* layers.

Result is written to `pilot_mars_all_layers.json`.  It is a screen, not
evidence; the preregistered criteria live in CLAIMS.md.
"""

from __future__ import annotations

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
torch.manual_seed(0)

MODEL = "google/gemma-3-12b-it"
PROMPT = "Fact: The color of the planet fourth from the Sun is "
INTERMEDIATE = "Mars"
ANSWER = "red"
N_CORPUS = 64
SEQ_LEN = 128
BATCH = 8
OUT = Path(__file__).resolve().parent / "pilot_mars_all_layers.json"


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model = jc.wrap(hf_model, tokenizer)

    encoded = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    print("prompt tokens:", [tokenizer.decode([i]) for i in encoded.input_ids[0].tolist()])

    with torch.inference_mode():
        generated = hf_model.generate(
            **encoded, max_new_tokens=6, do_sample=False, pad_token_id=tokenizer.eos_token_id
        )
    stack = jc.residual_stack(model, encoded.input_ids)
    continuation = tokenizer.decode(generated[0, encoded.input_ids.shape[1] :])
    print("greedy continuation:", repr(continuation))

    def single_token(word: str) -> int:
        pieces = tokenizer.encode(word, add_special_tokens=False)
        if len(pieces) != 1:
            raise ValueError(f"{word!r} -> {pieces}")
        return pieces[0]

    ids = {INTERMEDIATE: single_token(INTERMEDIATE), ANSWER: single_token(ANSWER)}
    print("token ids:", ids)

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    batches = jc.corpus_batches(
        tokenizer, dataset["text"], n_sequences=N_CORPUS, seq_len=SEQ_LEN, batch_size=BATCH
    )

    rows = []
    for layer in range(model.n_layers):
        hidden = stack[layer + 1][0, -1].detach().float()
        started = time.time()
        transported = jc.transport(model, layer, hidden, batches)[0]
        j_scores = model.unembed(transported).squeeze()
        logit_scores = model.unembed(hidden).squeeze()
        row = {
            "layer": layer,
            "j": {
                **{f"{w}_rank": jc.rank_of(j_scores, i) for w, i in ids.items()},
                "top": jc.top_tokens(tokenizer, j_scores, 8),
            },
            "logit": {
                **{f"{w}_rank": jc.rank_of(logit_scores, i) for w, i in ids.items()},
                "top": jc.top_tokens(tokenizer, logit_scores, 8),
            },
            "seconds": round(time.time() - started, 2),
        }
        rows.append(row)
        print(
            f"L{layer:02d}  J: Mars {row['j']['Mars_rank']:>6} red {row['j']['red_rank']:>6} "
            f"| logit: Mars {row['logit']['Mars_rank']:>6} red {row['logit']['red_rank']:>6} "
            f"| J top: {row['j']['top'][:5]}",
            flush=True,
        )

    OUT.write_text(
        json.dumps(
            {
                "model": MODEL,
                "prompt": PROMPT,
                "greedy_continuation": continuation,
                "n_corpus_sequences": N_CORPUS,
                "seq_len": SEQ_LEN,
                "token_ids": ids,
                "rows": rows,
            },
            indent=1,
        )
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
