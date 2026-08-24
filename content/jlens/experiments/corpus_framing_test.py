"""Does "in-domain" mean the topic, or the prompt construction?

`fig_in_domain.png` reported the tuned lens landing on the chord for
`spider -> dog` (1.02x). The lens grid, same item and estimator, reported 0.60x.
The corpora differ: the first is 60% two-hop ("the animal that spins webs"),
matching the probe's construction; the second is 13% two-hop and mostly direct
naming ("a spider").

This isolates that one variable. Same item, same size, same ridge; only the
framing of the corpus changes.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
import jlens_core as jc
hf_logging.set_verbosity_error()
HERE = Path(__file__).resolve().parent

ANIMALS = ("spider", "ant", "bee", "dog", "cat", "horse", "cow", "bird", "beetle",
           "octopus", "crab", "sheep", "pig", "frog", "chicken")
DESCRIPTIONS = ("spins webs", "barks and fetches sticks", "buzzes and makes honey",
                "marches in colonies", "purrs and chases mice", "lays eggs and has feathers",
                "grazes and moos", "gallops and neighs", "roots in mud", "croaks by the pond")
TWO_HOP = ("Fact: The number of legs on the animal that {} is ",
           "Fact: The animal that {} has this many legs: ",
           "Fact: Counting the legs of the creature that {} gives ",
           "Fact: The creature that {} walks on this many legs: ",
           "Fact: If you count the legs on the animal that {}, you get ")
ONE_HOP = ("Fact: The number of legs on a {} is ", "Fact: A {} walks on this many legs: ",
           "Fact: Counting the legs of a {} gives ", "Fact: How many legs does a {} have? ",
           "Fact: A {} is an animal with this many legs: ")

def corpus(tokenizer, lines, seq_len, device, target, batch=2):
    generator = torch.Generator().manual_seed(0)
    ids = []
    while len(ids) < target * seq_len:
        for i in torch.randperm(len(lines), generator=generator).tolist():
            ids.extend(tokenizer.encode(lines[i] + "\n", add_special_tokens=False))
    usable = (len(ids) // (seq_len - 1)) * (seq_len - 1)
    rows = torch.tensor(ids[:usable]).view(-1, seq_len - 1)
    rows = torch.cat([torch.full((rows.shape[0], 1), tokenizer.bos_token_id), rows], 1)
    return list(rows[:target].to(device).split(batch))

def fit_ols(model, batches, layer, ridge=1e-2):
    xs, ys = [], []
    for ids in batches:
        pos = jc.valid_positions(ids.shape[1]).to(model.device)
        stack = jc.residual_stack(model, ids)[:, :, pos]
        xs.append(stack[layer + 1].flatten(0, 1).float())
        ys.append(stack[model.n_layers].flatten(0, 1).float())
    x, y = torch.cat(xs), torch.cat(ys)
    mx, my = x.mean(0).double(), y.mean(0).double()
    cx, cy = (x - mx.float()).double(), (y - my.float()).double()
    xx = cx.T @ cx / cx.shape[0]; yx = cy.T @ cx / cx.shape[0]
    eye = torch.eye(model.d_model, dtype=torch.float64, device=model.device)
    return torch.linalg.solve(xx + ridge * torch.diagonal(xx).mean() * eye, yx.T).T.float(), x.shape[0]

def main():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-12b-it")
    hf = AutoModelForCausalLM.from_pretrained("google/gemma-3-12b-it", torch_dtype=torch.bfloat16,
                                              device_map="cuda", attn_implementation="eager")
    model = jc.wrap(hf, tokenizer); layer, device = 34, model.device
    gain = 1.0 + model.final_norm.weight.float()
    tok8 = tokenizer.encode("8", add_special_tokens=False)[0]
    tok4 = tokenizer.encode("4", add_special_tokens=False)[0]
    w = (model.lm_head.weight[tok4].float() - model.lm_head.weight[tok8].float()) * gain

    a = tokenizer("Fact: The number of legs on the animal that spins webs is ",
                  return_tensors="pt").to(device)
    b = tokenizer("Fact: The number of legs on the animal that barks and fetches sticks is ",
                  return_tensors="pt").to(device)
    pos = a.input_ids.shape[1] - 1
    start = jc.residual_stack(model, a.input_ids)[layer + 1][0, -1].float()
    end = jc.residual_stack(model, b.input_ids)[layer + 1][0, -1].float()
    delta = end - start

    alphas = torch.linspace(0, 1, 61)
    got = []
    for chunk in ((1 - alphas.to(device)[:, None]) * start + alphas.to(device)[:, None] * end).split(16):
        def inject(_m, _a, out, chunk=chunk):
            h = jc._block_output(out); e = h.clone(); e[:, pos] = chunk.to(h.dtype)
            return jc._rewrap(out, e)
        def grab(_m, _a, out): got.append(jc._block_output(out)[:, pos].float())
        hs = [model.layers[layer].register_forward_hook(inject),
              model.layers[-1].register_forward_hook(grab)]
        try:
            with torch.inference_mode(): model.forward(a.input_ids.expand(chunk.shape[0], -1))
        finally:
            for h in hs: h.remove()
    actual = (torch.cat(got) @ w).cpu().numpy()
    chord = float(actual[-1] - actual[0])
    print(f"chord = {chord:.0f}\n")

    dataset = load_dataset("NeelNanda/pile-10k", split="train")
    variants = {
        "two-hop  (matches the probe)": [t.format(d) for d in DESCRIPTIONS for t in TWO_HOP],
        "one-hop  (direct naming)": [t.format(x) for x in ANIMALS for t in ONE_HOP],
        "mixed  (half each)": ([t.format(d) for d in DESCRIPTIONS for t in TWO_HOP]
                               + [t.format(x) for x in ANIMALS for t in ONE_HOP]),
    }
    out = {"chord": chord, "variants": {}}
    for name, lines in variants.items():
        batches = corpus(tokenizer, lines, 128, device, 170)
        weight, n_tok = fit_ols(model, batches, layer)
        j = float(jc.transport(model, layer, delta[None], batches)[0] @ w)
        tuned = float((delta @ weight.T) @ w)
        out["variants"][name] = {"lines": len(lines), "tokens": n_tok,
                                 "tuned_over_chord": tuned / chord, "j_over_chord": j / chord}
        print(f"{name:30s} {len(lines):>4} lines  tuned/chord {tuned/chord:5.2f}   J/chord {j/chord:5.2f}",
              flush=True)
    web = jc.corpus_batches(tokenizer, dataset["text"], n_sequences=170, seq_len=128, batch_size=2, seed=5)
    weight, _ = fit_ols(model, web, layer)
    j = float(jc.transport(model, layer, delta[None], web)[0] @ w)
    tuned = float((delta @ weight.T) @ w)
    out["variants"]["web text"] = {"tuned_over_chord": tuned / chord, "j_over_chord": j / chord}
    print(f"{'web text':30s}       tuned/chord {tuned/chord:5.2f}   J/chord {j/chord:5.2f}")
    (HERE / "corpus_framing_test.json").write_text(json.dumps(out, indent=1))

if __name__ == "__main__":
    main()
