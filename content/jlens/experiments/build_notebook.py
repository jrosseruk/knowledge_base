"""Assemble `content/jlens/03_locality_vs_prediction.ipynb` from cell sources.

Keeping the notebook in a build script rather than editing JSON by hand makes it
easy to keep prose and code in sync. Every code cell runs live; no cell loads a
precomputed number in place of computing it. Where the standalone scripts in
this directory ran at larger scale, the notebook says so and computes its own
smaller version rather than importing theirs.
"""

from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path(__file__).resolve().parent.parent / "03_locality_vs_prediction.ipynb"

cells: list[tuple[str, str]] = []


def md(source: str) -> None:
    cells.append(("markdown", source.strip("\n")))


def code(source: str) -> None:
    cells.append(("code", source.strip("\n")))


# ---------------------------------------------------------------------------

md(r"""
# Locality vs prediction: what the Jacobian lens and the Tuned Lens each measure

The [Jacobian lens](https://transformer-circuits.pub/2026/workspace/index.html)
reads an intermediate activation by transporting it into the final-layer basis
with an **averaged local Jacobian**. The [Tuned
Lens](https://arxiv.org/abs/2303.08112) does it with a **fitted global
predictor**. On two-hop prompts they disagree: the J-lens surfaces the bridge
entity the model is currently holding, while the Tuned Lens jumps to the answer.

Neel Nanda's review of the paper proposes *why*:

> linear regression asks: "Given the model is in a context where it is thinking
> about basketball, what is our best guess for what it will be thinking about at
> the final layer?" [...] The Jacobian is more like: "If the model thought about
> this concept an infinitesimal amount more on an arbitrary prompt, what would
> it be more likely to say?" Because it's an infinitesimal amount, there isn't
> enough time for nonlinearities to change.

That explanation only has content if the downstream computation is genuinely
nonlinear. So this notebook keeps two things apart:

* **Claim 1 — the phenomenon.** At a layer where the model holds an unspoken
  intermediate, the three lenses read out different things.
* **Claim 2 — the explanation.** That difference is *caused* by nonlinearity in
  the computation downstream of the layer.

Claim 1 is reported in the paper; we reproduce it in an open 12B model. Claim 2
is not isolated by the paper's comparison and needs its own experiment.

The short version of what follows: **Claim 1 partly replicates. Claim 2's
premise holds and is measurable exactly. Claim 2's mechanism reproduces
cleanly — but only once the lens is fitted on a corpus that contains the gate,
and not on the `Mars -> red` example the story is usually told with.**
""")

md(r"""
## 1. A tangent is not a chord

Before any transformer. Given a map $F$ and a cloud of inputs, there are two
natural ways to summarise $F$ with one matrix:

* the **Jacobian** $\partial F/\partial h$ at a point — what $F$ does to an
  infinitesimal nudge;
* the **least-squares fit**
  $\arg\min_{A,b}\mathbb{E}\lVert F(h)-(Ah+b)\rVert^2$ over the cloud — what $F$
  does on average across the whole distribution.

The fact that makes Claim 2 testable at all:

> If $F$ is affine, $F(h) = Mh + c$, its Jacobian is $M$ everywhere **and** least
> squares recovers $M$ exactly. The two coincide. **Any** gap between them is a
> measurement of nonlinearity.
""")

code(r"""
import numpy as np

def affine(points):
    return points @ np.array([[1.0, 0.6], [-0.3, 0.9]]).T

def gated(points):
    "Coordinate 1 leaks into coordinate 2, but only once it clears a threshold."
    first, second = points[:, 0], points[:, 1]
    gate = 1.0 / (1.0 + np.exp(-6.0 * (first - 0.5)))
    return np.stack([first, 0.9 * second + 2.0 * gate], axis=1)

def jacobian_at(function, point, epsilon=1e-4):
    columns = []
    for axis in range(point.shape[0]):
        step = np.zeros_like(point); step[axis] = epsilon
        columns.append(
            (function((point + step)[None])[0] - function((point - step)[None])[0]) / (2 * epsilon)
        )
    return np.stack(columns, axis=1)

def least_squares_map(function, cloud):
    outputs = function(cloud)
    return np.linalg.lstsq(cloud - cloud.mean(0), outputs - outputs.mean(0), rcond=None)[0].T

cloud = np.random.default_rng(0).normal(0, 0.6, size=(4000, 2))
base = np.zeros(2)

for name, function in (("affine", affine), ("gated ", gated)):
    gap = np.abs(jacobian_at(function, base) - least_squares_map(function, cloud)).max()
    print(f"{name}: max |J - A| = {gap:.2e}")
""")

md(r"""
Machine precision for the affine map, a large gap once a gate is added. Section
5 runs exactly this comparison on Gemma instead of a $2\times2$ matrix.
""")

code(r"""
import sys
sys.path.insert(0, "experiments")
import toy_jacobian
toy_jacobian.main()
from IPython.display import Image
Image("experiments/fig_toy_jacobian.png")
""")

md(r"""
## 2. The model, and where A, the circuit, and B sit

**Gemma 3 12B IT** — instruction-tuned, 48 blocks, $d_{\text{model}}=3840$,
vocabulary 262144. The prompt is Anthropic's released multihop item:

> `Fact: The color of the planet fourth from the Sun is`

Nothing in the surface text mentions Mars, so to answer `red` the model must
compute a bridge entity. Writing the remaining computation after layer $\ell$ as
$F_\ell : h_\ell \mapsto h_{\text{final}}$:

```
tokens ──► … ──► h_ℓ ──────── F_ℓ ────────► h_final ──► W_U ──► "red"
                  ▲
                  │
            A = "Mars" lives here        B = "red" appears here
            (never written, never said)
```

The three lenses are three linear approximations to that same $F_\ell$:

| lens | approximation | fitted how |
|---|---|---|
| Logit | $h \mapsto h$ | not fitted |
| J-lens | $h \mapsto J_\ell h$ | $J_\ell=\mathbb{E}[\partial F_\ell/\partial h]$ over a corpus |
| Tuned | $h \mapsto h + W_\ell h + b_\ell$ | distilled against the model's own output |
""")

code(r"""
import os, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

import jlens_core as jc

MODEL = "google/gemma-3-12b-it"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
hf_model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
)
model = jc.wrap(hf_model, tokenizer)
print(f"{model.n_layers} blocks, d_model = {model.d_model}, vocab = {model.lm_head.weight.shape[0]}")

PROMPT = "Fact: The color of the planet fourth from the Sun is "
encoded = tokenizer(PROMPT, return_tensors="pt").to(model.device)
with torch.inference_mode():
    generated = hf_model.generate(**encoded, max_new_tokens=6, do_sample=False,
                                  pad_token_id=tokenizer.eos_token_id)
print("model answers:", repr(tokenizer.decode(generated[0, encoded.input_ids.shape[1]:])))
""")

md(r"""
### A plumbing detail that silently corrupts results

`output_hidden_states=True` does **not** give you the final residual stream.
HuggingFace decoder stacks apply the final norm *before* appending the last
entry, so `hidden_states[-1]` is post-norm and un-embedding it normalises twice.
Anything fitted against it — a Tuned Lens target, a logit-lens baseline — is
fitted against the wrong thing. `jc.residual_stack` takes the last entry from a
forward hook instead. The assertion is the check.
""")

code(r"""
stack = jc.residual_stack(model, encoded.input_ids)   # [n_layers + 1, batch, seq, d_model]
with torch.inference_mode():
    outputs = hf_model(encoded.input_ids, output_hidden_states=True, use_cache=False)
true_logits = outputs.logits.float()

naive = model.unembed(outputs.hidden_states[-1])          # the tempting, wrong way
correct = model.unembed(stack[model.n_layers])            # via the forward hook
print("unembed(hidden_states[-1]) reproduces the logits? ",
      torch.allclose(naive, true_logits, atol=1e-2),
      f"  (max abs diff {(naive - true_logits).abs().max():.2f})")
print("unembed(residual_stack[n_layers]) reproduces them?",
      torch.allclose(correct, true_logits, atol=1e-2),
      f"  (max abs diff {(correct - true_logits).abs().max():.2f})")
""")

md(r"""
## 3. The Jacobian lens in thirty lines

$$
J_\ell \;=\; \mathbb{E}_{\text{prompt}}\;\mathbb{E}_{t}\Big[\textstyle\sum_{t' \ge t}
\frac{\partial h_{\text{final},t'}}{\partial h_{\ell,t}}\Big],
\qquad
\text{lens}_\ell(h) = \operatorname{softmax}\big(W_U\,\mathrm{norm}(J_\ell h)\big)
$$

Fitting $J_\ell$ as a $3840\times3840$ matrix costs hundreds of backward passes
per prompt. We never need the matrix — only its action $J_\ell h$ on the
activation we want to read, which is a **Jacobian-vector product**: one
forward-mode pass per corpus sequence.

The trick that makes it exact: perturb $h_\ell$ by $h$ at *every* valid source
position at once. Causality means the tangent arriving at target position $t'$
collects contributions only from $t \le t'$, so summing the tangent over target
positions gives $\sum_t\sum_{t' \ge t}$ — the estimator — in a single pass.

Positions 0–15 are dropped: attention sinks have atypical residual statistics,
and the reference implementation excludes them. Getting this wrong is not
cosmetic — reading the lens at a single source position inside the sink band
makes `Mars` unrecoverable entirely.
""")

code(r"""
import inspect
print(inspect.getsource(jc.transport))
""")

md(r"""
Two checks, both decisive. The first is the strongest test in the whole
pipeline: at the last block $F$ is the identity, so $J$ must be too.
""")

code(r"""
from datasets import load_dataset
dataset = load_dataset("NeelNanda/pile-10k", split="train")
corpus = jc.corpus_batches(tokenizer, dataset["text"], n_sequences=32, seq_len=128, batch_size=8)

probe = torch.randn(model.d_model, device=model.device)
identity_check = jc.transport(model, model.n_layers - 1, probe, corpus[:1])[0]
cosine = torch.nn.functional.cosine_similarity(identity_check, probe, dim=0)
print(f"J[last] d vs d:  cos = {cosine:.7f},  norm ratio = {identity_check.norm()/probe.norm():.5f}")

dims = [3, 17, 555, 900]
basis = torch.zeros(len(dims), model.d_model, device=model.device)
for row, dim in enumerate(dims):
    basis[row, dim] = 1.0
rows = jc.transpose_transport(model, 20, basis, corpus[:2])
forward = jc.transport(model, 20, probe, corpus[:2])[0]
print("reverse mode (J^T e_i)·d:", [f"{(rows[i] @ probe).item():+.4f}" for i in range(len(dims))])
print("forward mode (J d)_i    :", [f"{forward[d].item():+.4f}" for d in dims])
""")

md(r"""
## 4. Claim 1 — what each lens sees, layer by layer

Read the last prompt token at every layer with the J-lens and the logit lens,
ranking `Mars` and `red` over the **full** vocabulary. A two-token contrast would
be far cheaper but could not support a claim about what the lens is reading.
""")

code(r"""
def single_token(word):
    ids = tokenizer.encode(word, add_special_tokens=False)
    assert len(ids) == 1, f"{word!r} is not a single token: {ids}"
    return ids[0]

targets = {"Mars": single_token("Mars"), "red": single_token("red")}
readout = stack[:, 0, -1].float()

rows = []
for layer in range(model.n_layers):
    hidden = readout[layer + 1]
    j_scores = model.unembed(jc.transport(model, layer, hidden, corpus)[0]).squeeze()
    logit_scores = model.unembed(hidden).squeeze()
    rows.append({
        "layer": layer,
        **{f"j_{w}": jc.rank_of(j_scores, i) for w, i in targets.items()},
        **{f"logit_{w}": jc.rank_of(logit_scores, i) for w, i in targets.items()},
        "j_top": jc.top_tokens(tokenizer, j_scores, 4),
    })

for row in rows[::4] + [rows[34]]:
    print(f"L{row['layer']:02d}  Mars: J {row['j_Mars']:>6} / logit {row['logit_Mars']:>6}"
          f"   J top: {row['j_top']}")
""")

md(r"""
The J-lens walks through the computation: `color` in the teens, `pigments` and
`颜色` in the twenties, `Earth`/`planet` around 30, then **`Mars` at rank 1 at
layer 34**, then `red`. That is Figure 51's `color → Mars → red` sequence in an
open model.

The part that is hardest to explain away is the right-hand panel below. While
`Mars` climbs to rank 1, the J-lens rank of `red` is simultaneously getting
*worse*, peaking around $10^5$ near layer 26. The J-lens is not merely noisy
about the answer — it is reading the bridge entity **instead of** it.
""")

code(r"""
import matplotlib.pyplot as plt
import jlens_plot as jp
jp.use_style()

figure, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
layers = [r["layer"] for r in rows]
for axis, word, title in ((axes[0], "Mars", "bridge entity 'Mars'"), (axes[1], "red", "answer 'red'")):
    for lens in ("j", "logit"):
        axis.plot(layers, [r[f"{lens}_{word}"] for r in rows],
                  color=jp.LENS_COLORS[lens], label=jp.LENS_LABELS[lens])
    axis.set_yscale("log"); axis.invert_yaxis(); axis.set_xlabel("layer")
    axis.set_title(title); axis.grid(axis="y", alpha=0.7)
    axis.axhline(1, color=jp.MUTED, lw=0.8, ls=":")
axes[0].set_ylabel("full-vocabulary rank")
axes[1].legend(loc="lower left")
plt.show()
""")

md(r"""
### At scale, Claim 1 only partly holds

`experiments/screen_multihop_lenses.py` runs all three lenses over Anthropic's
90-item `probe-swap` set (63 items survive the behavioural and single-token
filters). Its verdict, which we report as measured rather than as hoped:

| lens | pass@1 | pass@5 | pass@10 | median best rank |
|---|---|---|---|---|
| **J-lens** | **0.60** | 0.84 | 0.87 | **1** |
| Logit | 0.46 | **0.87** | **0.89** | 2 |
| Tuned | 0.24 | 0.63 | 0.68 | 3 |

The J-lens is clearly best at driving the intermediate to **rank 1**, and best
on median rank. But the logit lens edges it at pass@5 and pass@10, so the
preregistered criterion — written on pass@10 — **fails**. We record that rather
than switching to the metric that passes. It is consistent with the paper, which
says its own margin over the logit lens is *"modest on multihop"* and
substantial elsewhere; multihop is simply the distribution with released causal
counterfactuals.

The skip-ahead half does hold: at the layer where the J-lens best recovers A, it
ranks A above B on 75% of items, while the Tuned Lens ranks B above A on 57%
(median rank 7 for B against 13 for A).
""")

md(r"""
## 5. Claim 2, part one — is $F_\ell$ actually nonlinear?

Now the measurement that section 1 set up. $J_\ell$ and the least-squares map
$A_\ell$ are **equal whenever $F_\ell$ is affine**, so $\cos(J_\ell d, A_\ell d)$
on natural directions measures nonlinearity directly. It needs a floor: how well
does $J_\ell$ agree with an independent estimate of *itself*? Split the corpus in
half.

We fit the least-squares lens here in closed form — it is ordinary least
squares, so there is no optimiser and no learning rate to argue about.
""")

code(r"""
import torch.nn.functional as F

LAYERS = [8, 20, 28, 34, 40, 44, 46, 47]

# activations to fit the regression on, and directions to probe the maps with
fit_batches = jc.corpus_batches(tokenizer, dataset["text"], n_sequences=96,
                                seq_len=128, batch_size=8, seed=11)
pool = []
for input_ids in fit_batches:
    positions = jc.valid_positions(input_ids.shape[1]).to(model.device)
    pool.append(jc.residual_stack(model, input_ids)[:, :, positions].flatten(1, 2))
pool = torch.cat(pool, dim=1)
print("fitting on", pool.shape[1], "token positions")

generator = torch.Generator().manual_seed(3)
left = torch.randint(0, pool.shape[1], (16,), generator=generator)
right = torch.randint(0, pool.shape[1], (16,), generator=generator)

target = pool[model.n_layers].float()
centred_y = (target - target.mean(0)).double()
half = len(corpus) // 2

report = []
for layer in LAYERS:
    source = pool[layer + 1].float()
    centred_x = (source - source.mean(0)).double()
    cov_xx = centred_x.T @ centred_x / centred_x.shape[0]
    cov_yx = centred_y.T @ centred_x / centred_x.shape[0]
    ridge = 1e-2 * torch.diagonal(cov_xx).mean()
    regression = torch.linalg.solve(
        cov_xx + ridge * torch.eye(model.d_model, dtype=torch.float64, device=model.device),
        cov_yx.T,
    ).T.float()

    directions = (source[left.to(source.device)] - source[right.to(source.device)])
    directions = directions / directions.norm(dim=1, keepdim=True)
    first = jc.transport(model, layer, directions, corpus[:half])
    second = jc.transport(model, layer, directions, corpus[half:])
    full = 0.5 * (first + second)
    report.append({
        "layer": layer,
        "cos_J_vs_regression": float(F.cosine_similarity(full, directions @ regression.T, dim=1).mean()),
        "cos_J_splithalf": float(F.cosine_similarity(first, second, dim=1).mean()),
    })
    print(f"L{layer:02d}  cos(J, A) = {report[-1]['cos_J_vs_regression']:+.3f}"
          f"   split-half ceiling = {report[-1]['cos_J_splithalf']:+.3f}")
""")

md(r"""
Layer 47 comes out at **exactly 1.000**: there $F$ *is* the identity, so
$J = A = I$ is forced, and hitting it requires the Jacobian estimator, the OLS
solve and the residual convention to all be simultaneously correct.

From there the gap grows monotonically as you go earlier. At layer 34 — where
the J-lens reads `Mars` at rank 1 — $J$ agrees with an independent estimate of
itself at **0.99** but with the regression at only **0.65**. The divergence
between the two lenses at that depth is not an estimation artefact.

The full 48-layer version is `experiments/nonlinearity_vs_divergence.py`:
""")

code(r"""
Image("experiments/fig_nonlinearity.png")
""")

md(r"""
## 6. Claim 2, part two — but is there a *gate*?

Part one shows $F_\ell$ is nonlinear **as a map**, averaged over the corpus and
over all input directions. Neel's mechanism sketch says something narrower: that
a gate-like circuit converts `Mars` into `red`, and *that* is what the two
lenses disagree about. A function can be strongly nonlinear globally and almost
perfectly linear along one particular chord, so this needs its own test.

Interpolate between two states the model **actually produces**:

$$h(\alpha) = (1-\alpha)\,h_{\text{fourth planet}} + \alpha\,h_{\text{third planet}}$$

at one layer and one position, every other position from the first prompt's
forward pass. Both endpoints are on-distribution, and the size of the
intervention is set by the natural distance between two real model states rather
than chosen by us.

(An earlier attempt used the paper's lens-coordinate swap. It moved the
activation by 0.4% of its norm, produced a linear response, and failed to
redirect the output — which tests nothing, because a perturbation that small
cannot reach a gate. That negative is recorded in `experiments/LOG.md`.)
""")

code(r"""
PAIRS = [
    ("planet-color", "Fact: The color of the planet fourth from the Sun is ", "red",
     "Fact: The color of the planet third from the Sun is ", "blue"),
    ("legs-spider-dog", "Fact: The number of legs on the animal that spins webs is ", "8",
     "Fact: The number of legs on the animal that barks and fetches sticks is ", "4"),
]
alphas = torch.linspace(0, 1, 61)

def response_curve(base, counter, answer, counter_answer, layer):
    enc_a = tokenizer(base, return_tensors="pt").to(model.device)
    enc_b = tokenizer(counter, return_tensors="pt").to(model.device)
    start = jc.residual_stack(model, enc_a.input_ids)[layer + 1][0, -1].float()
    end = jc.residual_stack(model, enc_b.input_ids)[layer + 1][0, -1].float()
    grid = alphas.to(model.device)[:, None]
    path = (1 - grid) * start + grid * end
    position = enc_a.input_ids.shape[1] - 1

    captured = {}
    def inject(_m, _a, out):
        hidden = jc._block_output(out); edited = hidden.clone()
        edited[:, position] = chunk.to(hidden.dtype)
        return jc._rewrap(out, edited)
    def capture(_m, _a, out):
        captured.setdefault("h", []).append(jc._block_output(out)[:, position].float())

    for chunk in path.split(16):
        handles = [model.layers[layer].register_forward_hook(inject),
                   model.layers[-1].register_forward_hook(capture)]
        try:
            with torch.inference_mode():
                model.forward(enc_a.input_ids.expand(chunk.shape[0], -1))
        finally:
            for h in handles: h.remove()
    logits = model.unembed(torch.cat(captured["h"]))
    return (logits[:, single_token(counter_answer)] - logits[:, single_token(answer)]).cpu().numpy()

def width_10_90(response):
    low, high = response[0], response[-1]
    scaled = (response - low) / (high - low)
    inside = alphas.numpy()[(scaled >= 0.1) & (scaled <= 0.9)]
    return float(inside.max() - inside.min())

for name, base, answer, counter, counter_answer in PAIRS:
    for layer, role in ((34, "workspace"), (46, "control  ")):
        curve = response_curve(base, counter, answer, counter_answer, layer)
        print(f"{name:18s} L{layer} {role}  10-90 width = {width_10_90(curve):.3f}"
              f"   logit diff {curve[0]:+.2f} -> {curve[-1]:+.2f}")
""")

md(r"""
`legs-spider-dog` is a clean gate: flat, a sharp rise near $\alpha \approx 0.5$,
flat again, with a 10–90% width around 0.22. `planet-color` is not — its
response is very nearly a straight line, width around 0.83.

**So the Mars example, the one this story is usually told with, has no
detectable gate.** Its interpolation is linear to $R^2 = 0.995$ at an
intervention scale thirty times larger than the swap's.

Running this as a paired test across items
(`experiments/shape_paired_test.py`, 21 candidate pairs, 7 surviving the
behavioural filter) gives the transition sharper at the workspace layer on 6 of
7 items, median width 0.32 against 0.73, exact sign-flip permutation test
$p = 0.016$. And the items split by *answer type*:

| answer type | items | workspace width |
|---|---|---|
| numeric | legs ×3, sides-of-triangle, moons-of-planet | 0.22 – 0.45 |
| categorical | planet-colour, capital-language | 0.68, 0.83 |

Every numeric-answer item gates; neither categorical one does — and
`planet-color` is the single item whose transition is *less* sharp at the
workspace layer than at the control.
""")

code(r"""
Image("experiments/fig_shape_paired.png")
""")

md(r"""
## 7. Where the lenses' behaviour actually comes from

Section 6 left a puzzle. The downstream computation gates hard — the tangent at
$\alpha=0$ is 4,315 and the chord over $[0,1]$ is 40,911, a factor of nine —
yet all three deployed lenses have nearly the same shallow slope along that
path. So the tuned lens is not acting as the chord of this curve.

The toy in section 1 gets the tangent/chord separation because its fitting
corpus is two clusters **straddling the gate along the swept axis**. The
deployed lenses are fitted on web text, which never explores the
spider-versus-dog direction. So the missing separation may be a fact about the
fitting distribution rather than about $F_\ell$.

That is testable: refit both lenses, same estimators, on a corpus of two-hop
animal-property prompts instead of web text.
""")

code(r"""
%run experiments/in_domain_lenses.py
""")

code(r"""
Image("experiments/fig_in_domain.png")
""")

md(r"""
| corpus | J-lens slope | tuned slope | tuned / J |
|---|---|---|---|
| web text | 11,013 | 9,622 | 0.87 |
| **in-domain** | **26,570** | **41,564** | **1.56** |

against a tangent of 4,315 and a chord of **40,911**.

Refit where the gate lives, the regression lands within **1.6%** of the chord.
That is Neel's mechanism reproduced in a real model, and it says the flat result
above was a fact about the corpus, not about the computation.

Two things complicate the happy reading.

**The J-lens moves too**, from 11,013 to 26,570 — about 65% of the chord, not
the tangent. The toy keeps its average tangent small by clustering corpus mass
*away* from the gate; the animal corpus spreads across the leg-count axis and
puts plenty of mass *on* it. Separation needs the corpus to be in-domain **and**
to avoid the gate. In-domain alone is not enough.

**And fitting in-domain costs the J-lens the thing it is for.** Full-vocabulary
ranks at the unmodified spider state:

| lens | corpus | `spider` | `8` |
|---|---|---|---|
| J | web | **7** | 638 |
| J | in-domain | 14 | **53** |

A ~90x bridge-over-answer rank gap collapses to ~4x, and the lens starts
reporting the answer. This is consistent with the account rather than against
it: the Jacobian is meant to ask what the model would say *on an arbitrary
prompt*, and averaging it over a task-specific corpus reintroduces exactly the
predictive correlations that genericity strips out.

So there is a real trade-off, and no single fit gives both:

* **generic corpus** — the J-lens reports current content, Claim 1 works, and
  neither lens shows the tangent/chord geometry;
* **in-domain corpus** — the geometry appears cleanly, Claim 2 works, and the
  J-lens loses its readout specificity.
""")

md(r"""
## 8. One knob instead of two lenses

Sections 5–7 treat the J-lens and the Tuned Lens as two methods to be compared.
They are better understood as **two ends of one continuum**. Define the best
linear map at intervention scale $\sigma$:

$$
A_{\ell,\sigma} \;=\; \arg\min_A \;
\mathbb{E}_{h,\;\delta\sim q_\sigma}
\big\lVert F_\ell(h+\delta) - F_\ell(h) - A\delta \big\rVert^2
$$

As $\sigma\to0$ this is the J-lens, $\mathbb{E}_h[J_\ell(h)]$. With
$\delta = h' - h$ for independent natural states it is the least-squares
tuned-lens analogue $\operatorname{Cov}(F(h),h)\operatorname{Cov}(h)^{-1}$.

**And the whole family costs what the J-lens costs.** By the fundamental theorem
of calculus $F(h+\delta)-F(h) = \bar J(h,\delta)\,\delta$ exactly, with
$\bar J(h,\delta)=\int_0^1 J(h+\alpha\delta)\,d\alpha$. For isotropic
$\delta$ the minimiser is $\mathbb{E}[\bar J]$, and

$$
\mathbb{E}_{h,\delta}\big[\bar J(h,\delta)\big]
= \mathbb{E}_{h,\delta,\alpha}\big[J(h+\alpha\delta)\big]
= \mathbb{E}_{h'\sim p_\sigma}\big[J(h')\big]
$$

where $p_\sigma$ is the activation distribution **smeared at scale $\sigma$**.
So the scale-$\sigma$ lens is just the existing Jacobian estimator run over
perturbed activations — one extra term in the injection hook, no extra forward
passes, and $\sigma=0$ reduces to the estimator we already validated.
""")

code(r"""
%run experiments/integrated_lens.py
""")

code(r"""
Image("experiments/fig_integrated_lens.png")
""")

md(r"""
At $\sigma=0$ the lens reads the **bridge entity** and is nearly blind to the
answer — `8` sits at rank 676 for the spider prompt. By $\sigma\approx0.35$–$0.5$
the readout has **flipped**: the answer is top-10 and the bridge has faded. The
crossover happens at a consistent scale on all three probes, and past
$\sigma\approx0.75$ both degrade as the smearing starts averaging Jacobians over
states the model never visits.

So "reads current content" versus "skips to the output" is not a property of two
different methods. It is a property of **how far you push the representation
before looking**.

We also tried the obvious alternative — making the lens itself nonlinear, as
$A h + b + \sum_k v_k\,\sigma(w_k\!\cdot\!h + c_k)$ with $K=64$ gated units
fitted to the linear lens's residual. It fails: held-out $R^2$ is slightly
*worse* than the plain linear lens on both a web corpus (0.459 vs 0.463) and a
counting-domain corpus (0.898 vs 0.900), and the units decode as noise
(`experiments/gated_lens.py`).

The diagnosis is worth keeping. A gate is one direction, on one relation, in one
context. Fitted capacity goes to whatever explains the most *global* variance,
and that is never the spider-legs threshold. The integrated lens works because
it never has to **find** the gate — sweeping $\sigma$ traverses it by
construction. Probing at a chosen scale beats fitting nonlinearity into a
general-purpose lens.
""")

md(r"""
## 9. Which computations gate at all?

Not many, it turns out. Interpolating between length-matched prompt pairs and
requiring the intervention to be *causally effective* — swinging
$\text{logit}(B')-\text{logit}(B)$ by at least 4 nats **and** flipping its sign —
gives 10 scorable items (`experiments/effective_gate_test.py`):

```
numeric     : 0.250  0.333  0.333  0.350  0.433     (legs, polygon sides,
non-numeric : 0.550  0.583  0.633  0.667  0.800      basketball players, wheels)
```

Perfectly separated, exact permutation $p=0.024$, and numeric items sharpen
3–4× more against their own late-layer control. Antonyms and biological class
have small closed answer spaces and still behave like colour, so what gates is
**counting**, not small answer sets in general.

This precondition matters more than it sounds. Without it, three of four
non-numeric items in an earlier pass had swings under 8 nats that never flipped
the sign — flat responses that measured noise, not the model. That earlier
conclusion was withdrawn; `experiments/LOG.md` records it.
""")

code(r"""
Image("experiments/fig_effective_gate.png")
""")

md(r"""
## 10. Verdict

| claim | verdict |
|---|---|
| J-lens surfaces the bridge entity at a layer where the model has not said it | **holds** — `Mars` at full-vocabulary rank 1 (L34); `spider` at rank 7 against `8` at 638 |
| J-lens beats logit and tuned lenses at recovering it | **partly** — best at pass@1 (0.60 vs 0.46 vs 0.24) and median rank; logit lens edges it at pass@5/10, so the preregistered criterion fails |
| Tuned Lens skips ahead to the answer | **holds** — ranks B above A on 57% of items where the J-lens ranks A above B on 75% |
| $F_\ell$ is nonlinear, so a local Jacobian and a global regression *can* differ | **holds** — $\cos(J,A)=0.65$ at L34 against a 0.99 noise ceiling, and exactly 1.000 at L47 |
| the tuned lens behaves as a regression chord across a gate | **holds, conditionally** — within 1.6% of the chord when fitted on a corpus that straddles the gate; not at all on web text |
| a gate-like circuit converting `Mars` to `red` is what drives the difference | **fails** — that response is linear to $R^2=0.995$; the gating example is `spider -> 8`, not Mars |
| gate-like transitions exist in the workspace band generally | **holds, for counting** — 5 numeric items at widths 0.25–0.43 against 5 non-numeric at 0.55–0.80, perfectly separated, $p=0.024$ |
| the two lenses are one continuum in intervention scale | **holds** — sweeping $\\sigma$ flips the readout from bridge to answer on all three probes at a consistent scale |
| a nonlinear lens fitted on a corpus captures the gate | **fails** — 64 gated units are beaten by the plain linear lens on held-out data in both corpora |

Neel's explanation splits into a premise and a mechanism, and both survive, but
with conditions worth stating.

The **premise** — that $F_\ell$ is nonlinear enough for a local Jacobian and a
global regression to summarise it differently — is confirmed by a measurement
that is exact rather than suggestive, since $J_\ell = A_\ell$ is *forced* when
$F_\ell$ is affine, and the measurement returns exactly 1.000 at the layer
where that holds.

The **mechanism** — regression-as-chord, Jacobian-as-tangent — reproduces
cleanly, but only once the lens is fitted where the gate is. On the corpus the
lenses are actually deployed with, the geometry is absent. And the example the
story is usually told with, `Mars -> red`, has no gate at all: it is linear to
$R^2 = 0.995$ under an intervention thirty times larger than the paper's swap.
The project brief anticipated exactly this and asked that it not be presented as
Claim 2 evidence, so it is not.

Full chronology, including three implementation bugs and every criterion that
failed, is in `experiments/LOG.md`; the four preregistrations are in
`experiments/CLAIMS.md`.
""")

# ---------------------------------------------------------------------------


def main() -> None:
    notebook = {
        "cells": [
            {
                "cell_type": kind,
                "id": f"cell-{index:02d}",
                "metadata": {},
                "source": source.splitlines(keepends=True),
                **({"outputs": [], "execution_count": None} if kind == "code" else {}),
            }
            for index, (kind, source) in enumerate(cells)
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NOTEBOOK.write_text(json.dumps(notebook, indent=1))
    print("wrote", NOTEBOOK, f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
