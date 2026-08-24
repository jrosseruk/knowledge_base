# Lab log

Chronological record, including the negative results and the reasons for every
methodological change. Entries are appended, never rewritten.

## 2026-08-23, session 1 (prior work)

Explored Pythia basketball/LeBron, Gemma poetry, and a Gemma ordinal-gate
interpolation. Trained Tuned Lenses at layers 24/28/32/36 of Gemma 3 12B IT.

**Negative result.** On the exact multihop prompt
`Fact: The color of the planet fourth from the Sun is`, the average J-Lens
failed to recover `Mars` at any of the four probed layers. At L36:
J-Lens `Mars` rank 3,570; `red` rank 22,244. Recorded in
`gemma_mars_three_lens_screen.json`.

## 2026-08-23, session 2

### Reading the reference estimator

Read `jacobian-lens/jlens/fitting.py` (Anthropic's reference implementation,
Apache-2.0) and the paper's Methodological Details section. The session-1
estimator differed from the published recipe in three ways:

1. It injected the probe direction at **source position 8**. The reference sets
   `SKIP_FIRST_N_POSITIONS = 16` and excludes those positions explicitly,
   because early positions act as attention sinks and have atypical
   residual-stream statistics.
2. It averaged over a **single** source position rather than over all valid
   source positions of each corpus sequence.
3. It read out at four hand-picked layers. The paper's pass@k statistic is a
   **min over all layers**.

These three changes were decided by reading the reference spec, *before*
re-running anything. That ordering matters: the fix is not a response to the
size of the resulting number.

### Reimplementation (`jlens_core.py`)

Fitting the full `d_model x d_model` Jacobian costs `d_model / dim_batch`
backward passes per prompt. We only ever need `J_l h` (readout) and `J_l^T u`
(lens vector for one token), so both are computed directly:

* `J_l h` -- one **JVP** per corpus sequence. Perturbing `h_l[t]` by `h` at
  every valid source position at once and summing the tangent over valid
  target positions gives `sum_t sum_{t' >= t} (dh_final[t']/dh_l[t]) h`
  exactly, by causality.
* `J_l^T u` -- one **VJP**: cotangent `u` at every valid target position,
  gradient averaged over valid source positions.

Validation, both passing:

* transporting from the last block returns the input (cos = 0.9999986,
  norm ratio 1.00003), as it must when `J = I`;
* the forward-mode and reverse-mode paths agree to 6 significant figures on
  Gemma 3 1B (`(J d)_i` computed both ways).

### Corrected Mars pilot (`pilot_mars_all_layers.py`)

64 pile-10k sequences of 128 tokens, all 48 layers, full-vocabulary ranks.

**The session-1 negative does not survive the corrected estimator.** The J-lens
reads out, in layer order:

| layers | J-lens top tokens |
|---|---|
| 12-16 | ` color`, ` colors`, ` colours` |
| 17-22 | ` pigments`, `颜色`, `顏色` |
| 27 | ` asteroids`, ` volcanoes` |
| 30-33 | `Earth`, ` planet` |
| **34** | **`Mars` (rank 1)** |
| 39-40 | ` red`, ` reddish` |
| 43-44 | ` Mars` |
| 45-46 | `red` |

`Mars` reaches full-vocabulary rank 1 under the J-lens at L34, where the logit
lens has it at rank 12. This is the `color -> Mars -> red` sequence of the
paper's Figure 51, in an open 12B model. L47 is a further sanity check: the
J-lens and logit lens ranks are identical there (1152 / 74), as required when
`J` is the identity.

### Disk

`/` was already at 100% from an unrelated 1.1 TB project in the user's home.
Large lens matrices are therefore written to `/dev/shm/jlens-artifacts`
(`JLENS_ARTIFACTS` overrides). Only JSON/PNG artefacts go in the repository.

### Tuned Lens implementation

Checked against `AlignmentResearch/tuned-lens` rather than from memory. The
published lens is an affine translator with an **identity residual**,
`h -> h + W h + b`, `W` and `b` zero-initialised, read out through the model's
own final norm, and trained by **forward-KL distillation** against the model's
final-layer distribution -- not least squares on residuals.

Both objectives are therefore fitted, because Claim 1 and Claim 2 name
different objects:

* `tuned_lens_kl.pt` -- the published Tuned Lens. Claim 1 replicates Figure 51,
  which compares against *this*.
* `tuned_lens_ols.pt` -- `argmin ||h_final - (A h_l + b)||^2`, solved in closed
  form. This is what Neel's review means by "linear regression between residual
  streams", and it is the object Claim 2's argument is about.

### Bug: `output_hidden_states` does not give the final residual

Chasing an anomaly — the KL-trained translator at the last layer should be
exactly `W = 0` (its source *is* its target), but held-out KL there read 0.725,
then 0.624 after a first fix attempt — turned up a bug affecting every script.

HuggingFace decoder stacks append the hidden state to `all_hidden_states`
*after* applying the final norm:

```python
for decoder_layer in self.layers:
    all_hidden_states += (hidden_states,)
    hidden_states = decoder_layer(...)
hidden_states = self.norm(hidden_states)
all_hidden_states += (hidden_states,)          # <- post-norm
```

So `hidden_states[-1]` is **post-final-norm**, and `unembed(hidden_states[-1])`
normalises twice. Verified on Gemma 3 1B:

```
hidden_states[-1] == norm(last block output)?  True
unembed(hidden_states[-1]) == true logits?     False
unembed(pre-norm output)   == true logits?     True   (max abs diff 0.0)
```

Consequences, all now fixed by routing every script through
`jlens_core.residual_stack`, which takes indices `0 .. n_layers-1` from
`output_hidden_states` and index `n_layers` from a forward hook on the last
block:

* **Both Tuned Lenses were fitted against a double-normed target.** The KL lens
  was distilled toward `lm_head(norm(norm(h)))` rather than the model's actual
  output distribution, and the OLS lens regressed onto `norm(h_final)` rather
  than `h_final`. Both refitted.
* **The pilot's L47 row was wrong** (double-normed). Layers 0-46 are unaffected,
  so the `Mars`-at-L34 result does not depend on this; rerun anyway to confirm.
* The J-lens estimator in `jlens_core.transport` was **never** affected — it
  captures the last block's output with a hook and so always used the pre-norm
  residual.

Note that this does *not* by itself explain the last-layer translator drift:
under the old code the source and target at that layer were the same tensor, so
the gradient should still have been exactly zero. `fit_kl` now prints
`|W[last]|` every 25 steps so the drift can be caught in the act rather than
inferred after the fact. If it persists after the refit it is a separate bug and
will be recorded as one.

### Resolved: why the last-layer translator drifts

The live `|W[last]|` diagnostic caught it on the first step:

```
kl step    0/1500  mean KL = 21.5821  |W[last]| = 0.218  (4s)
```

One optimiser step, and a translator whose exact solution is `W = 0` has already
moved. The size is diagnostic. Adam's first update is exactly `±lr` for every
element with a nonzero gradient (bias correction makes `m/sqrt(v) = ±1` at step
one), so `||W||_F = lr * sqrt(k)` gives

```
k = (0.218 / 1e-3)^2 ~ 47,500  of  3840^2 = 14.7M elements  (0.3%)
```

So a small fraction of elements received a nonzero gradient where the analytic
value is exactly zero. The source and target at that layer are the same values,
but they are not computed by the same call: the target goes through `unembed`
under `torch.no_grad()` and the prediction goes through it inside the autograd
graph. Float32 matmul reductions are not bit-reproducible across those two
paths, so `q - p` is a few units in the last place rather than exactly zero.

The reason this is visible at all is **Adam's scale invariance**: it normalises
by the gradient's own magnitude, so a gradient of 1e-7 produces the same
parameter step as a gradient of 1e-1. Rounding noise is promoted to a full-size
update. Plain SGD would have ignored it; this is a hazard specific to adaptive
optimisers applied where the true gradient vanishes.

Consequence for the results: this affects layers where the true gradient is
comparable to the noise floor, i.e. the last few, where the translator should be
near-zero anyway. At the workspace layers the true gradient is orders of
magnitude larger and the effect is negligible. It is recorded here as a known
artefact of the last-layer sanity check rather than silently repaired; the
proper fixes, for anyone rerunning this, are to raise Adam's `eps` above the
noise floor, or simply to skip fitting the final layer whose solution is known
in closed form.

### Claim 2, criterion 2/3: the two maps disagree, far beyond noise

`nonlinearity_vs_divergence.py`, 24 probe directions drawn from natural
centred residual differences, 64 pile-10k sequences split in half for the
noise ceiling. `tuned_lens_ols.pt` is the right comparand here because OLS is
what provably equals the Jacobian when `F_l` is affine.

| layer | `cos(J, A_reg)` | split-half ceiling | gap |
|---|---|---|---|
| 0  | 0.147 | 0.749 | 0.602 |
| 12 | 0.240 | 0.895 | 0.655 |
| 24 | 0.490 | 0.973 | 0.484 |
| **34** | **0.651** | **0.991** | **0.340** |
| 42 | 0.791 | 0.997 | 0.205 |
| 45 | 0.933 | 0.999 | 0.067 |
| 46 | 0.970 | 1.000 | 0.030 |
| **47** | **1.000** | **1.000** | **0.000** |

Three things to note.

1. **L47 is exactly 1.000.** There `F_l` *is* the identity, so `J_l = A_l = I`
   is forced. The measurement returning exactly that is a hard check on the
   whole pipeline — the Jacobian estimator, the OLS solve, and the residual
   convention all have to be right simultaneously for it to come out.
2. **The gap is monotone in depth**, closing smoothly from 0.60 to 0.00. The
   nonlinearity of the remaining computation decreases as there is less
   computation remaining, which is the only sensible shape.
3. **At the workspace layers the gap is 10-20x the noise floor.** At L34 —
   the layer where the J-lens reads `Mars` at rank 1 — J agrees with an
   independent estimate of itself at 0.991 but with the regression at 0.651.
   The divergence between the two lenses at that depth is not an estimation
   artefact.

This establishes criterion 2. Criterion 3 (that the readout divergence tracks
this curve) needs the layerwise screen, which is still running.

### Confirmation of the Adam diagnosis

The refit, which now stops at L46, gives the control that the earlier run
lacked. After a single optimiser step:

```
old run, last fitted layer 47:   |W[L47]| = 0.218
new run, last fitted layer 46:   |W[L46]| = 3.838
```

Adam's first update is exactly `±lr` per element with a nonzero gradient, so a
full-size step on every element of a 3840x3840 matrix gives
`3840 * 1e-3 = 3.84`. L46 lands on that number: its gradient is real and dense.
L47 gave 0.218, i.e. `sqrt(k) * lr` with k ~ 0.3% of elements — the signature of
float32 rounding residue on a gradient whose true value is zero. The two cases
are cleanly separated by three orders of magnitude in element count, which
confirms the diagnosis rather than leaving it as a plausible story.

### Tuned Lens refit, corrected

`--objective kl`, 600 steps, layers 0-46 fitted and L47 set analytically.
Held-out KL against the model's true output distribution:

| layer | tuned | logit |
|---|---|---|
| 0  | 6.600 | 99.403 |
| 8  | 4.321 | 32.765 |
| 16 | 3.523 | 11.579 |
| 24 | 3.109 |  7.485 |
| 34 | 1.693 |  4.205 |
| 44 | 0.489 |  1.857 |
| 46 | 0.386 |  0.279 |
| 47 | 0.000 |  0.000 |

L47 is exactly zero, as the analytic fix requires. The lens beats the logit
lens at every layer except L46, where it is slightly worse (0.386 vs 0.279):
600 steps is not enough to converge a translator whose optimum is very close to
the identity. Two caveats to carry into the writeup:

* the lens is somewhat undertrained relative to the published recipe. For
  Claim 1 this is conservative — an undertrained Tuned Lens should skip ahead to
  the output *less*, not more, so it biases against the effect we are looking
  for rather than manufacturing it.
* the OLS lens (fitted in closed form, no such caveat) is the one used for the
  Claim 2 map comparison, so that result does not inherit this limitation.

### Claim 1 at scale: partial replication, one preregistered criterion fails

`screen_multihop_lenses.py` over Anthropic's 90-item `probe-swap` set. 63 items
pass the behavioural filter (model answers correctly) and the single-token
requirement on both A and B. Full-vocabulary ranks, min over all 48 layers.

| lens | pass@1 | pass@5 | pass@10 | median best A rank |
|---|---|---|---|---|
| J-lens | **0.60** | 0.84 | 0.87 | **1** |
| Logit  | 0.46 | **0.87** | **0.89** | 2 |
| Tuned  | 0.24 | 0.63 | 0.68 | 3 |

**The preregistered criterion "J-lens recovers A better than logit and tuned"
was written on pass@10 and FAILS**: 0.873 for the J-lens against 0.889 for the
logit lens. It is recorded as a failure. The metric is not being switched to
the one that passes.

What is true is narrower and should be stated that way: the J-lens is the best
of the three at getting the bridge entity to **rank 1** (0.60 vs 0.46 vs 0.24)
and has the best median rank, but the logit lens is marginally better at
getting it into the top 5 or top 10.

This is consistent with the paper's own characterisation. It reports that the
J-lens margin over the logit lens is "modest on multihop and association, but
substantial on multilingual, order-of-operations, poetry, and typo", and that
"the logit lens ... captures much of the same workspace content as the J-lens
in the layers where it does work". Multihop is the distribution where the paper
expects the smallest gap, and we chose it because it is the one with released
causal counterfactuals, not because it maximises the effect. The Tuned Lens
trailing both also matches the paper.

The skip-ahead criterion **passes**. At the layer where the J-lens best recovers
A:

* J-lens ranks A above B on 75% of items;
* Tuned Lens ranks B above A on 57% of items;
* median Tuned rank of B is 7 against 13 for A.

So at the depth where the J-lens is reading the intermediate, the Tuned Lens is
already reporting the answer.

### Claim 2, causal sweep: NEGATIVE on the preregistered criteria

`causal_sweep.py`, single layer, single position (the readout token), lens-
coordinate swap, alpha in [0, 1] for the criteria and swept to 2.0 for shape.
Pre-switch layer taken from the Claim 1 screen (the layer at which the J-lens
best recovers A), chosen before any causal measurement.

| item | layer | R^2 linear | gate index | top token at alpha=1 |
|---|---|---|---|---|
| mars-color | 34 | 0.895 | 0.13 | `\n` (unchanged) |
| ex-planet-color-third-fourth | 30 | 0.989 | 0.09 | ` blue` (unchanged) |
| planet-rings-saturn2 | 45 | 0.983 | 0.32 | ` spectacular` (unchanged) |

* **Criterion 1 (nonlinearity) FAILS.** Required R^2 < 0.9 and gate index > 0.5;
  measured R^2 0.90-0.99 and gate index 0.09-0.32. The model's response to this
  intervention is close to linear.
* **Criterion 4 (causal redirection) FAILS.** The bridge swap never moves the
  top output token to the counterfactual answer.
* **Criterion 5 (timing) FAILS**, and in the direction opposite to the
  prediction: the answer swap moves the output far more than the bridge swap
  (4.06 vs 0.38 nats on mars-color).

The proximate reason for 1 and 4 is visible in the diagnostics: the swap moves
the activation by `|delta|/|h| = 0.004`. A 0.4% perturbation does not leave the
linear regime, so a linear response is what should be expected and no
conclusion about gating can be drawn from it either way.

**A design flaw in criterion 5, stated plainly.** The "answer swap" writes along
the J-lens vectors of `red` and `blue`, which at these depths are close to the
corresponding unembedding directions. Adding a component along
`v_B' - v_B` therefore raises `logit(B') - logit(B)` almost mechanically,
whether or not B is yet represented. Criterion 5 as preregistered is confounded
and its failure is not clean evidence about the model. A sound version of the
timing control would need an intervention on B that is not collinear with the
readout direction used to score it. This is recorded as a flaw in the
experiment, not as a finding about Gemma.

Next: the paper applies its swap at every prompt position across a band of
layers, not at a single position. Running that variant (`--sites all`, still a
single layer so that `F_l` remains a function of one perturbation path) to see
whether a larger, more faithful intervention redirects the output. Both are
reported; whichever is stronger, the single-position result above stands as
recorded.

### All-positions swap: also negative

`causal_sweep.py --sites all`, the variant closer to the paper's protocol
(swap written to every prompt position), still at a single layer so that the
perturbed object remains one path.

| item | layer | logit(B') - logit(B) at alpha 0 -> 1 | top token at alpha=1 |
|---|---|---|---|
| mars-color | 34 | -1.12 -> -0.56 | `\n` (unchanged) |
| ex-planet-color-third-fourth | 30 | -2.62 -> -0.69 | ` blue` (unchanged) |
| planet-rings-saturn2 | 45 | -12.34 -> -10.00 | ` spectacular` (unchanged) |

Criterion 4 fails here too: 0 of 3 items redirect. Roughly double the effect of
the single-position swap, still nowhere near a flip.

For context, the paper reports J-lens coordinate swaps succeeding on 54% of
multihop items for Haiku 4.5, 70% for Sonnet 4.5 and 70% for Opus 4.5. We
measure 0/3 on Gemma 3 12B. Two differences are worth naming before treating
this as a contradiction:

* the paper applies the swap across the whole **workspace band** of layers, not
  a single layer. We hold to one layer deliberately, so that `F_l` stays a
  function of a single perturbed vector and the tangent/chord comparison is
  well defined. That methodological choice costs causal strength.
* Gemma 3 12B is far smaller than the models the paper swaps in, and the paper's
  own swap success rate rises with scale (54% -> 70% -> 70%).

So this is not a refutation of the paper's swap result. It is a negative result
for *our* single-layer version of it, and it means the causal half of Claim 2
is not established by this route.

### The decisive negative: `Mars -> red` is approximately linear

`activation_interpolation.py` (CLAIMS.md Addendum 2). Interpolate at one layer
and one position between two residuals the model actually produces:

```
h(alpha) = (1 - alpha) h["fourth planet"] + alpha h["third planet"]
```

Intervention scale `|delta|/|h|` is 0.12-0.23 -- thirty to fifty times the
lens-coordinate swap's 0.004, and set by the natural distance between two real
model states rather than chosen.

| pair | layer | \|d\|/\|h\| | R^2 linear | 10-90 width | logit(B')-logit(B) |
|---|---|---|---|---|---|
| planet-color | 34 | 0.130 | **0.995** | 0.83 | -0.94 -> +1.00 |
| planet-color | 46 | 0.122 | **1.000** | 0.77 | -0.94 -> +3.06 |
| capital-language | 34 | 0.230 | **0.988** | 0.68 | -9.41 -> -1.38 |
| capital-language | 46 | 0.162 | **0.999** | 0.77 | -9.41 -> +6.56 |

* **Criterion 2 (nonlinearity) FAILS decisively.** Required `R^2 < 0.9`;
  measured 0.988 to 1.000. The response is very nearly a straight line.
* **Criterion 3 (threshold shape) FAILS.** Required the 10%-to-90% rise to
  occupy less than half the sweep; measured 0.68 to 0.83, i.e. the change is
  spread across essentially the whole path.

**This is the result the project brief anticipated.** It warned: "Mars-to-red
could be implemented approximately linearly after Mars is represented. [...] Do
not present it as evidence for Claim 2 unless the causal sweep actually
establishes a nonlinear transition." The sweep does not. Measured at an
adequate scale, moving the bridge representation from `Mars` toward `Earth`
changes the answer logits proportionally, with no threshold anywhere along the
path. `Mars -> red` is a **Claim 1 replication only**.

### How this squares with the map result

These two findings are not in tension, and the distinction matters.

* `cos(J_l, A_l) = 0.65` against a 0.99 ceiling says `F_l` is nonlinear **as a
  map**, averaged over the corpus and over all input directions.
* `R^2 = 0.995` says `F_l` is nearly linear **along this one path**, between
  these two particular states.

A function can be strongly nonlinear globally while being almost perfectly
linear along a specific chord through it. So the general premise Neel's
explanation needs -- that `F_l` is nonlinear enough for a local Jacobian and a
global regression to summarise it differently -- is **measured and real**. What
is *not* established is the specific mechanism sketch: that a gate-like circuit
converting `Mars` into `red` is what drives the J-lens/Tuned-lens divergence on
this prompt. On this prompt, no such gate is detectable.

Criterion 4 as written ("top-1 token flips to B'") was also ill-posed and is
withdrawn rather than scored: Gemma emits a newline before the answer on these
prompts, so the top-1 token is `\n` regardless. The sign of
`logit(B') - logit(B)` is the meaningful quantity and is reported instead; it
does cross zero for planet-color at L34 (alpha = 0.37), so the intervention is
causally effective even though the response is linear.

### Third pair, and the final verdict on the interpolation

Fixed `animal-legs` (Gemma answers `8`/`6` as digits, not words) and added
`instrument-count`, which fails behaviourally and is dropped: the model answers
"36" for the number of strings on Yo-Yo Ma's instrument.

| pair | layer | \|d\|/\|h\| | R^2 | gate index | 10-90 width |
|---|---|---|---|---|---|
| planet-color | 34 | 0.130 | 0.9954 | 0.56 | 0.83 |
| planet-color | 46 | 0.122 | 0.9997 | 0.33 | 0.77 |
| **animal-legs** | **34** | 0.177 | **0.9298** | **0.74** | **0.45** |
| animal-legs | 46 | 0.190 | 0.9964 | 0.19 | 0.73 |
| capital-language | 34 | 0.230 | 0.9879 | 0.36 | 0.68 |
| capital-language | 46 | 0.162 | 0.9985 | 0.20 | 0.77 |

`animal-legs` at L34 is the one case with a gate-like signature: gate index 0.74
(criterion passes, > 0.5), 10-90 width 0.45 (passes, < 0.5), logit difference
swinging -8.62 to +7.62. And its own late-layer control behaves as predicted --
R^2 rises 0.930 -> 0.996 and the width nearly doubles, so the transition really
is sharper at the workspace layer than near the output. That is the shape
Claim 2 predicts.

**But the experiment fails as preregistered.**

* Criterion 2 (`R^2 < 0.9`): fails everywhere, including animal-legs at 0.930.
* Criterion 3 (width < 0.5): passes only for animal-legs at L34.
* Criterion 6 (reproducible on at least two further pairs): **fails**, one of
  three.

One suggestive case out of three, which does not clear the threshold it was
measured against, is a lead rather than a result. It is recorded as such. The
`spider -> eight legs` family is where a follow-up should look -- it is also the
example the project brief singled out -- but establishing it would need a
proper item set, not a single prompt pair noticed after the fact.

### The paired shape test: a positive result, and a family split

`shape_paired_test.py` (CLAIMS.md Addendum 3). 21 candidate pairs, 14 dropped
by the behavioural/single-token filter (Gemma answers two of them in Chinese,
misreads several others), leaving n = 7. Workspace layer 34, control layer 46.

| item | workspace width | control width | difference |
|---|---|---|---|
| legs-spider-dog | 0.217 | 0.700 | -0.483 |
| legs-ant-bird | 0.233 | 0.717 | -0.483 |
| sides-triangle | 0.283 | 0.733 | -0.450 |
| planet-moons | 0.317 | 0.750 | -0.433 |
| legs-spider-bee | 0.450 | 0.733 | -0.283 |
| capital-language | 0.683 | 0.767 | -0.083 |
| planet-color | 0.833 | 0.767 | **+0.067** |

* 6 of 7 sharper at the workspace layer. Sign test p = 0.0625; exact sign-flip
  permutation test on the magnitudes **p = 0.0156**.
* Median width 0.317 at the workspace layer against 0.733 at the control -- the
  transition is more than twice as concentrated.

**The split across items is the more interesting finding.** Every item with a
**numeric** answer gates sharply (widths 0.22-0.45); both items with a
**categorical** answer do not (0.68, 0.83):

```
numeric     : legs x3, sides-triangle, planet-moons   median width 0.283
categorical : planet-color, capital-language          median width 0.758
```

A plausible reading is that producing a count forces a commitment to one member
of a small ordered set, which is naturally implemented as a thresholded
selection, whereas colour and language may be represented more gradedly. That is
a hypothesis this experiment suggests, not one it tests.

**`planet-color` is one of the two items showing no gate** -- and it is the one
with the positive sign, the only item where the workspace transition is *less*
sharp than the control. So the Mars example fails the gate test from two
independent directions: its own interpolation is linear (`R^2 = 0.995`), and in
the paired comparison it is the single item that moves the wrong way.

Caveats, all material:

* n = 7 after 67% attrition. The numeric/categorical contrast is 5 items against
  2 and is an observation, not an established split.
* Layer 34 was selected from the Mars item's J-lens peak and then applied to
  every item, rather than chosen per item.
* Sharper transitions at mid-depth are consistent with gating but do not
  identify a circuit. Nothing here localises the mechanism.

### Addendum 4 result: the fitting distribution is what produces tangent-vs-chord

`in_domain_lenses.py`. Layer 34, spider -> dog path. Both lenses refitted with
the same estimators on a corpus of 375 two-hop animal-property prompts, versus
the deployed fit on pile-10k web text.

Reference quantities, both properties of the curve itself:

```
tangent of F at alpha = 0 :   4,315
chord of F over [0, 1]    :  40,911      (a factor of 9.5 -- the gate)
```

| corpus | J-lens slope | tuned slope | tuned / J | OLS R^2 | fit tokens |
|---|---|---|---|---|---|
| web text | 11,013 | 9,622 | 0.87 | 0.858 | 7,104 |
| in-domain | 26,570 | **41,564** | **1.56** | 0.918 | 18,315 |

* **Prediction 1 confirmed, and tightly.** Refit in-domain, the regression lands
  at 41,564 against a chord of 40,911 -- within 1.6%. Fit on a distribution that
  straddles the gate, the tuned lens *is* the chord. Neel's mechanism, in a real
  model.
* **Prediction 3 confirmed.** The tuned/J ratio moves 0.87 -> 1.56; the two
  lenses genuinely separate.
* **Prediction 2 fails as written.** The J-lens was predicted to stay near the
  tangent; it moved 11,013 -> 26,570, about 65% of the chord. The reason is
  structural rather than surprising: notebook 01's toy deliberately clusters its
  corpus *away* from the gate (mass at 0.25 and 1.75, gate at 1.0), so its
  average tangent stays small. The animal corpus spreads across the leg-count
  axis, putting substantial mass *on* the gate, so the average tangent is large.
  Separation of the two lenses needs the corpus to be in-domain **and** to avoid
  the gate -- being in-domain alone is not enough.

### An unpredicted cost: in-domain fitting degrades the J-lens readout

Full-vocabulary ranks at the unmodified spider state:

| lens | corpus | `spider` | `8` |
|---|---|---|---|
| J | web | **7** | 638 |
| J | in-domain | 14 | **53** |
| tuned | web | 119 | 6 |
| tuned | in-domain | 49 | 1 |

The web-fit J-lens separates bridge from answer by a factor of ~90 in rank. The
in-domain J-lens collapses that to ~4, and starts reporting the answer.

This is consistent with Neel's account rather than against it. The Jacobian is
supposed to ask what the model would say "on an arbitrary prompt"; averaging it
over a task-specific corpus reintroduces exactly the predictive correlations
that genericity was there to strip out. So there is a real trade-off:

* **generic corpus** -> the J-lens reports current content (good for Claim 1),
  but neither lens exhibits the tangent/chord geometry;
* **in-domain corpus** -> the geometry appears cleanly (good for Claim 2), but
  the J-lens loses the readout specificity that makes it useful.

Answering the question that prompted this: fitting in-corpus does *not* simply
make the lenses "work better". It makes them behave like the toy, at the cost of
the property the J-lens is actually for.
