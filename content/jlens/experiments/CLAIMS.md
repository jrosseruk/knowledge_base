# Claims under test

This project separates a result already reported in the J-Lens paper from Neel Nanda's proposed explanation for it.

References:
- /home/mac/knowledge_base/jlens.html
- /home/mac/knowledge_base/neels_jlen_review.pdf

## Claim 1: the three lenses answer different questions

At a layer where the model is holding an unspoken intermediate concept, the lenses should behave differently:

- **J-Lens** should recover the intermediate concept currently represented in the residual stream.
- **Tuned Lens** should skip ahead and predict the model's eventual output.
- **Logit Lens** may be noisy or misleading, especially before its basis becomes aligned with the final residual-stream basis.

For the paper's two-hop example,

\[
\text{fourth planet} \longrightarrow \text{Mars} \longrightarrow \text{red},
\]

the predicted layerwise pattern is:

| Lens | Expected readout before the answer is computed |
|---|---|
| J-Lens | `Mars` |
| Tuned Lens | `red` |
| Logit Lens | noise, followed later by `Mars` or `red` |

This is an empirical claim reported qualitatively in Figure 51 of the J-Lens paper. Our first goal is to reproduce it in an open, instruction-tuned model.

## Claim 2: locality explains the J-Lens/Tuned-Lens difference

Neel's proposed mechanism is that the two fitted linear maps summarize different aspects of the nonlinear downstream function \(F_\ell\):

\[
h_\ell \xrightarrow{F_\ell} h_{\mathrm{final}}.
\]

The J-Lens averages local derivatives,

\[
A_\ell^{J}=\mathbb{E}_{h}\!\left[\frac{\partial F_\ell(h)}{\partial h}\right],
\]

so it asks approximately:

> If this concept became infinitesimally stronger, what would the model become more disposed to say, before a downstream nonlinear circuit switches state?

The Tuned Lens fits a global affine predictor,

\[
A_\ell^{T},b_\ell^{T}
=\arg\min_{A,b}\mathbb{E}_{h}\left[\lVert F_\ell(h)-(Ah+b)\rVert^2\right],
\]

so it asks approximately:

> Given everything correlated with this activation, what final state or output should we expect?

If a later nonlinear circuit conditionally maps `Mars` to `red`, a regression can learn the across-context chord `Mars` \(\Rightarrow\) `red`, while a local Jacobian can remain aligned with the currently represented variable `Mars`.

Unlike Claim 1, this mechanistic explanation is not isolated by the paper's three-lens comparison. It requires a causal experiment.

## Decisive experiment

We count the experiment as supporting both claims only if all of the following hold at the same preselected layer:

1. The model answers the two-hop prompt correctly.
2. J-Lens ranks the intermediate `Mars` highly.
3. Tuned Lens ranks the eventual answer `red` substantially above `Mars`.
4. Logit Lens is less informative than J-Lens about `Mars`.
5. Swapping the `Mars` representation for another planet changes the final answer to the corresponding property of that planet.
6. Swapping the answer direction itself at the same layer has a weaker effect, independently indicating that the answer is not yet represented there.
7. An intervention-strength sweep produces a nonlinear downstream answer curve. On the same axes, the J-Lens local line follows the tangent while the Tuned Lens follows the across-distribution prediction or chord.

Failure of any item is recorded rather than repaired by selecting a different metric after seeing the result. In particular, a nonlinear response without a strong full-vocabulary J-Lens readout of `Mars` is not evidence for the complete hypothesis.

## Sources

- `neels_jlen_review.pdf`, pp. 36–40: factual recall, working memory, and “Why Jacobians rather than linear regression?”
- `jlens.html`, Figures 51–55: qualitative and quantitative comparisons of J-Lens, Logit Lens, and Tuned Lens.

---

# Addendum: preregistered criteria for the Claim 2 mechanism test

Written 2026-08-23, after the Claim 1 pilot (`LOG.md`) and **before** any
causal measurement was made.

## The identity the test rests on

Write the remaining computation as \(F_\ell : h_\ell \mapsto h_{\text{final}}\).
If \(F_\ell\) were globally affine, \(F_\ell(h) = Mh + c\), then

* its Jacobian is \(M\) everywhere, so the corpus-averaged Jacobian \(J_\ell = M\);
* least squares recovers it exactly, so the regression matrix \(A_\ell = M\).

So **\(J_\ell = A_\ell\) whenever \(F_\ell\) is affine.** Any measured gap
between the two maps is therefore a direct measurement of the nonlinearity that
Neel's explanation requires, not an analogy for it. This converts Claim 2 from
a story into a quantity.

## A common readout functional

The model and all three lenses are compared through one fixed linear
functional, so that no normalisation constant can be tuned:

\[
s(x) = \langle w_{B'} - w_B,\; x\rangle, \qquad
w_v = W_U[v] \odot (\text{final-norm gain}),
\]

the unnormalised logit difference between the counterfactual answer and the
true answer. Along an intervention path \(h(\alpha)\),

\[
\text{actual}(\alpha) = s(F_\ell(h(\alpha))), \quad
\text{J}(\alpha) = s(J_\ell h(\alpha)), \quad
\text{T}(\alpha) = s(A_\ell h(\alpha) + b_\ell), \quad
\text{L}(\alpha) = s(h(\alpha)).
\]

The three lens curves are straight lines by construction. The question is where
each line sits relative to the model's real response.

## The intervention

The paper's lens-coordinate swap (Figure 4C), on a real internal
representation. With \(V = [v_A; v_{A'}]\) the J-lens vectors of the bridge
entity and its counterfactual and \(a = (VV^\top)^{-1}Vh\) the dual
coordinates, exchanging the two coordinates leaves the orthogonal component
untouched and moves \(h\) by \(\delta = (a_2 - a_1)(v_A - v_{A'})\). We sweep
\(h(\alpha) = h + \alpha\delta\). No input-embedding mixture is used anywhere.

## Criteria

The mechanism test supports Claim 2 only if all of the following hold.

1. **Nonlinearity is real.** At the pre-switch layer, `actual` over
   \(\alpha \in [0,1]\) is poorly fit by a straight line
   (\(R^2 < 0.9\)), **and** the tangent and chord slopes disagree by more than
   50%: \(|T_0 - C| / \max(|T_0|, |C|) > 0.5\), where \(T_0\) is the pointwise
   tangent from a JVP and \(C = \text{actual}(1) - \text{actual}(0)\).

2. **The maps differ, and not by noise.** At that layer,
   \(\cos(J_\ell d, A_\ell d)\) over natural probe directions is far below the
   split-half reliability ceiling of \(J_\ell\) itself.

3. **The divergence tracks the nonlinearity across layers.** Layers where the
   two maps agree are layers where the two readouts agree. A late layer at
   which \(\cos(J_\ell d, A_\ell d)\) approaches the ceiling must show a
   straight `actual` curve and near-coincident lens lines. This is the internal
   control: without it, a single nonlinear anecdote proves nothing.

4. **Causal redirection.** Swapping \(A \to A'\) flips the model's top output
   to \(B'\).

5. **Causal timing.** The same swap applied to the *answer* pair
   \(B \leftrightarrow B'\) at the same layer moves the output substantially
   less than the bridge swap, independently indicating that \(A\) is present
   and used at that layer while \(B\) is not yet represented.

6. **Full-vocabulary readouts.** All rank claims are over the entire
   vocabulary. A targeted two-token contrast may screen but may not conclude.

Failure of any item is reported as a failure. In particular, a nonlinear
response curve *without* criterion 3 is not evidence that nonlinearity explains
the lens difference — only that both happen to be present.

---

# Addendum 2: a second, adequately-scaled causal probe

Written 2026-08-23 **after** the preregistered lens-coordinate swap failed
criteria 1, 4 and 5, and before running the replacement. The failure stands in
the record (`LOG.md`); this does not replace it.

## Why a second probe rather than a reinterpretation of the first

The swap moved the activation by `|delta|/|h| = 0.004`. A 0.4% perturbation
cannot distinguish "F is locally linear" from "F has a gate somewhere along
this direction", because it never travels far enough to reach one. The first
experiment therefore did not test the hypothesis; it under-powered it. Reporting
its linear response as evidence *against* gating would be as wrong as reporting
it as evidence for.

## The replacement

Interpolate between two **real internal states** of the model, at one layer and
one position:

    P  = "Fact: The color of the planet fourth from the Sun is "   -> red
    P' = "Fact: The color of the planet third from the Sun is "    -> blue

    h(alpha) = (1 - alpha) h_P + alpha h_P',   alpha in [0, 1]

with every other position taken from `P`'s forward pass. At `alpha = 0` and
`alpha = 1` the state is one the model actually produces, so the endpoints are
on-distribution by construction and the intervention magnitude is the natural
one. This is a residual-stream interpolation at a mid-network layer, **not** an
input-embedding mixture.

## Criteria, fixed before running

1. **Behavioural**: the model answers both prompts correctly under greedy
   decoding.
2. **Nonlinearity**: over `alpha in [0, 1]`, the response
   `s = logit(blue) - logit(red)` has linear-fit `R^2 < 0.9` **and** gate index
   `|tangent - chord| / max(|tangent|, |chord|) > 0.5`.
3. **Threshold shape**: the 10%-to-90% transition occupies less than half the
   sweep range, i.e. the response is concentrated rather than uniform.
4. **Output flip**: the top-1 token changes from `red` to `blue` somewhere in
   the sweep.
5. **Affine control**: the same sweep at a late layer, where
   `cos(J, A_reg) > 0.95`, is markedly more linear than at the workspace layer.
6. **Reproducibility**: the same qualitative shape on at least two further
   natural prompt pairs, not just this one.

Failure of any item is reported as a failure. In particular, if the response is
linear at an adequate intervention scale, that is evidence that `Mars -> red` is
implemented approximately linearly once `Mars` is represented — which is exactly
the possibility the project brief warned against assuming away.

---

# Addendum 3: the shape test, done properly

Written 2026-08-23 after inspecting `fig_interpolation.png`.

## An admission about criterion 2

Addendum 2's criterion 2 used linear-fit `R^2 < 0.9`. That is a **bad detector
of the shape it was meant to detect**. A sigmoid whose endpoints are far apart
has most of its variance explained by a straight line: `animal-legs` at L34 is
visibly a clean gate — flat, sharp rise near alpha = 0.45, flat — and still
scores `R^2 = 0.93`. The two other statistics in the same preregistration, gate
index (0.74) and 10-90 transition width (0.45), both flagged it correctly.

The failure of criterion 2 is recorded and stands. But it is a failure of the
statistic, not evidence that the response is linear, and it would be wrong to
report it as the latter. Fixing this by inspecting one curve and re-scoring it
would be exactly the fishing the brief warns against, so instead:

## The paired test

Take `N >= 8` prompt pairs of the form (bridge A -> answer B, bridge A' -> B'),
require both members answered correctly, and interpolate the layer-`l` residual
as in Addendum 2. For each pair measure the 10-90 transition width at the
workspace layer and at a late control layer.

**Prediction, fixed before running:** if gate-like circuits sit in the workspace
band, transition widths should be systematically *smaller* at the workspace
layer than at the late control layer, tested as a paired comparison across
items (sign test / median difference). This is a within-item contrast, so it
does not depend on any absolute width threshold.

Secondary: report the fraction of items whose workspace-layer width falls below
0.5, as a descriptive statistic rather than a pass/fail gate.

A null result — widths no smaller at the workspace layer — means the sigmoid in
`animal-legs` is idiosyncratic and no general gating claim is supported.

---

# Addendum 4: do the lenses behave as tangent and chord when fitted in-domain?

Written 2026-08-23, before running.

## Why

On the `spider -> 8` / `dog -> 4` path at layer 34, the model's response is a
clean sigmoid, but all three deployed lenses have nearly the same shallow slope
(J 11013, tuned 9524, logit 12838, against a network chord of 40911). So the
tuned lens is *not* acting as the chord of this curve.

Notebook 01's toy gets the tangent/chord separation because its fitting corpus
is two clusters straddling the gate **along the swept axis**. The deployed
lenses are fitted on pile-10k web text, which never explores the
spider-versus-dog direction. The separation may therefore be absent here
because of the fitting distribution rather than because the mechanism is wrong.

## The test

Refit both lenses at layer 34 on an **in-domain** corpus: a few hundred
two-hop animal-property prompts, so that the layer-34 activations genuinely
vary along the "which animal" direction with mass either side of the gate.
Same estimators, same code, only the corpus changes. Then re-measure the slope
along the spider -> dog path against two quantities that are properties of the
curve itself:

* the **tangent**, `d/dalpha s(F(h(alpha)))` at `alpha = 0`, by JVP;
* the **chord**, `s(F(h(1))) - s(F(h(0)))`.

The regression compared is ordinary least squares in closed form for both
corpora, since OLS is the object Neel's argument names.

## Predictions, fixed before running

1. **Tuned lens moves toward the chord.** Its in-domain slope should be
   substantially larger than its web-text slope of 9524 and closer to 40911.
2. **J-lens does not.** Its in-domain slope should stay well below the chord
   and nearer the local tangent, because even in-domain most corpus mass sits
   off the narrow gate.
3. **They separate.** The ratio (tuned slope / J slope) should be clearly
   greater in-domain than the current 0.86.

If both move together, or neither moves, the locality account does not survive
this test and prediction 3 is the one that matters.

## The obvious objection, stated up front

Fitting a lens on the distribution it is then probed against is circular in the
sense that it constructs the toy. That is the point: it isolates whether the
*fitting distribution* is what produces the tangent/chord separation. It does
not show that the deployed lenses behave this way on web text -- we have
already measured that they do not.

---

# Addendum 5: which relations gate, and does an in-domain corpus rescue the rest?

Written 2026-08-24, before running. Revised once before running, to fold in the
in-domain question from Addendum 4.

## What prompted it

The paired shape test found sharp transitions on every item whose answer was
**numeric** (legs x3, sides of a triangle, moons of a planet; 10-90 widths
0.22-0.45) and on neither item whose answer was **categorical** (planet colour
0.83, capital-to-language 0.68). Five items against two, consistent with at
least two readings:

* **H-numeric** -- counting is special: producing a count means selecting one
  element of a small *ordered* set, and that selection is thresholded.
* **H-discrete** -- nothing to do with number: any relation whose answer comes
  from a small closed set is thresholded, and colour and language gate less
  because their answer spaces are large and graded.

They differ on families whose answers are discrete and small but not numeric --
a **first letter**, a **state of matter**. H-numeric predicts those behave like
colour; H-discrete predicts they behave like counts.

## The second question

Addendum 4 showed the tuned lens becomes the chord (within 1.6%) once fitted on
a corpus that straddles the gate. That invites the hope that fitting in-domain
simply makes the lenses work everywhere. It should not, and the reason is
arithmetic: **if `F` is linear along the path, its tangent and its chord are
identically equal**, so there is nothing for a Jacobian and a regression to
separate into, whatever corpus either is fitted on.

## Design

Families, each with several items and its own generated in-domain corpus:

| group | families |
|---|---|
| numeric | animal legs, geometric sides, sport team size and scoring |
| discrete non-numeric | first letter, state of matter |
| graded categorical | planet colour, capital-to-language (the France case) |

For each item, at layer 34 with layer 46 as the within-item control:

1. the model's response curve -- tangent at `alpha=0`, chord over `[0,1]`,
   10-90 transition width;
2. the J-lens and least-squares slopes along the same path, fitted **twice**:
   once on web text and once on that family's own corpus.

Because both lenses are linear, a slope is `<w, M delta>` and needs a single
transported direction, not the whole path.

Both members of a pair must be answered correctly under greedy decoding and
both answers must be single tokens. Attrition is reported.

## Predictions, fixed before running

1. The numeric group reproduces its sharp transitions; the graded categorical
   group does not.
2. **The discriminating test**: where the discrete non-numeric group's median
   width falls decides between H-numeric and H-discrete.
3. **The conditional-rescue test**: in-domain fitting raises the tuned/J slope
   ratio **only for items whose response gates**. Concretely, the change in
   that ratio from web to in-domain should track the item's gate strength
   (chord / tangent), and should be near zero for the graded categorical items.

Reported as measured. If families do not separate at all, the earlier 5-vs-2
split was noise, and that is the finding.

---

# Addendum 6: an effectiveness precondition, and a correction

Written 2026-08-24, after inspecting the survey's logit swings and **before**
the rerun.

## The correction

Addendum 5's survey was scored on transition width alone. Checking how far each
intervention actually moved the output shows that most non-numeric items were
never moved at all:

| item | swing (nats) | crosses zero | width |
|---|---|---|---|
| 7 numeric items | 13.6 - 16.3 | yes | 0.217 - 0.483 |
| up-wet | 0.66 | no | 0.733 |
| light-fast | 2.52 | no | 0.817 |
| paris-madrid | 8.03 | no | 0.683 |
| mars-earth | 1.94 | yes | 0.833 |

A transition width computed on a response that never moves is a measurement of
noise. **The conclusion "H-numeric beats H-discrete" is therefore withdrawn**:
the discrete group had no effective interventions, so it did not test anything.
The only non-numeric item with an intervention that flips the sign is
`mars-earth`, and one item decides nothing.

The same defect voids the few-shot items in `controlled_gate_test.py` (swings
0.12 - 1.03). Their flat responses reflect the final-position residual of a
few-shot prompt not carrying the entity distinction, not an absence of gating.

## The precondition, fixed now

An item's width is interpretable only if the intervention is **causally
effective**:

1. `|logit(B') - logit(B)|` swings by at least **4 nats** across the path, and
2. the difference **changes sign**, i.e. the intervention actually reverses
   which answer the model prefers.

Items failing either are reported as attrition and never scored.

## The fix to the intervention

The failures share a cause: the pairs differ in a word several tokens before
the readout, so interpolating only the final residual moves little. The rerun
uses **token-length-matched pairs** differing in a single token, and
interpolates the residual at **every position** rather than only the last. The
path is still `(1-alpha) h_A + alpha h_B` between two states the model really
produces, so the endpoints remain on-distribution.

## Prediction, fixed before running

With effective interventions on both sides, the numeric group keeps its sharp
transitions. The open question is what the non-numeric group does, and this
time it will be a real test rather than an artefact. No prediction is offered
for it; the point of the rerun is that Addendum 5 could not answer it.
