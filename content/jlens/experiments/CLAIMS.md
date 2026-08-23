# Claims under test

This project separates a result already reported in the J-Lens paper from Neel Nanda's proposed explanation for it.

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
