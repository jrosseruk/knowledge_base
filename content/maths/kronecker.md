# Kronecker-factored approximations: K-FAC & EKFAC

**Status:** stub — placeholder so the homepage timeline has something to link to.

## The one-liner

For a fully-connected layer with input $a$ and output pre-activation gradient
$g$, the Fisher block is $\mathbb{E}[a a^\top] \otimes \mathbb{E}[g g^\top]$
*if* the two are independent. K-FAC assumes exactly that, which turns an
intractable dense block into a tiny Kronecker product you can invert per
factor.

EKFAC keeps K-FAC's eigenbasis but re-learns the eigenvalues from data.

## Coming soon

- Derivation of the Kronecker structure under the independence assumption.
- What EKFAC fixes about K-FAC and what it doesn't.
- Why this is the approximation under the hood of most scalable influence-
  function pipelines.
