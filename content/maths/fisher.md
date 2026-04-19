# The Fisher information matrix, four ways

**Status:** stub — placeholder so the homepage timeline has something to link to.

## The one-liner

Four definitions of "the Fisher" that people use interchangeably:

1. **Score covariance.** $F = \mathbb{E}_{p_\theta}[\nabla\log p_\theta\,\nabla\log p_\theta^\top]$.
2. **Expected Hessian (of negative log-likelihood).** $F = -\mathbb{E}_{p_\theta}[\nabla^2 \log p_\theta]$.
3. **Gauss–Newton.** $G = J^\top H_y J$, with $J$ the model Jacobian and $H_y$ the output Hessian.
4. **Empirical Fisher.** Same as (1) but with expectation over *data* labels rather than model samples.

(1) and (2) agree for regular models. (3) agrees with (1) for exponential-
family outputs. (4) is the impostor in the room and quietly misbehaves.

## Coming soon

- Worked derivations of the equivalences and where they break.
- Why "empirical Fisher" is not a Fisher and what breaks if you preconditioning
  with it.
- Natural gradient as preconditioned GD and the connection to mirror descent.
