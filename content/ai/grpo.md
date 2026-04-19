# GRPO, demystified

**Status:** stub — placeholder so the homepage timeline has something to link to.

## The one-liner

Group-Relative Policy Optimisation (GRPO) is PPO where, instead of a learned
value baseline, you subtract the mean reward of a *group* of samples drawn
from the same prompt. No critic network, cheaper to train, higher variance.

## Coming soon

- Precise statement of the estimator and its bias / variance tradeoff against
  PPO's GAE.
- What "group" choices look like in practice (same-prompt vs same-task).
- A toy bandit experiment comparing PPO, REINFORCE, and GRPO.
