# Hessian–vector products without forming the Hessian

**Status:** stub — placeholder so the homepage timeline has something to link to.

## The one-liner

If $L(\theta)$ is twice differentiable and $v$ is a fixed vector, then

$$H v \;=\; \nabla_\theta\bigl(\nabla_\theta L(\theta)^\top v\bigr),$$

so you can compute $Hv$ with a single backward pass through a scalar that is
itself the output of a backward pass. Two `vjp`s, no dense $H$.

## Coming soon

- Pearlmutter (1994), the original trick.
- Gauss–Newton vector products and why they're usually what you actually want.
- Finite-difference alternatives when the loss isn't twice differentiable.
- A PyTorch implementation with `torch.autograd.grad(..., create_graph=True)`.
