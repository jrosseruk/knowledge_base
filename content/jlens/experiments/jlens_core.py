"""Faithful, dependency-free implementations of the three lenses.

Everything here is plain PyTorch + HuggingFace `transformers`.  The
`jacobian-lens` reference package is *not* imported; its estimator is
re-derived below so the whole pipeline can run live in a notebook.

The Jacobian lens (Anthropic, *Verbalizable Representations Form a Global
Workspace in Language Models*) is

    lens_l(h) = softmax( W_U norm( J_l h ) ),
    J_l       = E_prompt E_{t in P} [ sum_{t' in P, t' >= t}  d h_final[t'] / d h_l[t] ]

where `P` is the set of "valid" positions of a corpus sequence: the reference
implementation drops the first 16 positions (attention sinks have atypical
residual statistics) and the last position.

Fitting the full `d_model x d_model` matrix costs `d_model / dim_batch`
backward passes per prompt.  We never need the matrix itself -- only

  * `J_l h`      for readouts, and
  * `J_l^T u`    for the lens *vector* of a single vocabulary token,

so we compute those directly:

  * `J_l h` is a **Jacobian-vector product**.  Perturbing `h_l[t]` by `h` at
    every valid source position `t` at once and summing the resulting tangent
    over valid target positions `t'` gives exactly
    `sum_t sum_{t' >= t} (d h_final[t'] / d h_l[t]) h` by causality, so one
    forward-mode pass per corpus sequence yields the estimator.
  * `J_l^T u` is a **vector-Jacobian product**: place cotangent `u` at every
    valid target position, backpropagate, and average the gradient over valid
    source positions.

Both are exact -- no finite differences, no low-rank approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.nn.functional as F
from torch import nn

SKIP_FIRST_N_POSITIONS = 16


# --------------------------------------------------------------------------
# model plumbing
# --------------------------------------------------------------------------


@dataclass
class LensModel:
    """A minimal view of a decoder-only HF model: blocks, final norm, unembed."""

    hf_model: nn.Module
    tokenizer: object
    text_module: nn.Module
    layers: nn.ModuleList
    final_norm: nn.Module
    lm_head: nn.Module
    n_layers: int
    d_model: int
    device: torch.device

    def unembed(self, residual: torch.Tensor) -> torch.Tensor:
        dtype = self.lm_head.weight.dtype
        return self.lm_head(self.final_norm(residual.to(dtype))).float()

    def forward(self, input_ids: torch.Tensor):
        return self.text_module(input_ids=input_ids, use_cache=False)


def wrap(hf_model: nn.Module, tokenizer) -> LensModel:
    """Locate the residual stack inside a `*ForCausalLM` / `*ForConditionalGeneration`."""
    candidates = ["model.language_model", "model", "language_model", "transformer"]
    text_module = None
    for path in candidates:
        node = hf_model
        try:
            for part in path.split("."):
                node = getattr(node, part)
        except AttributeError:
            continue
        if all(hasattr(node, a) for a in ("layers", "norm", "embed_tokens")):
            text_module = node
            break
    if text_module is None:
        raise ValueError(f"could not find the decoder stack in {type(hf_model).__name__}")

    text_config = hf_model.config.get_text_config()
    hf_model.eval()
    for parameter in hf_model.parameters():
        parameter.requires_grad_(False)
    return LensModel(
        hf_model=hf_model,
        tokenizer=tokenizer,
        text_module=text_module,
        layers=text_module.layers,
        final_norm=text_module.norm,
        lm_head=hf_model.lm_head,
        n_layers=text_config.num_hidden_layers,
        d_model=text_config.hidden_size,
        device=text_module.embed_tokens.weight.device,
    )


def _block_output(module_output):
    return module_output[0] if isinstance(module_output, tuple) else module_output


def _rewrap(module_output, new_hidden):
    if isinstance(module_output, tuple):
        return (new_hidden, *module_output[1:])
    return new_hidden


def valid_positions(seq_len: int, skip_first: int = SKIP_FIRST_N_POSITIONS) -> torch.Tensor:
    if seq_len <= skip_first + 1:
        raise ValueError(f"sequence of {seq_len} tokens is too short for skip_first={skip_first}")
    return torch.arange(skip_first, seq_len - 1)


# --------------------------------------------------------------------------
# corpus
# --------------------------------------------------------------------------


def corpus_batches(
    tokenizer,
    texts: Sequence[str],
    *,
    n_sequences: int,
    seq_len: int = 128,
    batch_size: int = 8,
    device: str | torch.device = "cuda",
    seed: int = 0,
) -> list[torch.Tensor]:
    """Tokenize `texts` into fixed-length sequences, each starting with BOS."""
    bos = tokenizer.bos_token_id
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(texts), generator=generator).tolist()

    sequences: list[list[int]] = []
    for index in order:
        text = texts[index]
        if not text or len(text) < 400:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)[: seq_len - 1]
        if len(ids) < seq_len - 1:
            continue
        sequences.append(([bos] if bos is not None else []) + ids)
        if len(sequences) >= n_sequences:
            break
    if len(sequences) < n_sequences:
        raise ValueError(f"only found {len(sequences)} usable sequences")

    tensor = torch.tensor(sequences, dtype=torch.long, device=device)
    return list(tensor.split(batch_size))


# --------------------------------------------------------------------------
# Jacobian lens: J_l h  (forward mode)
# --------------------------------------------------------------------------


def transport(
    model: LensModel,
    layer: int,
    directions: torch.Tensor,
    batches: Sequence[torch.Tensor],
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    progress: Callable[[int, int], None] | None = None,
) -> torch.Tensor:
    """Return `J_l @ d` for every row `d` of `directions`, shape `[n_dirs, d_model]`.

    `layer` indexes the transformer block whose *output* is the source residual,
    i.e. `hidden_states[layer + 1]` in HuggingFace's convention.
    """
    if directions.ndim == 1:
        directions = directions[None]
    directions = directions.to(model.device, torch.float32)
    n_directions = directions.shape[0]

    source_module = model.layers[layer]
    target_module = model.layers[-1]
    totals = torch.zeros(n_directions, model.d_model, dtype=torch.float64, device=model.device)
    n_sequences = 0

    for batch_index, input_ids in enumerate(batches):
        n_batch, seq_len = input_ids.shape
        positions = valid_positions(seq_len, skip_first).to(model.device)
        # One JVP handles every direction at once by replicating each corpus
        # sequence `n_directions` times along the batch axis.
        expanded_ids = input_ids.repeat_interleave(n_directions, dim=0)
        pattern = directions.repeat(n_batch, 1)  # [n_batch * n_directions, d_model]

        def run(alpha: torch.Tensor) -> torch.Tensor:
            captured: dict[str, torch.Tensor] = {}

            def inject(_module, _args, module_output):
                hidden = _block_output(module_output)
                delta = torch.zeros_like(hidden)
                delta[:, positions] = (alpha * pattern).to(hidden.dtype)[:, None, :]
                return _rewrap(module_output, hidden + delta)

            def capture(_module, _args, module_output):
                captured["h"] = _block_output(module_output)

            handles = [
                source_module.register_forward_hook(inject),
                target_module.register_forward_hook(capture),
            ]
            try:
                model.forward(expanded_ids)
            finally:
                for handle in handles:
                    handle.remove()
            # sum over valid target positions; causality restricts the
            # contribution at t' to source positions t <= t'.
            return captured["h"][:, positions].float().sum(dim=1)

        zero = torch.zeros((), device=model.device, dtype=torch.float32)
        one = torch.ones((), device=model.device, dtype=torch.float32)
        with torch.no_grad():
            _, tangent = torch.func.jvp(run, (zero,), (one,))
        tangent = tangent.double().view(n_batch, n_directions, model.d_model)
        totals += tangent.sum(dim=0) / len(positions)

        n_sequences += n_batch
        if progress is not None:
            progress(batch_index + 1, len(batches))

    return (totals / n_sequences).float()


def lens_vectors(
    model: LensModel,
    layer: int,
    token_ids: Sequence[int],
    batches: Sequence[torch.Tensor],
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> torch.Tensor:
    """Rows of `W_U J_l` for `token_ids`: the J-lens *vectors*, shape `[n_tokens, d_model]`.

    The unembedding is linearized at the mean final-layer residual so that the
    RMS norm contributes a fixed diagonal scale, matching the reference
    implementation's `unembed(J h)` up to a per-readout scalar.
    """
    weight = model.lm_head.weight[list(token_ids)].float()  # [n_tokens, d_model]
    gain = getattr(model.final_norm, "weight", None)
    if gain is not None:
        scale = (1.0 + gain.float()) if _gemma_style(model.final_norm) else gain.float()
        weight = weight * scale
    return transpose_transport(model, layer, weight, batches, skip_first=skip_first)


def _gemma_style(norm: nn.Module) -> bool:
    return type(norm).__name__.startswith("Gemma")


def transpose_transport(
    model: LensModel,
    layer: int,
    cotangents: torch.Tensor,
    batches: Sequence[torch.Tensor],
    *,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> torch.Tensor:
    """Return `J_l^T @ u` for every row `u` of `cotangents`, shape `[n_rows, d_model]`."""
    if cotangents.ndim == 1:
        cotangents = cotangents[None]
    cotangents = cotangents.to(model.device, torch.float32)

    source_module = model.layers[layer]
    target_module = model.layers[-1]
    totals = torch.zeros(cotangents.shape[0], model.d_model, dtype=torch.float64, device=model.device)
    n_sequences = 0

    for input_ids in batches:
        seq_len = input_ids.shape[1]
        positions = valid_positions(seq_len, skip_first).to(model.device)
        source: dict[str, torch.Tensor] = {}
        captured: dict[str, torch.Tensor] = {}

        def start_graph(_module, _args, module_output):
            hidden = _block_output(module_output).detach().requires_grad_(True)
            source["h"] = hidden
            return _rewrap(module_output, hidden)

        def capture(_module, _args, module_output):
            captured["h"] = _block_output(module_output)

        handles = [
            source_module.register_forward_hook(start_graph),
            target_module.register_forward_hook(capture),
        ]
        with torch.enable_grad():
            try:
                model.forward(input_ids)
            finally:
                for handle in handles:
                    handle.remove()
            target = captured["h"]
            for row_index in range(cotangents.shape[0]):
                grad_outputs = torch.zeros_like(target)
                grad_outputs[:, positions] = cotangents[row_index].to(target.dtype)
                (grad,) = torch.autograd.grad(
                    target,
                    source["h"],
                    grad_outputs=grad_outputs,
                    retain_graph=row_index < cotangents.shape[0] - 1,
                )
                totals[row_index] += grad[:, positions].float().sum(dim=(0, 1)).double() / len(positions)
        n_sequences += input_ids.shape[0]

    return (totals / n_sequences).float()


# --------------------------------------------------------------------------
# Tuned lens
# --------------------------------------------------------------------------


class TunedLens(nn.Module):
    """Affine translator `h_l -> h_final`, one per layer, trained on natural text.

    Parameterized as `h + W h + b` (identity residual), the standard
    initialization from Belrose et al.
    """

    def __init__(self, d_model: int, layers: Sequence[int]):
        super().__init__()
        self.layers = list(layers)
        self.translators = nn.ModuleDict(
            {str(layer): nn.Linear(d_model, d_model) for layer in self.layers}
        )
        for translator in self.translators.values():
            nn.init.zeros_(translator.weight)
            nn.init.zeros_(translator.bias)

    def forward(self, hidden: torch.Tensor, layer: int) -> torch.Tensor:
        return hidden + self.translators[str(layer)](hidden)


def rank_of(scores: torch.Tensor, token_id: int) -> int:
    """One-indexed full-vocabulary rank."""
    return int((scores > scores[token_id]).sum().item()) + 1


def top_tokens(tokenizer, scores: torch.Tensor, k: int = 10) -> list[str]:
    return [tokenizer.decode([i]) for i in torch.topk(scores, k).indices.tolist()]


def residual_stack(model: LensModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Residual stream at every depth, `[n_layers + 1, batch, seq, d_model]`.

    HuggingFace's `output_hidden_states` is *not* usable for the last entry:
    decoder stacks append `self.norm(hidden_states)` after the loop, so
    `hidden_states[-1]` is the **post-final-norm** state and
    `unembed(hidden_states[-1])` double-normalises and does not reproduce the
    model's logits.  Index `n_layers` here is the true pre-norm output of the
    last block, captured with a hook, so every index obeys the same convention
    and `unembed(stack[n_layers])` is exactly the model's output.
    """
    captured: dict[str, torch.Tensor] = {}

    def capture(_module, _args, module_output):
        captured["h"] = _block_output(module_output)

    handle = model.layers[-1].register_forward_hook(capture)
    try:
        with torch.inference_mode():
            hidden_states = model.hf_model(
                input_ids, output_hidden_states=True, use_cache=False
            ).hidden_states
    finally:
        handle.remove()
    # hidden_states[i] for i < n_layers is the pre-norm input to block i.
    return torch.stack([*hidden_states[: model.n_layers], captured["h"]], dim=0)
