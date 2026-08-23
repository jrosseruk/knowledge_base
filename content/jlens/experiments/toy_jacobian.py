"""The 2x2 picture behind the whole paper: a tangent is not a chord.

Two linear summaries of the same nonlinear map `F`:

* the **Jacobian** `dF/dh` at a point -- what F does to an *infinitesimal*
  nudge, before any downstream nonlinearity has time to change state;
* the **least-squares fit** `argmin E||F(h) - (Ah + b)||^2` over a data
  distribution -- what F does *on average across the whole cloud*.

They are the same matrix when F is affine and different otherwise, which is the
entire content of the J-lens / Tuned-lens distinction in two dimensions.

Run directly to produce `fig_toy_jacobian.png`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jlens_plot as jp


def affine(points: np.ndarray) -> np.ndarray:
    matrix = np.array([[1.0, 0.6], [-0.3, 0.9]])
    return points @ matrix.T


def gated(points: np.ndarray) -> np.ndarray:
    """A gate: coordinate 1 leaks into coordinate 2 only once it clears a threshold."""
    first = points[:, 0]
    second = points[:, 1]
    gate = 1.0 / (1.0 + np.exp(-6.0 * (first - 0.5)))
    return np.stack([first, 0.9 * second + 2.0 * gate], axis=1)


def jacobian_at(function, point: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Numerical Jacobian, columns = response to a nudge along each input axis."""
    columns = []
    for axis in range(point.shape[0]):
        step = np.zeros_like(point)
        step[axis] = epsilon
        columns.append((function((point + step)[None])[0] - function((point - step)[None])[0]) / (2 * epsilon))
    return np.stack(columns, axis=1)


def least_squares_map(function, cloud: np.ndarray) -> np.ndarray:
    """argmin_A E||F(h) - (A h + b)||^2, returning A."""
    outputs = function(cloud)
    centered_in = cloud - cloud.mean(0)
    centered_out = outputs - outputs.mean(0)
    return np.linalg.lstsq(centered_in, centered_out, rcond=None)[0].T


def main() -> None:
    jp.use_style()
    rng = np.random.default_rng(0)
    cloud = rng.normal(0, 0.6, size=(4000, 2))
    base = np.array([0.0, 0.0])

    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))

    # ---- (a) the scalar picture: tangent vs chord --------------------------
    axis = axes[0]
    grid = np.linspace(-1.5, 2.0, 400)
    curve = gated(np.stack([grid, np.zeros_like(grid)], axis=1))[:, 1]
    axis.plot(grid, curve, color=jp.INK, lw=2.4, label="$F$ (the real computation)")

    slope = jacobian_at(gated, base)[1, 0]
    axis.plot(
        grid,
        curve[np.argmin(np.abs(grid))] + slope * grid,
        color=jp.LENS_COLORS["j"],
        label="tangent at $h$  (Jacobian)",
    )
    chord_x = np.array([-1.0, 1.5])
    chord_y = gated(np.stack([chord_x, np.zeros_like(chord_x)], axis=1))[:, 1]
    chord_slope = (chord_y[1] - chord_y[0]) / (chord_x[1] - chord_x[0])
    axis.plot(
        grid,
        chord_y[0] + chord_slope * (grid - chord_x[0]),
        color=jp.LENS_COLORS["tuned"],
        label="chord across the data  (regression)",
    )
    axis.plot([0], [curve[np.argmin(np.abs(grid))]], "o", color=jp.INK, zorder=5)
    axis.set_xlabel("input coordinate $h_1$")
    axis.set_ylabel("output coordinate")
    axis.set_title("(a) One nonlinear gate,\ntwo linear summaries", fontsize=10)
    axis.grid(alpha=0.7)
    axis.legend(loc="upper left", fontsize=8)

    # ---- (b) affine F: the two matrices coincide ---------------------------
    axis = axes[1]
    jacobian = jacobian_at(affine, base)
    regression = least_squares_map(affine, cloud)
    _matrix_panel(
        axis,
        jacobian,
        regression,
        "(b) When $F$ is affine\nthe two agree exactly",
    )

    # ---- (c) gated F: they come apart --------------------------------------
    axis = axes[2]
    jacobian = jacobian_at(gated, base)
    regression = least_squares_map(gated, cloud)
    _matrix_panel(
        axis,
        jacobian,
        regression,
        "(c) Add the gate and\nthey come apart",
    )

    figure.savefig(Path(__file__).resolve().parent / "fig_toy_jacobian.png")
    print("wrote fig_toy_jacobian.png")
    print("affine   : max |J - A| =", np.abs(jacobian_at(affine, base) - least_squares_map(affine, cloud)).max())
    print("gated    : max |J - A| =", np.abs(jacobian_at(gated, base) - least_squares_map(gated, cloud)).max())


def _matrix_panel(axis, jacobian, regression, title) -> None:
    """Draw the two 2x2 matrices side by side as annotated cells."""
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    axis.axis("off")
    axis.set_title(title, fontsize=10)
    for offset, (matrix, name, color) in enumerate(
        ((jacobian, r"Jacobian  $\partial F/\partial h$", jp.LENS_COLORS["j"]),
         (regression, r"Regression  $A$", jp.LENS_COLORS["tuned"]))
    ):
        left = 0.4 + offset * 5.0
        axis.text(left + 1.8, 5.2, name, ha="center", fontsize=9.5, color=color)
        for row in range(2):
            for column in range(2):
                axis.add_patch(
                    plt.Rectangle(
                        (left + column * 1.8, 3.0 - row * 1.8),
                        1.7,
                        1.7,
                        facecolor=color,
                        alpha=0.10 + 0.22 * min(abs(matrix[row, column]) / 2.2, 1.0),
                        edgecolor=color,
                        lw=1.0,
                    )
                )
                axis.text(
                    left + column * 1.8 + 0.85,
                    3.0 - row * 1.8 + 0.85,
                    f"{matrix[row, column]:+.2f}",
                    ha="center",
                    va="center",
                    fontsize=10.5,
                    color=jp.INK,
                )
    gap = np.abs(jacobian - regression).max()
    axis.text(
        5.0,
        0.5,
        f"max |J − A| = {gap:.3f}",
        ha="center",
        fontsize=10,
        color=jp.INK if gap > 0.01 else jp.MUTED,
    )


if __name__ == "__main__":
    main()
