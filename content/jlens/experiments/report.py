"""Score the experiments against the preregistered criteria in CLAIMS.md.

Reads whatever result JSONs exist and prints a verdict table.  Every threshold
here is copied from CLAIMS.md and none of them may be edited after seeing a
result; if a criterion fails, the failure is the finding.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load(name: str):
    path = HERE / name
    return json.loads(path.read_text()) if path.exists() else None


def linear_fit_r2(x: np.ndarray, y: np.ndarray) -> float:
    design = np.stack([x, np.ones_like(x)], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    residual = y - design @ coefficients
    total = y - y.mean()
    return float(1.0 - residual.dot(residual) / total.dot(total))


def verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def report_claim1() -> None:
    data = load("multihop_three_lens_screen.json")
    if data is None:
        print("Claim 1: multihop screen not yet run")
        return
    summary = data["summary"]
    print("\n=== Claim 1: the three lenses answer different questions ===")
    print(f"items scored: {summary['n_items']}")
    print(f"{'lens':<12}{'pass@1':>9}{'pass@5':>9}{'pass@10':>9}{'median best A rank':>21}")
    for lens in ("j", "logit", "tuned"):
        row = summary["lenses"][lens]
        print(
            f"{lens:<12}{row['pass@1']:>9.2f}{row['pass@5']:>9.2f}"
            f"{row['pass@10']:>9.2f}{row['median_best_A_rank']:>21}"
        )
    aligned = summary["at_j_best_layer"]
    print("\nat the layer where the J-lens best recovers A:")
    for key, value in aligned.items():
        print(f"  {key:<32}{value}")

    j = summary["lenses"]["j"]
    print(
        f"\n  criterion: J-lens recovers A better than logit and tuned  -> "
        f"{verdict(j['pass@10'] > summary['lenses']['logit']['pass@10'] and j['pass@10'] > summary['lenses']['tuned']['pass@10'])}"
    )
    print(
        f"  criterion: Tuned lens prefers B over A where J-lens reads A -> "
        f"{verdict(aligned['fraction_tuned_prefers_B'] > 0.5)}"
    )


def report_claim2_maps() -> None:
    data = load("nonlinearity_vs_divergence.json")
    if data is None:
        print("\nClaim 2 (maps): nonlinearity test not yet run")
        return
    rows = data["rows"]
    print("\n=== Claim 2, criterion 2/3: does J_l differ from A_l, beyond noise? ===")
    print(f"{'layer':>6}{'cos(J, A_reg)':>16}{'split-half ceiling':>21}{'gap':>9}")
    for row in rows:
        gap = row["cos_J_splithalf"] - row["cos_J_vs_regression"]
        marker = "  <-- affine here" if gap < 0.05 else ""
        if row["layer"] % 4 == 0 or gap < 0.05:
            print(
                f"{row['layer']:>6}{row['cos_J_vs_regression']:>16.3f}"
                f"{row['cos_J_splithalf']:>21.3f}{gap:>9.3f}{marker}"
            )


def report_claim2_sweep() -> None:
    data = load("causal_sweep.json")
    if data is None:
        print("\nClaim 2 (sweep): causal sweep not yet run")
        return
    print("\n=== Claim 2, criteria 1/4/5: the intervention sweep ===")
    for record in data["results"]:
        print(f"\n-- {record['name']}  layer {record['layer']}  ({record['role']}), sites={record.get('sites')}")
        print(f"   baseline continuation: {record['baseline_continuation']!r}")
        for swap_name, sweep in record["sweeps"].items():
            alphas = np.array(sweep["alphas"])
            actual = np.array(sweep["actual"])
            window = (alphas >= 0) & (alphas <= 1)
            r2 = linear_fit_r2(alphas[window], actual[window])
            index_zero = int(np.argmin(np.abs(alphas)))
            index_one = int(np.argmin(np.abs(alphas - 1.0)))
            chord = float(actual[index_one] - actual[index_zero])
            tangent = sweep["pointwise_tangent_slope"]
            gate = abs(tangent - chord) / max(abs(tangent), abs(chord), 1e-9)
            logit_difference = np.array(sweep["true_logit_difference"])
            flipped = sweep.get("top_token", [None] * len(alphas))[index_one]
            print(
                f"   {swap_name:<7} R^2(linear, 0..1) = {r2:.3f}   tangent = {tangent:+.1f}   "
                f"chord = {chord:+.1f}   gate index = {gate:.2f}"
            )
            print(
                f"   {'':<7} true logit(B')-logit(B): {logit_difference[index_zero]:+.2f} "
                f"-> {logit_difference[index_one]:+.2f}   top token at alpha=1: {flipped!r}"
            )
        if record["role"] == "preswitch":
            bridge = record["sweeps"]["bridge"]
            answer = record["sweeps"]["answer"]
            alphas = np.array(bridge["alphas"])
            index_one = int(np.argmin(np.abs(alphas - 1.0)))
            index_zero = int(np.argmin(np.abs(alphas)))
            bridge_effect = abs(
                np.array(bridge["true_logit_difference"])[index_one]
                - np.array(bridge["true_logit_difference"])[index_zero]
            )
            answer_effect = abs(
                np.array(answer["true_logit_difference"])[index_one]
                - np.array(answer["true_logit_difference"])[index_zero]
            )
            print(
                f"   criterion 5 (timing): bridge swap moves {bridge_effect:.2f} nats, "
                f"answer swap moves {answer_effect:.2f} -> {verdict(bridge_effect > answer_effect)}"
            )


if __name__ == "__main__":
    report_claim1()
    report_claim2_maps()
    report_claim2_sweep()
