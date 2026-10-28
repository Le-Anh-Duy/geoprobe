"""Verify the archived FP32 run and regenerate Table 1 without a GPU.

Run from any directory: python experiments/layer_search/verify_results.py
Requires NumPy and pandas. Only the trusted, checksummed archive is unpickled.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/layer_search"


def bootstrap_ci(values):
    rng = np.random.default_rng(42)
    means = np.empty(10000)
    for index in range(len(means)):
        means[index] = values[rng.integers(0, len(values), len(values))].mean()
    return np.quantile(means, [0.025, 0.975]).tolist()


def proposal_mask(record):
    """NumPy equivalent of the original script's CLIP patch-mask geometry."""
    width, height = record["width"], record["height"]
    selected = [p for p in record["proposals"] if p["score"] >= 0.4][:100]
    mask = np.zeros((16, 16), dtype=bool)
    scale = 224 / min(width, height)
    crop_x, crop_y = (width * scale - 224) / 2, (height * scale - 224) / 2
    for item in selected:
        x1, y1, x2, y2 = item["box"]
        x1, x2 = max(0, x1 * scale - crop_x), min(224, x2 * scale - crop_x)
        y1, y2 = max(0, y1 * scale - crop_y), min(224, y2 * scale - crop_y)
        if x2 <= x1 or y2 <= y1:
            continue
        c0, c1 = int(x1 // 14), min(16, int((x2 - 1e-6) // 14) + 1)
        r0, r1 = int(y1 // 14), min(16, int((y2 - 1e-6) // 14) + 1)
        mask[r0:r1, c0:c1] = True
    return len(selected), int(mask.sum())


def main():
    manifest = json.loads((RESULTS / "checksums.json").read_text(encoding="utf-8"))
    for name, expected in manifest.items():
        path = ROOT / name
        assert path.stat().st_size == expected["bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected["sha256"], name

    with np.load(RESULTS / "fp32/synergy_arrays.npz", allow_pickle=True) as archive:
        arrays = {name: archive[name] for name in archive.files}
    with np.load(RESULTS / "candidate_screening/search_arrays.npz", allow_pickle=True) as archive:
        assert np.array_equal(arrays["names"], archive["names"])
        assert np.array_equal(arrays["discovery"], archive["discovery"])
    names, discovery = arrays["names"], arrays["discovery"]
    assert len(names) == len(set(names)) == 2997
    expected_split = np.asarray([
        int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 3 == 0 for name in names
    ])
    assert np.array_equal(discovery, expected_split)
    assert discovery.sum() == 1079 and (~discovery).sum() == 1918
    assert arrays["delta"].shape == arrays["shift"].shape == arrays["feature_cosine"].shape == (2997, 83)
    for key in ("delta", "shift", "feature_cosine"):
        assert np.isfinite(arrays[key]).all(), key

    records = [json.loads(line) for line in
               (RESULTS / "proposal_cache/proposals.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["name"] for r in records] == names.tolist()
    counts, patches = np.asarray([proposal_mask(r) for r in records]).T
    active = patches > 0
    assert counts.sum() == 6152 and (counts > 0).sum() == 2081
    assert active.sum() == 2058 and (active & ~discovery).sum() == 1321
    assert np.all(arrays["delta"][~active] == 0)
    assert np.all(arrays["shift"][~active] == 0)

    summary = pd.read_csv(RESULTS / "fp32/synergy_summary.csv")
    prior = pd.read_csv(RESULTS / "candidate_screening/search_summary.csv")
    ranked = prior.query("family == 'single_layer' and split == 'discovery' and clipped_mean_delta_km > 0").sort_values("clipped_mean_delta_km", ascending=False)
    assert [int(name.split("_")[-1]) for name in ranked.config] == [2, 4, 7, 12, 3]
    assert len(summary) == 249 and not summary.duplicated(["split", "config"]).any()
    config_names = arrays["config_names"].tolist()
    splits = {"discovery": discovery, "holdout": ~discovery, "all": np.ones(len(names), bool)}
    for split_name, split in splits.items():
        rows = summary[summary.split == split_name].set_index("config").loc[config_names]
        assert (rows.n == split.sum()).all()
        assert (rows.kind == "single_control").sum() == 5
        assert (rows.kind == "layer_combination").sum() == 78
        for key, column in (("delta", "mean_delta_km"), ("shift", "mean_shift_km"),
                            ("feature_cosine", "mean_feature_cosine")):
            # Per-column reduction preserves the float32 accumulation used in the run.
            actual = [arrays[key][split, i].mean() for i in range(83)]
            np.testing.assert_allclose(actual, rows[column], rtol=1e-6, atol=1e-5)
        clipped = [np.clip(arrays["delta"][split, i], -2500, 2500).mean() for i in range(83)]
        np.testing.assert_allclose(clipped, rows.clipped_mean_delta_km, atol=1e-5)

    result = json.loads((RESULTS / "fp32/result.json").read_text(encoding="utf-8"))
    assert result["evaluation_precision"] == "deterministic_fp32_no_tf32"
    assert result["determinism_max_abs_feature_difference"] == 0
    selected = summary.query("split == 'discovery' and kind == 'layer_combination'").sort_values("clipped_mean_delta_km", ascending=False).iloc[0].config
    assert selected == result["selected_on_discovery"]["config"] == "linear_layers_02_07_12"
    single = "fixed_layers_02"
    held = summary[summary.split == "holdout"].set_index("config")
    disc = summary[summary.split == "discovery"].set_index("config")
    selected_delta = np.clip(arrays["delta"][~discovery, config_names.index(selected)], -2500, 2500)
    single_delta = np.clip(arrays["delta"][~discovery, config_names.index(single)], -2500, 2500)
    selected_ci, single_ci = bootstrap_ci(selected_delta), bootstrap_ci(single_delta)
    paired_ci = bootstrap_ci(selected_delta - single_delta)
    np.testing.assert_allclose(selected_ci, result["locked_synergy_clipped_delta_ci95_km"], atol=1e-6)
    np.testing.assert_allclose(paired_ci, result["locked_synergy_minus_single_ci95_km"], atol=1e-6)
    # Baseline mean follows from E[base - intervention] + E[intervention].
    # The original run did not save per-image baseline errors in its NPZ.
    baseline_estimates = held.mean_error_km + held.mean_delta_km
    assert np.ptp(baseline_estimates) < 0.001
    baseline_mean = float(baseline_estimates.loc[selected])
    table = [{"configuration": "GeoCLIP baseline", "discovery_clipped_delta_km": 0.0,
              "holdout_mean_error_km": baseline_mean, "holdout_clipped_delta_km": 0.0,
              "ci95_low_km": None, "ci95_high_km": None,
              "holdout_mean_shift_km": 0.0, "holdout_mean_feature_cosine": 1.0}]
    for config, label, ci in ((single, "Layer 2", single_ci), (selected, "Layers 2, 7, 12 (2/k)", selected_ci)):
        row = held.loc[config]
        table.append({"configuration": label,
                      "discovery_clipped_delta_km": float(disc.loc[config].clipped_mean_delta_km),
                      "holdout_mean_error_km": float(row.mean_error_km),
                      "holdout_clipped_delta_km": float(row.clipped_mean_delta_km),
                      "ci95_low_km": ci[0], "ci95_high_km": ci[1],
                      "holdout_mean_shift_km": float(row.mean_shift_km),
                      "holdout_mean_feature_cosine": float(row.mean_feature_cosine)})
    assert [f"{row['holdout_mean_error_km']:.2f}" for row in table] == ["1716.31", "1731.44", "1728.86"]
    assert [f"{v:.2f}" for v in single_ci] == ["-18.83", "9.96"]
    assert [f"{v:.2f}" for v in selected_ci] == ["-20.43", "12.40"]

    derived = RESULTS / "derived"
    derived.mkdir(exist_ok=True)
    table_frame = pd.DataFrame(table)
    table_frame.to_csv(derived / "table1.csv", index=False, lineterminator="\n")
    pd.DataFrame({"name": names, "split": np.where(discovery, "discovery", "holdout"),
                  "proposal_count_at_0_4": counts, "active_patches": patches,
                  "mask_coverage": patches / 256}).to_csv(derived / "split_assignment.csv", index=False, lineterminator="\n")
    report = {"images": 2997, "configurations": 83, "summary_rows_verified": 249,
              "selected_config": selected, "paired_clipped_gain_km": float((selected_delta - single_delta).mean()),
              "paired_ci95_km": paired_ci,
              "active_holdout_images": int((active & ~discovery).sum()),
              "active_selected_clipped_delta_km": float(selected_delta[active[~discovery]].mean()),
              "baseline_mean_note": "Inferred from archived mean error + mean delta; no per-image FP32 baseline was exported."}
    (derived / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(table_frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"Verified {len(manifest)} checksums, 2,997 image assignments and all 249 summary rows.")


if __name__ == "__main__":
    main()
