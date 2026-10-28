# Region specificity of the attention intervention

Is GeoCLIP's response specific to the region the detector found? Produced by
[`experiments/eval_region_specificity_colab.py`](../../experiments/eval_region_specificity_colab.py)
(private Kaggle notebook, version 2, Tesla T4, FP32 with TF32 disabled) on
2026-10-28; coverage breakdown by
[`experiments/analyze_region_specificity.py`](../../experiments/analyze_region_specificity.py).
Supersedes [`../random_mask_control`](../random_mask_control/): same masks, but
every forward pass is logged, with embedding cosine and top-1 change.

| Mode | Config | Target | Control (5 draws) |
|---|---|---|---|
| `union` | Table 1: layers {2, 7, 12}, a = +2/3, b = −2/3 | Table 1 mask (union of WeDetect-Uni boxes ≥ 0.4) | `shift`: same mask, random wrapped offset; `scatter`: same patch count, uniform |
| `box` | Layer 2, a = −2, b = 0 (suppression, as in Fig. 1) | One box ≥ 0.4 | `background`: same rectangle, placed where it overlaps no proposal ≥ 0.15 |

Contrasts: per unit, target minus the mean of its controls, averaged per image,
95% bootstrap CI over images (10,000 resamples). `*` = CI excludes 0. The
coverage bins were fixed before this run; the ≤10% displacement result had
already been seen on the previous run.

Sanity check: `union` target reproduces Table 1 (holdout-active mean
displacement 275.18 km; the `random_mask_control` contrasts to the cent).

## `union`, 2,058 active images

Arm means: 1 − cosine 0.0023 / 0.0020 / 0.0016, top-1 changed 14.9% / 14.5% /
12.8% (target / shift / scatter).

| Coverage | n | target − shift, 1 − cos (×10⁻⁴) | target − shift, top-1 changed | target − scatter, 1 − cos (×10⁻⁴) |
|---|---|---|---|---|
| all | 2058 | +3.0 [2.5, 3.6]* | +0.4 pp [−0.9, 1.7] | +7.6 [7.0, 8.2]* |
| ≤10% | 319 | +5.7 [4.4, 7.1]* | +2.3 pp [−0.9, 5.6] | +6.5 [5.2, 7.9]* |
| 10–25% | 327 | +4.9 [3.1, 6.8]* | +3.7 pp [0.3, 7.5]* | +9.4 [7.8, 11.3]* |
| 25–50% | 438 | +2.6 [1.2, 4.1]* | −0.1 pp [−3.3, 3.2] | +11.6 [10.0, 13.2]* |
| 50–80% | 457 | +2.3 [1.4, 3.4]* | −1.4 pp [−4.3, 1.6] | +9.3 [8.2, 10.5]* |
| >80% | 517 | +1.1 [0.6, 1.6]* | −0.9 pp [−2.7, 0.8] | +2.1 [1.6, 2.7]* |

Holdout (1,321) agrees: target − shift 1 − cos +2.8 [2.2, 3.4]×10⁻⁴, positive
in every bin. Accuracy (Δ) differs in no pooled contrast except shift − scatter
(+27.5 km [5.9, 49.8]).

Confound: wrapped shifts split masks (mean connected components 1.64 vs 1.29
at ≤10%, up to 2.59 vs 1.36 at 25–50%) and sit slightly further from the image
centre. The target − shift gap is largest where these differences are smallest
(≤10%), but a shape/centrality share cannot be excluded.

## `box`, 2,314 boxes in 899 images

3,334 boxes had no free background position and were skipped, so the test
covers mostly small boxes (median 9 patches).

| Metric | target − background, all | holdout (1,415) |
|---|---|---|
| 1 − cosine (×10⁻⁵) | +3.3 [2.3, 4.4]* | +3.5 [2.2, 4.8]* |
| top-1 changed | +1.3 pp [0.4, 2.2]* (2.5% vs 1.6%) | +1.2 pp [0.1, 2.4]* |
| displacement | +25.1 km [3.0, 53.6]* | +20.3 km [0.4, 51.8]* |
| Δ (improvement) | −4.0 km [−27.1, 17.3] | −11.5 km [−36.5, 6.6] |

The cosine effect comes from boxes larger than 12 patches; boxes of ≤4 patches
show none. Background placements are further from the centre (mean radius 6.40
vs 5.67 patches).

## Files

`union/`, `box/`: `draws.csv` (one row per forward pass), `summary.json`,
`done_images.txt` (image → boxes skipped). `execution.log`: Kaggle output.
