# Area-matched random-mask control

Do WeDetect-Uni regions change GeoCLIP more than an arbitrary region of the same
size? Produced by
[`experiments/eval_random_mask_control_colab.py`](../../experiments/eval_random_mask_control_colab.py),
run as a private Kaggle notebook (version 4, Tesla T4, FP32 with TF32 disabled,
1,404 s) on 2026-10-28.

For each of the 2,997 Img2GPS3k images, the Table 1 configuration (layers
{2, 7, 12}, a = +2/3, b = −2/3) is applied three ways in one batch:

| Arm | Mask |
|---|---|
| `union` | The exact Table 1 mask, from [`../layer_search/proposal_cache/proposals.jsonl`](../layer_search/proposal_cache/) |
| `shift` | That mask translated to a random non-zero grid offset with wrap-around; area and shape preserved. Mean of 5 draws |
| `scatter` | The same number of patches drawn uniformly. Mean of 5 draws |

Draws are seeded from the file name, so they are identical across runs.

## Sanity check

The `union` arm reproduces Table 1 on different hardware: holdout baseline mean
error 1716.31 km and clipped Δ −5.32 km on the 1,321 holdout images with an
active mask, both equal to the archived FP32 run.

## Results

Paired per-image differences, 95% bootstrap CI over images (10,000 resamples).
Positive means the left arm is larger.

| Contrast | Displacement, 2,058 active | Displacement, 1,321 holdout active | Improvement Δ, 2,058 active |
|---|---|---|---|
| `union` − `shift` | +9.61 km [−45.58, 65.70] | +16.66 km [−53.54, 89.85] | −10.89 km [−59.08, 36.33] |
| `union` − `scatter` | +30.48 km [−22.91, 85.27] | +44.30 km [−21.72, 113.90] | +16.62 km [−30.51, 62.96] |
| `shift` − `scatter` | +20.86 km [−5.07, 47.23] | +27.64 km [−5.79, 62.87] | +27.50 km [5.91, 49.79] |

WeDetect regions neither move predictions further nor reduce error more than an
arbitrary region of the same area and shape. Contrasts are unclipped: the
controls are already means over five draws while `union` is a single run, so
clipping at ±2500 km would truncate `union` more often and bias the contrast.

## Files

| File | Contents |
|---|---|
| `summary.json` | Configuration, per-arm metrics and all contrasts for the `active` and `holdout_active` subsets |
| `per_image.csv` | One row per image: split, baseline, and distance / improvement / displacement for each arm (`shift`, `scatter` averaged over draws) |
| `execution.log` | Kaggle stdout/stderr of the run |
