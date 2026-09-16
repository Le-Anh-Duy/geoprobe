# Experiments

Dataset-scale evaluation of the intervention on all 2,997 Img2GPS3k images
(Kaggle handle `lbgan2000/imgps3k-yfcc4k-cleaned`).

Every run caches WeDetect-Uni proposals once and reuses them, so a layer or
strength change never changes the regions.

## Scripts

| File | Role |
|---|---|
| `eval_img2gps3k_wedetect_colab.py` | Full-dataset evaluation, written for `colab exec -f`. Exports WeDetect-Uni to ONNX inside the VM, checkpoints per image so an interrupted run resumes, and writes `summary.json` + `per_image.csv` |
| `analyze_img2gps3k_wedetect.py` | Turns a run directory into `analysis/`: outcome distribution, proposal-count and geographic breakdowns |
| `notebooks/geoclip_group_random_search_colab.ipynb` | Earlier random search over three contiguous layer groups (early 0–7, middle 8–15, late 16–23). Superseded by the per-layer search below; kept because it motivated it |
| `layer_search/` | The discovery/holdout per-layer search reported in the paper — **not yet committed**, see below |

```bash
python experiments/analyze_img2gps3k_wedetect.py \
    results/img2gps3k_wedetect_union_all_layers_a2_bneg2_threshold_0.4 \
    --output-dir results/img2gps3k_wedetect_union_all_layers_a2_bneg2_threshold_0.4/analysis
```

## Committed runs

All four use objectness threshold `0.4` and NMS IoU `0.7`, yielding proposals
in 2,050 of the 2,997 images.

| Directory | Configuration |
|---|---|
| `img2gps3k_wedetect_threshold_0.4` | Each proposal applied independently, layer 23, `a=1, b=0`. Supports the per-proposal and oracle rows |
| `img2gps3k_wedetect_union_threshold_0.4` | Union of proposals, layer 23, `a=1, b=0` |
| `img2gps3k_wedetect_union_all_layers_a2_bneg2_threshold_0.4` | Union, all 24 layers, `a=2, b=-2`, Tesla T4 |
| `..._g4_hf` | Same configuration re-run on an RTX PRO 6000 Blackwell |

Each directory holds `summary.json` (config, proposal statistics, aggregate
metrics), `per_image.csv` (per-image predictions and geodesic errors) and
`execution_log.ipynb` (the captured remote execution).

## Missing: the per-layer discovery/holdout search

Section 4 and Table 1 of the paper report a different, later experiment:

* proposals cached at threshold `0.4` giving **6,152 boxes in 2,081 images**,
  **2,058** of which keep a non-empty mask after CLIP's centre crop;
* a deterministic discovery/holdout split by file-name hash
  (**1,079** discovery, **1,918** holdout);
* all 26 subsets of two to five of the candidate layers `{2,4,7,12,3}` under
  three strength schedules (`2`, `2/sqrt(k)`, `2/k`), plus five single-layer
  controls — 78 combinations and 5 controls;
* deterministic FP32 with TF32 disabled, seed 42, 1,081.9 s on an
  NVIDIA RTX PRO 6000 Blackwell;
* selection score = mean of `clip(d_base - d_int, -2500, 2500)` km, with
  10,000 paired bootstrap resamples for confidence intervals.

Those proposal counts do not match the four committed runs above, so this
search used its own cached proposals and cannot be reconstructed from them.

To close the gap, add to `layer_search/`:

1. the search notebook or script that ran on Colab/Kaggle;
2. its cached proposal file (or the code path that regenerates it);
3. the per-combination results table, keyed by layer subset and schedule, with
   discovery and holdout scores;
4. the split assignment, or the hash function that derives it from file names.

Until then the paper's claim that the implementation and experiment scripts are
publicly available is only partly satisfied.
