# Per-layer discovery/holdout search

This directory contains the original Kaggle sources for Section 4 and Table 1.
The complete exported run artifacts are in
[`results/layer_search`](../../results/layer_search/README.md).
The confirmatory run is `giahuytran1/wedetect-geoclip-layer-synergy`, version 3,
output session `347899775`: deterministic FP32, TF32 disabled, seed 42,
1,081.9358 seconds on an NVIDIA RTX PRO 6000 Blackwell Server Edition.

## Check the published results locally

From the repository root, on a CPU:

```bash
python -m pip install numpy pandas
python experiments/layer_search/verify_results.py
```

This checks the SHA-256 archive manifest before loading the trusted NPZ files,
the proposal/image ordering, the filename-hash split, the 83 configurations
across all three splits, and all 249 aggregate rows against the per-image
arrays. It recomputes 10,000-resample bootstrap intervals with seed 42 and writes:

* `results/layer_search/derived/table1.csv`;
* `results/layer_search/derived/split_assignment.csv` (one row per image,
  including proposal count, active patch count and coverage);
* `results/layer_search/derived/verification.json`.

The split rule is
`int(sha256(name.encode()).hexdigest()[:8], 16) % 3 == 0` for discovery;
all remaining names form holdout. This produces 1,079/1,918 images.

| Configuration | Discovery clipped delta (km) | Holdout error (km) | Holdout clipped delta (km) | 95% CI (km) | Shift (km) | Cosine |
|---|---:|---:|---:|---|---:|---:|
| GeoCLIP baseline | 0.00 | 1716.31 | 0.00 | — | 0.00 | 1.0000 |
| Layer 2 | +4.95 | 1731.44 | -4.30 | [-18.83, 9.96] | 136.15 | 0.9993 |
| Layers 2, 7, 12 (`2/k`) | +11.94 | 1728.86 | -3.66 | [-20.43, 12.40] | 189.53 | 0.9984 |

Delta is `baseline_error - intervention_error`, clipped per image to
[-2500, 2500] km before averaging; positive means improvement. Configuration
selection uses discovery only. The selected combination's paired advantage over
layer 2 is +0.6358 km, with 95% CI [-16.8324, 17.5709] km, so this run does not
support a held-out improvement from combining these layers.

## Original sources

| Python export | Kaggle notebook source | Purpose |
|---|---|---|
| [`prepare_offline_assets.py`](prepare_offline_assets.py) | [`notebooks/prepare_offline_assets.ipynb`](notebooks/prepare_offline_assets.ipynb) | Download model assets and prepare the offline bundle |
| [`cache_proposals.py`](cache_proposals.py) | [`notebooks/cache_proposals.ipynb`](notebooks/cache_proposals.ipynb) | Cache WeDetect proposals once at threshold 0.15 and NMS IoU 0.7 |
| [`search_signals.py`](search_signals.py) | [`notebooks/search_signals.ipynb`](notebooks/search_signals.ipynb) | Earlier mixed-precision screening; discovery ranks candidate layers 2, 4, 7, 12, 3 |
| [`search_synergy.py`](search_synergy.py) | [`notebooks/search_synergy.ipynb`](notebooks/search_synergy.ipynb) | Confirmatory FP32 evaluation of 78 combinations and five single-layer controls |

The Python exports contain the original notebook code cells. Run the notebooks
in Kaggle/IPython, where `display` is provided. The source notebooks have no
captured outputs; original stdout/stderr is archived separately in each
`execution.log`. These are source exports, not newly executed notebooks.

## Rerun inference

1. Obtain the Img2GPS3k dataset `lbgan2000/imgps3k-yfcc4k-cleaned` on Kaggle.
   The scripts locate `im2gps3k_places365.csv` below `/kaggle/input` and expect
   its sibling `im2gps3ktest/im2gps3ktest/` image directory.
2. Run the offline-assets notebook with internet enabled. Its output contains
   `offline_assets/asset_manifest.json`, `clip-vit-large-patch14/`, `geo-clip/`
   and the WeDetect assets. Attach that output to the inference notebook as an
   input. The archived [`asset_manifest.json`](../../results/layer_search/asset_manifest.json)
   records the original model sources, WeDetect revisions and available hashes.
3. For the archived confirmatory configuration, attach
   `results/layer_search/proposal_cache/` and
   `results/layer_search/candidate_screening/` as Kaggle inputs, preserving the
   filenames. These supply `proposals.jsonl`, `search_summary.csv` and
   `search_arrays.npz`. Together with the original images and offline model
   bundle, they are all the inputs read by `search_synergy.ipynb`.
4. Select an NVIDIA RTX PRO 6000 Blackwell GPU, disable internet, and run
   `notebooks/search_synergy.ipynb`. The original code explicitly checks this GPU
   and writes to `/kaggle/working/wedetect_geoclip_layer_synergy/`.
5. To regenerate the entire pipeline instead, run the cache notebook, then the
   signal-search notebook, then the synergy notebook, attaching each stage's
   output to its successor. A changed screening run can select different layers;
   use the archived screening inputs to retain the original candidate selection.

The runtime log records PyTorch `2.10.0+cu128`, CUDA 12.8 and GPU capability
(12, 0). Other imported packages include NumPy, pandas, Pillow, matplotlib,
transformers, and torchvision; asset preparation also uses huggingface_hub,
ONNX and WeDetect export dependencies. A full original environment lock was not
exported. The original asset preparation code does not pin every upstream
revision, so the archived hashes and inputs are the reference for comparison.

## Scope of the archive

All files exported by the FP32 run, proposal-cache run and candidate-screening
run are included, together with their logs and source notebooks. Model weights
and benchmark images remain setup inputs. The original FP32 NPZ stores
per-image deltas, shifts and cosines, but does **not** store per-image absolute
errors, predicted GPS coordinates or baseline embeddings. No such outputs are
invented here. The baseline mean in the derived Table 1 is inferred from the
archived mean intervention error plus mean delta; float32 rounding does not
affect its printed value of 1716.31 km.

Earlier screening uses a different numerical path and its baseline must not be
substituted for the confirmatory baseline. The recorded zero maximum feature
difference tests two consecutive FP32 encodings of eight images within one
process. It does not establish cross-process or cross-hardware reproducibility.
