# Table 1: archived deterministic FP32 run

These are the original exported outputs for the paper's discovery/holdout layer
search, retrieved from the authenticated Kaggle API on 2026-10-28.
See [`experiments/layer_search`](../../experiments/layer_search/README.md) for
source code, the protocol and instructions to verify or rerun the experiment.

| Directory/file | Contents |
|---|---|
| `fp32/result.json` | Selected configuration, locked-holdout metrics, bootstrap intervals, precision and timing |
| `fp32/synergy_summary.csv` | All 83 configurations on discovery, holdout and all images (249 rows) |
| `fp32/synergy_arrays.npz` | Image names, discovery mask, configuration order, and 2,997 × 83 arrays of delta, prediction shift and embedding cosine |
| `fp32/layer_synergy_overview.png` | Original holdout comparison and discovery-to-holdout plot |
| `fp32/execution.log` | Complete exported stdout/stderr events for Kaggle synergy v3 |
| `proposal_cache/` | Original `proposals.jsonl`, cache summary and execution log; at threshold 0.4 there are 6,152 boxes in 2,081 images, with 2,058 non-empty masks after cropping |
| `candidate_screening/` | All exported outputs and the execution log of the earlier mixed-precision search used to select candidate layers |
| `asset_manifest.json` | Original asset manifest, model sources, revisions and available hashes |
| `provenance.json` | Source notebook metadata, versions, FP32 output-session ID and archive limitations |
| `checksums.json` | SHA-256 and byte sizes of original artifacts, source exports and documentation; excludes itself and regenerable `derived/` files |
| `derived/` | Regenerated Table 1, explicit image split/mask assignments and verification report |

Positive delta means lower error than baseline. Only the `fp32/` results are
the confirmatory Table 1 run; the screening results are historical selection
inputs. The NPZ's `names` array has NumPy object dtype, so load this trusted
archive with `allow_pickle=True` after checksum verification. It does not include
absolute per-image FP32 errors or predicted coordinates, because the original
run did not export them.

```bash
python -m pip install numpy pandas
python experiments/layer_search/verify_results.py
```

The check needs no GPU or benchmark images. It verifies every aggregate delta,
shift and cosine against the NPZ arrays and regenerates the reported Table 1
confidence intervals. The run's raw artifact bytes are preserved; `derived/`
contains newly generated views of those artifacts.
