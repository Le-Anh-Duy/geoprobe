"""Is GeoCLIP's response to an attention intervention specific to the region?

    MODE=union python experiments/eval_region_specificity_colab.py
    MODE=box   python experiments/eval_region_specificity_colab.py

Every arm below is one forward pass, logged individually (``draws.csv``), so
accuracy, medians and top-1 change rates are comparable across arms. Arms of
one image share a batch, hence hardware, precision and gallery ties.

MODE=union (default config: Table 1, layers {2,7,12}, a=+2/3, b=-2/3)
    ``target``  the exact Table 1 mask (union of cached WeDetect-Uni boxes);
    ``shift``   that mask translated by a random non-zero offset, wrapping;
    ``scatter`` the same number of patches drawn uniformly.
    Same masks as ``eval_random_mask_control_colab.py``, now with per-draw
    outputs and embedding cosine.

MODE=box (default config: layer 2, a=-2, b=0, i.e. suppress the region as in
the demonstration)
    ``target``      one WeDetect-Uni box (score >= 0.4) at a time;
    ``background``  a rectangle of the same size placed where it overlaps no
                    proposal at all (score >= 0.15), so the control lies on
                    background rather than on another object.
    A box with no free background position is skipped and counted.

Contrasts are per unit (``target`` minus the mean of its controls), averaged
over the units of an image, then bootstrapped over images.

Expected assets (under /content, or /kaggle/working on Kaggle): an unpacked
geoclip_source/ (or geoclip_source.zip), intervention.py and proposals.jsonl.
Finished images are checkpointed, so re-running resumes.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import types
import zipfile
import zlib
from pathlib import Path

import numpy as np

MODE = os.getenv("MODE", "union")
if MODE not in ("union", "box"):
    raise ValueError(f"MODE must be 'union' or 'box', got {MODE!r}")
DEFAULTS = {"union": ("2,7,12", 2.0 / 3.0, -2.0 / 3.0), "box": ("2", -2.0, 0.0)}[MODE]
LAYER_SPEC = os.getenv("INTERVENTION_LAYERS", DEFAULTS[0])
ATTENTION_A = float(os.getenv("INTERVENTION_A", str(DEFAULTS[1])))
ATTENTION_B = float(os.getenv("INTERVENTION_B", str(DEFAULTS[2])))

DATASET_HANDLE = os.getenv("DATASET_HANDLE", "lbgan2000/imgps3k-yfcc4k-cleaned")
SCORE_THRESHOLD = 0.4   # Table 1's box rule: score >= 0.4, at most 100 boxes
PROPOSAL_TOP_K = 100
OBJECT_THRESHOLD = 0.15  # anything the cache kept counts as "not background"
REPEATS = int(os.getenv("RANDOM_REPEATS", "5"))
SEED = int(os.getenv("SEED", "42"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "48"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "50"))
BOOTSTRAP_RESAMPLES = int(os.getenv("BOOTSTRAP_RESAMPLES", "10000"))
CLIP_KM = 2500.0
THRESHOLDS_KM = (1, 25, 200, 750, 2500)
GRID = 16
PATCHES = GRID * GRID

CONTENT = Path(os.getenv(
    "WORK_DIR", "/kaggle/working" if Path("/kaggle/working").is_dir() else "/content"
))
SOURCE_ZIP = CONTENT / "geoclip_source.zip"
SOURCE_DIR = CONTENT / "geoclip_source"
PROPOSALS_PATH = CONTENT / "proposals.jsonl"
OUTPUT_DIR = CONTENT / os.getenv("OUTPUT_NAME", f"img2gps3k_region_specificity_{MODE}")
DRAWS_PATH = OUTPUT_DIR / "draws.csv"
DONE_PATH = OUTPUT_DIR / "done_images.txt"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"


# --------------------------------------------------------------------------
# Masks. Pure numpy, so this part self-checks before anything heavy loads.
# --------------------------------------------------------------------------

def boxes_mask(record: dict, threshold: float, top_k: int | None, only: int | None = None) -> np.ndarray:
    """``boxes_to_mask`` of layer_search/search_synergy.py, generalised.

    With threshold=0.4, top_k=100 and only=None this is exactly the Table 1
    mask. ``only`` rasterises a single box of that same selection.
    """
    width, height = record["width"], record["height"]
    selected = [p for p in record["proposals"] if p["score"] >= threshold][:top_k]
    if only is not None:
        selected = selected[only:only + 1]
    mask = np.zeros((GRID, GRID), dtype=bool)
    scale = 224 / min(width, height)
    crop_x, crop_y = (width * scale - 224) / 2, (height * scale - 224) / 2
    for item in selected:
        x1, y1, x2, y2 = item["box"]
        x1, x2 = max(0, x1 * scale - crop_x), min(224, x2 * scale - crop_x)
        y1, y2 = max(0, y1 * scale - crop_y), min(224, y2 * scale - crop_y)
        if x2 <= x1 or y2 <= y1:
            continue
        c0, c1 = int(x1 // 14), min(GRID, int((x2 - 1e-6) // 14) + 1)
        r0, r1 = int(y1 // 14), min(GRID, int((y2 - 1e-6) // 14) + 1)
        mask[r0:r1, c0:c1] = True
    return mask.reshape(PATCHES)


def table1_split(name: str) -> str:
    """The discovery/holdout rule of layer_search/search_synergy.py."""
    return "discovery" if int(hashlib.sha256(name.encode()).hexdigest()[:8], 16) % 3 == 0 else "holdout"


def rng_for(name: str, unit: int) -> np.random.Generator:
    # Not hash(): str hashing is salted per process and would change on resume.
    # For unit -1 this is the generator of eval_random_mask_control_colab.py.
    key = [SEED, zlib.crc32(name.encode("utf-8"))]
    return np.random.default_rng(key if unit < 0 else key + [unit])


def union_controls(union: np.ndarray, rng: np.random.Generator) -> tuple[list, list]:
    """Wrapped shifts and scatters, drawn in the same order as the first control."""
    count = int(union.sum())
    shifts, scatters = [], []
    for _ in range(REPEATS):
        while True:
            dy, dx = int(rng.integers(0, GRID)), int(rng.integers(0, GRID))
            if (dy, dx) != (0, 0):
                break
        grid = union.reshape(GRID, GRID)
        shifts.append(np.roll(np.roll(grid, dy, axis=0), dx, axis=1).reshape(PATCHES))
        scatter = np.zeros(PATCHES, dtype=bool)
        scatter[rng.choice(PATCHES, size=count, replace=False)] = True
        scatters.append(scatter)
    return shifts, scatters


def background_rectangles(box: np.ndarray, objects: np.ndarray, rng: np.random.Generator) -> list:
    """Up to REPEATS same-size rectangles that overlap no object patch.

    ``box`` is one rasterised box, hence a rectangle; placements stay fully
    inside the grid (no wrapping), so the control has exactly its shape.
    """
    rows, cols = np.nonzero(box.reshape(GRID, GRID))
    h, w = rows.max() - rows.min() + 1, cols.max() - cols.min() + 1
    occupied = objects.reshape(GRID, GRID)
    free = [
        (r, c)
        for r in range(GRID - h + 1)
        for c in range(GRID - w + 1)
        if not occupied[r:r + h, c:c + w].any()
    ]
    picks = rng.choice(len(free), size=min(REPEATS, len(free)), replace=False) if free else []
    masks = []
    for index in picks:
        r, c = free[index]
        mask = np.zeros((GRID, GRID), dtype=bool)
        mask[r:r + h, c:c + w] = True
        masks.append(mask.reshape(PATCHES))
    return masks


def image_units(name: str, record: dict) -> tuple[list[dict], int]:
    """(units, skipped). A unit is one target mask with its control masks."""
    if MODE == "union":
        union = boxes_mask(record, SCORE_THRESHOLD, PROPOSAL_TOP_K)
        if not union.any():
            return [], 0
        shifts, scatters = union_controls(union, rng_for(name, -1))
        return [{"unit": 0, "target": union, "shift": shifts, "scatter": scatters}], 0

    objects = boxes_mask(record, OBJECT_THRESHOLD, None)
    count = sum(p["score"] >= SCORE_THRESHOLD for p in record["proposals"][:PROPOSAL_TOP_K])
    units, seen, skipped = [], set(), 0
    for index in range(count):
        box = boxes_mask(record, SCORE_THRESHOLD, PROPOSAL_TOP_K, only=index)
        key = box.tobytes()
        if not box.any() or key in seen:
            continue
        seen.add(key)
        controls = background_rectangles(box, objects, rng_for(name, index))
        if not controls:
            skipped += 1
            continue
        units.append({"unit": index, "target": box, "background": controls})
    return units, skipped


def _self_check() -> None:
    record = {"width": 224, "height": 224, "proposals": [
        {"box": [0, 0, 56, 42], "score": 0.9},      # 3x4 patches, top-left
        {"box": [0, 0, 56, 42], "score": 0.8},      # duplicate mask
        {"box": [140, 140, 196, 182], "score": 0.5},
        {"box": [0, 196, 224, 224], "score": 0.2},  # object, never a target
    ]}
    union = boxes_mask(record, SCORE_THRESHOLD, PROPOSAL_TOP_K)
    assert union.sum() == 12 + 12
    one = boxes_mask(record, SCORE_THRESHOLD, PROPOSAL_TOP_K, only=0)
    assert one.reshape(GRID, GRID)[0:3, 0:4].all() and one.sum() == 12
    objects = boxes_mask(record, OBJECT_THRESHOLD, None)
    assert objects.sum() == 12 + 12 + 2 * GRID

    for unit in (0, 2):
        box = boxes_mask(record, SCORE_THRESHOLD, PROPOSAL_TOP_K, only=unit)
        controls = background_rectangles(box, objects, rng_for("a.jpg", unit))
        assert len(controls) == REPEATS
        for mask in controls:
            assert mask.sum() == box.sum(), "same area"
            assert not (mask & objects).any(), "must lie on background"
        again = background_rectangles(box, objects, rng_for("a.jpg", unit))
        assert all(np.array_equal(x, y) for x, y in zip(controls, again)), "reproducible"

    full = np.ones(PATCHES, dtype=bool)
    assert background_rectangles(one, full, rng_for("a.jpg", 0)) == []

    shifts, scatters = union_controls(union, rng_for("a.jpg", -1))
    assert all(m.sum() == union.sum() for m in shifts + scatters)
    assert not any(np.array_equal(m, union) for m in shifts)

    global MODE
    saved, MODE = MODE, "box"
    units, skipped = image_units("a.jpg", record)
    MODE = saved
    assert [u["unit"] for u in units] == [0, 2] and skipped == 0, "duplicate box dropped"


_self_check()


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

import kagglehub  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageFile  # noqa: E402
from transformers import AutoProcessor  # noqa: E402

ImageFile.LOAD_TRUNCATED_IMAGES = True
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_source() -> None:
    if not (SOURCE_DIR / "geoclip" / "__init__.py").is_file():
        if not SOURCE_ZIP.is_file():
            raise FileNotFoundError(f"Missing {SOURCE_DIR} and {SOURCE_ZIP}")
        with zipfile.ZipFile(SOURCE_ZIP) as archive:
            archive.extractall(SOURCE_DIR)
    if (CONTENT / "intervention.py").is_file() and not (SOURCE_DIR / "intervention.py").is_file():
        (SOURCE_DIR / "intervention.py").write_bytes((CONTENT / "intervention.py").read_bytes())
    sys.path.insert(0, str(SOURCE_DIR))


def haversine_km(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    predicted = np.asarray(predicted, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    lat1, lon1 = np.radians(target[..., 0]), np.radians(target[..., 1])
    lat2, lon2 = np.radians(predicted[..., 0]), np.radians(predicted[..., 1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    value = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 6371.0088 * 2 * np.arctan2(np.sqrt(value), np.sqrt(np.maximum(0, 1 - value)))


def enable_batched_masks(state) -> None:
    """Lets every item of a batch carry its own mask."""

    def key_bias_for_layer(self, layer_idx: int, num_positions: int):
        a, b = self.layer_ab.get(layer_idx, (0.0, 0.0))
        if (a == 0.0 and b == 0.0) or self.in_region_mask is None:
            return None
        mask = self.in_region_mask
        bias = torch.full((mask.shape[0], num_positions), b, dtype=torch.float32, device=mask.device)
        bias[:, 0] = 0.0
        bias[:, 1:].masked_fill_(mask, a)
        return bias[:, None, None, :]

    state.key_bias_for_layer = types.MethodType(key_bias_for_layer, state)


ensure_source()
from geoclip import GeoCLIP  # noqa: E402
import intervention  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("This evaluation requires a GPU runtime")
device = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = False  # as in the Table 1 FP32 pass
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)

dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
csv_path = dataset_path / "test_set" / "im2gps3k_places365.csv"
image_dir = dataset_path / "test_set" / "im2gps3ktest" / "im2gps3ktest"
frame = pd.read_csv(csv_path)
frame["image_path"] = frame["name"].map(lambda name: str(image_dir / name))
proposals = {
    record["name"]: record
    for record in map(json.loads, PROPOSALS_PATH.read_text(encoding="utf-8").splitlines())
}
if set(proposals) != set(frame["name"]):
    raise RuntimeError("proposals.jsonl does not cover exactly the Img2GPS3k images")
if MAX_IMAGES > 0:
    frame = frame.iloc[:MAX_IMAGES]
frame = frame.reset_index(drop=True)
targets = frame[["LAT", "LON"]].to_numpy(dtype=np.float64)

model = GeoCLIP().to(device).eval()
model.image_encoder.image_processor = AutoProcessor.from_pretrained(
    "openai/clip-vit-large-patch14", use_fast=False
)
state = intervention.patch_vision_tower(model.image_encoder.CLIP)
enable_batched_masks(state)
active_layers = [int(v) for v in LAYER_SPEC.split(",") if v.strip()]
with torch.inference_mode():
    gallery_features = F.normalize(model.location_encoder(model.gps_gallery.to(device)), dim=1)
gallery_cpu = model.gps_gallery.cpu().numpy()
print("CONFIG", json.dumps({
    "mode": MODE, "layers": active_layers, "a": ATTENTION_A, "b": ATTENTION_B,
    "repeats": REPEATS, "seed": SEED, "max_images": MAX_IMAGES,
    "gpu": torch.cuda.get_device_name(0),
}))


def encode(pixels: torch.Tensor, masks: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(normalised features, gallery indices); masks=None is the baseline."""
    if masks is None:
        state.layer_ab, state.in_region_mask = {}, None
    else:
        state.layer_ab = {layer: (ATTENTION_A, ATTENTION_B) for layer in active_layers}
        state.in_region_mask = torch.from_numpy(masks).to(device)
    try:
        with torch.inference_mode():
            features = F.normalize(model.image_encoder(pixels.to(device)), dim=1)
            indices = (features @ gallery_features.T).argmax(dim=1)
    finally:
        state.layer_ab, state.in_region_mask = {}, None
    return features.cpu().numpy(), indices.cpu().numpy()


def load_pixels(paths) -> torch.Tensor:
    images = []
    for path in paths:
        with Image.open(path) as source:
            images.append(source.convert("RGB"))
    return model.image_encoder.image_processor(images=images, return_tensors="pt")["pixel_values"]


baseline_features = np.empty((len(frame), gallery_features.shape[1]), np.float32)
baseline_index = np.empty(len(frame), np.int64)
for start in range(0, len(frame), 64):
    stop = min(start + 64, len(frame))
    baseline_features[start:stop], baseline_index[start:stop] = encode(
        load_pixels(frame["image_path"][start:stop]), None
    )
baseline_gps = gallery_cpu[baseline_index]
baseline_distance = haversine_km(baseline_gps, targets)
print("BASELINE_DONE", {"mean_error_km": float(baseline_distance.mean())})

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------

rows: list[dict] = pd.read_csv(DRAWS_PATH).to_dict("records") if DRAWS_PATH.is_file() else []
done: dict[str, int] = (
    json.loads(DONE_PATH.read_text()) if DONE_PATH.is_file() else {}
)  # image name -> units skipped for lack of background
print("RESUME", {"images": len(done), "rows": len(rows)})


def checkpoint() -> None:
    pd.DataFrame(rows).to_csv(DRAWS_PATH, index=False)
    DONE_PATH.write_text(json.dumps(done))


run_start = time.perf_counter()
for row_index, row in frame.iterrows():
    name = str(row["name"])
    if name in done:
        continue
    units, skipped = image_units(name, proposals[name])
    if units:
        arms = [arm for arm in ("target", "shift", "scatter", "background") if arm in units[0]]
        labels, masks = [], []
        for unit in units:
            for arm in arms:
                group = [unit["target"]] if arm == "target" else unit[arm]
                for draw, mask in enumerate(group):
                    labels.append((unit, arm, draw))
                    masks.append(mask)
        pixels = load_pixels([row["image_path"]])
        features, indices = [], []
        for start in range(0, len(masks), BATCH_SIZE):
            chunk = np.stack(masks[start:start + BATCH_SIZE])
            f, i = encode(pixels.expand(len(chunk), -1, -1, -1), chunk)
            features.append(f)
            indices.append(i)
        features, indices = np.concatenate(features), np.concatenate(indices)
        gps = gallery_cpu[indices]
        distance = haversine_km(gps, targets[row_index])
        displacement = haversine_km(gps, baseline_gps[row_index])
        cosine = features @ baseline_features[row_index]
        for k, (unit, arm, draw) in enumerate(labels):
            rows.append({
                "name": name,
                "split": table1_split(name),
                "unit": unit["unit"],
                "unit_patches": int(unit["target"].sum()),
                "arm": arm,
                "draw": draw,
                "baseline_distance_km": float(baseline_distance[row_index]),
                "distance_km": float(distance[k]),
                "displacement_km": float(displacement[k]),
                "cosine": float(cosine[k]),
                "changed": bool(indices[k] != baseline_index[row_index]),
            })
    done[name] = skipped
    if len(done) % CHECKPOINT_EVERY == 0 or len(done) == len(frame):
        checkpoint()
        print(f"PROGRESS {len(done)}/{len(frame)} rows={len(rows)} "
              f"elapsed={time.perf_counter() - run_start:.1f}s", flush=True)
checkpoint()

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------

draws = pd.DataFrame(rows)
draws["delta_km"] = draws["baseline_distance_km"] - draws["distance_km"]
draws["clipped_delta_km"] = draws["delta_km"].clip(-CLIP_KM, CLIP_KM)  # per draw, so fair
draws["one_minus_cosine"] = 1.0 - draws["cosine"]
draws["changed"] = draws["changed"].astype(float)
METRICS = ("one_minus_cosine", "changed", "displacement_km", "delta_km", "clipped_delta_km")
PAIRS = {"union": [("target", "shift"), ("target", "scatter"), ("shift", "scatter")],
         "box": [("target", "background")]}[MODE]


def bootstrap(values: np.ndarray) -> list[float]:
    rng = np.random.default_rng(SEED)
    means = values[rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarise(subset: pd.DataFrame) -> dict:
    arms = {}
    for arm, group in subset.groupby("arm"):
        arms[arm] = {
            "draws": int(len(group)),
            **{f"mean_{m}": float(group[m].mean()) for m in METRICS},
            "median_displacement_km": float(group["displacement_km"].median()),
            "mean_distance_km": float(group["distance_km"].mean()),
            "median_distance_km": float(group["distance_km"].median()),
            "accuracy": {f"acc_{t}_km": float((group["distance_km"] <= t).mean()) for t in THRESHOLDS_KM},
        }
    # Per unit: mean over draws of each arm; per image: mean over its units.
    per_unit = subset.groupby(["name", "unit", "arm"])[list(METRICS)].mean().unstack("arm")
    contrasts = {}
    for left, right in PAIRS:
        contrasts[f"{left}_vs_{right}"] = {}
        for metric in METRICS:
            diff = (per_unit[(metric, left)] - per_unit[(metric, right)]).groupby(level="name").mean()
            values = diff.to_numpy(dtype=np.float64)
            low, high = bootstrap(values)
            contrasts[f"{left}_vs_{right}"][metric] = {
                "mean_difference": float(values.mean()), "ci95": [low, high],
                "significant": bool(low > 0 or high < 0), "images": int(len(values)),
            }
    return {
        "images": int(subset["name"].nunique()),
        "units": int(subset.groupby(["name", "unit"]).ngroups),
        "baseline_mean_distance_km": float(subset.groupby("name")["baseline_distance_km"].first().mean()),
        "arms": arms,
        "contrasts": contrasts,
    }


summary = {
    "mode": MODE,
    "config": {"layers": active_layers, "a": ATTENTION_A, "b": ATTENTION_B, "repeats": REPEATS,
               "seed": SEED, "score_threshold": SCORE_THRESHOLD, "object_threshold": OBJECT_THRESHOLD,
               "proposal_top_k": PROPOSAL_TOP_K, "clip_km": CLIP_KM, "tf32": False,
               "bootstrap_resamples": BOOTSTRAP_RESAMPLES},
    "gpu": torch.cuda.get_device_name(0),
    "evaluated_images": len(done),
    "units_skipped_no_background": int(sum(done.values())),
    "baseline_mean_distance_km_all_images": float(baseline_distance.mean()),
    "subsets": {
        "all": summarise(draws),
        "holdout": summarise(draws[draws["split"] == "holdout"]),
    },
    "runtime_seconds": time.perf_counter() - run_start,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("SUMMARY", json.dumps(summary, indent=2))
