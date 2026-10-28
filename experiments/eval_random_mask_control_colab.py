"""Area-matched random-region control for the attention intervention.

    colab exec -f experiments/eval_random_mask_control_colab.py

Section 4 of the paper states that "whether WeDetect regions are more
informative than arbitrary ones requires an area-matched random-mask control".
This is that control.

For every image the script runs the same intervention three ways and compares
them pairwise on identical hardware in one pass, so precision and
nearest-neighbour ties cannot differ between arms:

* ``union``   -- the union of the WeDetect-Uni proposals cached for Table 1
                 (``results/layer_search/proposal_cache/proposals.jsonl``),
                 masked exactly as ``layer_search/search_synergy.py`` does;
* ``shift``   -- that same mask translated to a random offset on the 16x16
                 grid, wrapping around. Area, shape and contiguity are
                 preserved exactly; only the content inside the region
                 changes. This is the control that isolates content;
* ``scatter`` -- the same number of patches drawn uniformly at random. Matches
                 area alone, so comparing it against ``shift`` separates a
                 contiguity effect from a content effect.

If ``union`` does not beat ``shift``, the regions carry no more geolocation
evidence than an arbitrary patch of the same shape somewhere else, and the
displacement reported in the paper is a property of perturbing attention
rather than of the objects found.

Expected remote assets (under /content, or /kaggle/working on Kaggle):

* geoclip_source.zip, or an unpacked geoclip_source/
* intervention.py
* proposals.jsonl

Per-image results are checkpointed, so a second execution resumes after a
runtime interruption. Set MAX_IMAGES for a smoke test.
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

import kagglehub
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from transformers import AutoProcessor

DATASET_HANDLE = os.getenv("DATASET_HANDLE", "lbgan2000/imgps3k-yfcc4k-cleaned")
# Table 1's mask rule: boxes scoring >= 0.4, at most 100 of them.
SCORE_THRESHOLD = 0.4
PROPOSAL_TOP_K = 100

# Defaults reproduce the configuration of Table 1: layers {2,7,12} under the
# 2/k schedule, which is magnitude 2/3 at each of the three active layers.
LAYER_SPEC = os.getenv("INTERVENTION_LAYERS", "2,7,12")
ATTENTION_A = float(os.getenv("INTERVENTION_A", str(2.0 / 3.0)))
ATTENTION_B = float(os.getenv("INTERVENTION_B", str(-2.0 / 3.0)))

RANDOM_REPEATS = int(os.getenv("RANDOM_REPEATS", "5"))
SEED = int(os.getenv("SEED", "42"))
BASELINE_BATCH_SIZE = int(os.getenv("BASELINE_BATCH_SIZE", "64"))
MAX_IMAGES = int(os.getenv("MAX_IMAGES", "0"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "25"))
BOOTSTRAP_RESAMPLES = int(os.getenv("BOOTSTRAP_RESAMPLES", "10000"))
CLIP_KM = float(os.getenv("CLIP_KM", "2500"))

CONTENT = Path(os.getenv(
    "WORK_DIR", "/kaggle/working" if Path("/kaggle/working").is_dir() else "/content"
))
SOURCE_ZIP = CONTENT / "geoclip_source.zip"
SOURCE_DIR = CONTENT / "geoclip_source"
PROPOSALS_PATH = CONTENT / "proposals.jsonl"
OUTPUT_DIR = CONTENT / os.getenv("OUTPUT_NAME", "img2gps3k_random_mask_control")
PER_IMAGE_PATH = OUTPUT_DIR / "per_image.csv"
SUMMARY_PATH = OUTPUT_DIR / "summary.json"
THRESHOLDS_KM = (1, 25, 200, 750, 2500)
GRID = 16
PATCHES = GRID * GRID

ImageFile.LOAD_TRUNCATED_IMAGES = True
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Mask construction -- the only genuinely new logic here, so it self-checks.
# --------------------------------------------------------------------------

def shifted_mask(mask: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Translate a patch mask on the grid, wrapping at the edges.

    Wrapping rather than clipping keeps the active-patch count identical, which
    is the whole point of an area-matched control. A blob that straddles an edge
    comes back on the other side, so contiguity is preserved up to that seam.
    """
    grid = mask.reshape(GRID, GRID)
    return np.roll(np.roll(grid, dy, axis=0), dx, axis=1).reshape(PATCHES)


def scattered_mask(count: int, rng: np.random.Generator) -> np.ndarray:
    """A mask of exactly ``count`` patches drawn uniformly without replacement."""
    mask = np.zeros(PATCHES, dtype=bool)
    mask[rng.choice(PATCHES, size=count, replace=False)] = True
    return mask


def control_masks(union: np.ndarray, name: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Both control families for one image, seeded from its file name.

    Seeding per image rather than from a running counter keeps each image's
    controls identical whether the run completes in one go or resumes from a
    checkpoint partway through.
    """
    # Not hash(): str hashing is salted per process, which would change the
    # controls of every image processed after a resume.
    rng = np.random.default_rng([SEED, zlib.crc32(name.encode("utf-8"))])
    count = int(union.sum())
    shifts, scatters = [], []
    for _ in range(RANDOM_REPEATS):
        # A zero offset would hand back the original mask; exclude it.
        while True:
            dy, dx = int(rng.integers(0, GRID)), int(rng.integers(0, GRID))
            if (dy, dx) != (0, 0):
                break
        shifts.append(shifted_mask(union, dy, dx))
        scatters.append(scattered_mask(count, rng))
    return shifts, scatters


def table1_mask(record: dict) -> np.ndarray:
    """Copy of ``boxes_to_mask`` in layer_search/search_synergy.py.

    Kept identical rather than reusing intervention.build_in_region_mask, so the
    ``union`` arm is exactly the mask that produced Table 1.
    """
    width, height = record["width"], record["height"]
    selected = [p for p in record["proposals"] if p["score"] >= SCORE_THRESHOLD][:PROPOSAL_TOP_K]
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


def _self_check() -> None:
    union = np.zeros(PATCHES, dtype=bool)
    union.reshape(GRID, GRID)[2:6, 3:9] = True  # a 4x6 contiguous block
    assert union.sum() == 24

    for dy in range(GRID):
        for dx in range(GRID):
            moved = shifted_mask(union, dy, dx)
            assert moved.sum() == 24, "shift must preserve area"
    assert np.array_equal(shifted_mask(union, 0, 0), union)
    assert not np.array_equal(shifted_mask(union, 3, 0), union)
    # Wrapping, not clipping: pushing the block off the bottom brings it back.
    assert shifted_mask(union, GRID, GRID).sum() == 24
    assert np.array_equal(shifted_mask(union, GRID, GRID), union)

    rng = np.random.default_rng(0)
    for count in (1, 24, PATCHES):
        assert scattered_mask(count, rng).sum() == count

    a_shift, a_scatter = control_masks(union, "image-a.jpg")
    b_shift, b_scatter = control_masks(union, "image-a.jpg")
    assert all(np.array_equal(x, y) for x, y in zip(a_shift, b_shift)), "must be reproducible"
    assert all(np.array_equal(x, y) for x, y in zip(a_scatter, b_scatter))
    assert all(m.sum() == 24 for m in a_shift + a_scatter)
    c_shift, _ = control_masks(union, "image-b.jpg")
    assert not all(np.array_equal(x, y) for x, y in zip(a_shift, c_shift)), "must differ per image"


_self_check()


# --------------------------------------------------------------------------
# Remote setup
# --------------------------------------------------------------------------

def ensure_source() -> None:
    package_marker = SOURCE_DIR / "geoclip" / "__init__.py"
    if not package_marker.is_file():
        if not SOURCE_ZIP.is_file():
            raise FileNotFoundError(f"Missing uploaded source archive: {SOURCE_ZIP}")
        SOURCE_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(SOURCE_ZIP) as archive:
            for member in archive.infolist():
                relative = Path(*member.filename.replace("\\", "/").split("/"))
                target = SOURCE_DIR / relative
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    while chunk := source.read(1024 * 1024):
                        destination.write(chunk)
    for local_module in ("intervention.py",):
        source = CONTENT / local_module
        target = SOURCE_DIR / local_module
        if source.is_file() and not target.is_file():
            target.write_bytes(source.read_bytes())
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
    """One independent mask per item in an inference batch.

    The three arms of this experiment differ only in their mask, so batching
    them lets one forward pass evaluate the union together with all of its
    controls on identical weights and precision.
    """

    def key_bias_for_layer(self, layer_idx: int, num_positions: int):
        a, b = self.layer_ab.get(layer_idx, (0.0, 0.0))
        if (a == 0.0 and b == 0.0) or self.in_region_mask is None:
            return None
        mask = self.in_region_mask
        if mask.ndim == 1:
            bias = torch.full((num_positions,), b, dtype=torch.float32, device=mask.device)
            bias[0] = 0.0
            bias[1:].masked_fill_(mask, a)
            return bias
        if mask.ndim != 2 or mask.shape[1] != num_positions - 1:
            raise ValueError(f"Unexpected batched mask shape {tuple(mask.shape)}")
        bias = torch.full(
            (mask.shape[0], num_positions), b, dtype=torch.float32, device=mask.device
        )
        bias[:, 0] = 0.0
        bias[:, 1:].masked_fill_(mask, a)
        return bias[:, None, None, :]

    state.key_bias_for_layer = types.MethodType(key_bias_for_layer, state)


def paired_bootstrap(values: np.ndarray, resamples: int, seed: int) -> tuple[float, float]:
    """Percentile interval for a mean, resampling images rather than arms."""
    rng = np.random.default_rng(seed)
    if not len(values):
        return float("nan"), float("nan")
    draws = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[draws].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


ensure_source()
from geoclip import GeoCLIP  # noqa: E402
import intervention  # noqa: E402

if not torch.cuda.is_available():
    raise RuntimeError("This evaluation requires a Colab GPU runtime")
device = torch.device("cuda")

# The confirmatory pass of the paper disables TF32; keep that here so this
# control is comparable with Table 1 rather than with the screening runs.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.manual_seed(SEED)

active_layers_spec = LAYER_SPEC.strip().lower()
print("CONFIG", json.dumps({
    "dataset": DATASET_HANDLE,
    "layers": LAYER_SPEC,
    "a": ATTENTION_A,
    "b": ATTENTION_B,
    "random_repeats": RANDOM_REPEATS,
    "seed": SEED,
    "max_images": MAX_IMAGES,
}, sort_keys=True))
print("GPU", torch.cuda.get_device_name(0))

dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
csv_path = dataset_path / "test_set" / "im2gps3k_places365.csv"
image_dir = dataset_path / "test_set" / "im2gps3ktest" / "im2gps3ktest"
frame = pd.read_csv(csv_path)
frame["image_path"] = frame["name"].map(lambda name: str(image_dir / name))
missing = frame.loc[~frame["image_path"].map(os.path.isfile), "name"].tolist()
if missing:
    raise RuntimeError(f"Missing labeled images: {missing[:10]}")
proposals = {
    record["name"]: record
    for record in map(json.loads, PROPOSALS_PATH.read_text(encoding="utf-8").splitlines())
}
if set(proposals) != set(frame["name"]):
    raise RuntimeError("proposals.jsonl does not cover exactly the Img2GPS3k images")
if MAX_IMAGES > 0:
    frame = frame.iloc[:MAX_IMAGES].copy()
frame = frame.reset_index(drop=True)
targets = frame[["LAT", "LON"]].to_numpy(dtype=np.float64)
print(f"DATASET path={dataset_path} labeled_images={len(frame)}")

model = GeoCLIP().to(device).eval()
model.image_encoder.image_processor = AutoProcessor.from_pretrained(
    "openai/clip-vit-large-patch14", use_fast=False
)
state = intervention.patch_vision_tower(model.image_encoder.CLIP)
enable_batched_masks(state)
num_attention_layers = intervention.num_layers(model.image_encoder.CLIP)
if active_layers_spec == "all":
    active_layers = list(range(num_attention_layers))
else:
    active_layers = [int(v.strip()) for v in LAYER_SPEC.split(",") if v.strip()]
    invalid = [layer for layer in active_layers if not 0 <= layer < num_attention_layers]
    if invalid:
        raise ValueError(f"Invalid attention layers: {invalid}")
with torch.inference_mode():
    gallery_features = F.normalize(
        model.location_encoder(model.gps_gallery.to(device)), dim=1
    )
    logit_scale = model.logit_scale.exp()
gallery_cpu = model.gps_gallery.cpu().numpy()
print(f"GEOCLIP_READY layers={active_layers}")

def predict_pixels(pixel_values: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    with torch.inference_mode():
        features = F.normalize(model.image_encoder(pixel_values.to(device)), dim=1)
        logits = logit_scale * (features @ gallery_features.T)
        probabilities = logits.softmax(dim=1)
        confidence, indices = probabilities.max(dim=1)
    return indices.cpu().numpy(), confidence.cpu().numpy()


baseline_indices: list[int] = []
baseline_confidence: list[float] = []
state.layer_ab = {}
state.in_region_mask = None
for start in range(0, len(frame), BASELINE_BATCH_SIZE):
    batch = frame.iloc[start : start + BASELINE_BATCH_SIZE]
    images = []
    for image_path in batch["image_path"]:
        with Image.open(image_path) as source:
            images.append(source.convert("RGB"))
    pixels = model.image_encoder.image_processor(images=images, return_tensors="pt")[
        "pixel_values"
    ]
    indices, confidence = predict_pixels(pixels)
    baseline_indices.extend(indices.tolist())
    baseline_confidence.extend(confidence.tolist())
    done = min(start + BASELINE_BATCH_SIZE, len(frame))
    if done == len(frame) or done % (BASELINE_BATCH_SIZE * 5) == 0:
        print(f"BASELINE {done}/{len(frame)}")

baseline_gps = gallery_cpu[np.asarray(baseline_indices, dtype=np.int64)]
baseline_distances = haversine_km(baseline_gps, targets)
print("BASELINE_DONE")

records: dict[str, dict] = {}
if PER_IMAGE_PATH.is_file():
    previous = pd.read_csv(PER_IMAGE_PATH)
    records = {str(row["name"]): row.to_dict() for _, row in previous.iterrows()}
    print(f"RESUME existing={len(records)}")


def save_checkpoint() -> None:
    ordered = [records[name] for name in frame["name"] if name in records]
    pd.DataFrame(ordered).to_csv(PER_IMAGE_PATH, index=False)
    print(f"CHECKPOINT rows={len(ordered)} path={PER_IMAGE_PATH}")


run_start = time.perf_counter()
processed = 0
for row_index, row in frame.iterrows():
    name = str(row["name"])
    if name in records:
        continue
    base_gps = baseline_gps[row_index]
    base_distance = float(baseline_distances[row_index])
    target = targets[row_index]
    record = {
        "name": name,
        "split": table1_split(name),
        "target_lat": float(target[0]),
        "target_lon": float(target[1]),
        "baseline_lat": float(base_gps[0]),
        "baseline_lon": float(base_gps[1]),
        "baseline_confidence": float(baseline_confidence[row_index]),
        "baseline_distance_km": base_distance,
        "proposal_count": 0,
        "union_patch_count": 0,
        "union_coverage": 0.0,
        "union_distance_km": base_distance,
        "union_delta_km": 0.0,
        "union_displacement_km": 0.0,
        "shift_distance_km": base_distance,
        "shift_delta_km": 0.0,
        "shift_displacement_km": 0.0,
        "scatter_distance_km": base_distance,
        "scatter_delta_km": 0.0,
        "scatter_displacement_km": 0.0,
        "error": "",
    }
    try:
        with Image.open(row["image_path"]) as source:
            image = source.convert("RGB")
        union = table1_mask(proposals[name])
        record["proposal_count"] = sum(
            p["score"] >= SCORE_THRESHOLD for p in proposals[name]["proposals"]
        )
        if union.any():
            count = int(union.sum())
            shifts, scatters = control_masks(union, name)
            arms = [union] + shifts + scatters

            pixels = model.image_encoder.image_processor(
                images=[image], return_tensors="pt"
            )["pixel_values"]
            state.layer_ab = {layer: (ATTENTION_A, ATTENTION_B) for layer in active_layers}
            state.in_region_mask = torch.from_numpy(np.stack(arms)).to(device)
            indices, _ = predict_pixels(pixels.repeat(len(arms), 1, 1, 1))
            state.layer_ab = {}
            state.in_region_mask = None

            gps = gallery_cpu[indices.astype(np.int64)]
            distances = haversine_km(gps, target)
            displacements = haversine_km(gps, base_gps)

            end_shift = 1 + RANDOM_REPEATS
            record.update({
                "union_patch_count": count,
                "union_coverage": count / PATCHES,
                "union_distance_km": float(distances[0]),
                "union_delta_km": base_distance - float(distances[0]),
                "union_displacement_km": float(displacements[0]),
                # One value per image: the mean over repeats, so every image
                # contributes equally to the paired comparison below.
                "shift_distance_km": float(distances[1:end_shift].mean()),
                "shift_delta_km": base_distance - float(distances[1:end_shift].mean()),
                "shift_displacement_km": float(displacements[1:end_shift].mean()),
                "scatter_distance_km": float(distances[end_shift:].mean()),
                "scatter_delta_km": base_distance - float(distances[end_shift:].mean()),
                "scatter_displacement_km": float(displacements[end_shift:].mean()),
            })
    except Exception as exc:  # Keep going; per-image failures are reported.
        state.layer_ab = {}
        state.in_region_mask = None
        record["error"] = repr(exc)
        print(f"IMAGE_ERROR name={name} error={exc!r}")

    records[name] = record
    processed += 1
    total_done = sum(name in records for name in frame["name"])
    if processed % CHECKPOINT_EVERY == 0 or total_done == len(frame):
        save_checkpoint()
        print(f"PROGRESS {total_done}/{len(frame)} elapsed={time.perf_counter() - run_start:.1f}s")

save_checkpoint()
result = pd.read_csv(PER_IMAGE_PATH)
active = result[result["union_patch_count"] > 0].reset_index(drop=True)


def arm_summary(prefix: str, subset: pd.DataFrame) -> dict:
    distances = subset[f"{prefix}_distance_km"].to_numpy(dtype=np.float64)
    return {
        "mean_distance_km": float(distances.mean()),
        "median_distance_km": float(np.median(distances)),
        "mean_displacement_km": float(subset[f"{prefix}_displacement_km"].mean()),
        "mean_clipped_improvement_km": float(
            np.clip(subset[f"{prefix}_delta_km"].to_numpy(dtype=np.float64),
                    -CLIP_KM, CLIP_KM).mean()
        ),
        "accuracy": {
            f"acc_{t}_km": float(np.mean(distances <= t)) for t in THRESHOLDS_KM
        },
    }


def contrast(left: str, right: str, subset: pd.DataFrame, metric: str = "delta") -> dict:
    """Per-image ``left - right`` difference, bootstrapped over images.

    ``delta`` is improvement (accuracy); ``displacement`` is how far each arm
    moves the prediction from baseline (sensitivity). Unclipped on purpose:
    ``shift``/``scatter`` are already means over repeats while ``union`` is a
    single run, so clipping at CLIP_KM would truncate union's jumps far more
    often than the controls' averages and bias every contrast against union.
    """
    def column(arm: str) -> np.ndarray:
        return subset[f"{arm}_{metric}_km"].to_numpy(dtype=np.float64)

    paired = column(left) - column(right)
    low, high = paired_bootstrap(paired, BOOTSTRAP_RESAMPLES, SEED)
    return {
        "mean_difference_km": float(paired.mean()),
        "ci95": [low, high],
        "significant": bool(low > 0 or high < 0),
    }


baseline_distances_all = result["baseline_distance_km"].to_numpy(dtype=np.float64)
summary = {
    "dataset": DATASET_HANDLE,
    "evaluated_images": int(len(result)),
    "active_mask_images": int(len(active)),
    "failed_images": int((result["error"].fillna("").astype(str) != "").sum()),
    "gpu": torch.cuda.get_device_name(0),
    "config": {
        "layers": active_layers,
        "a": ATTENTION_A,
        "b": ATTENTION_B,
        "score_threshold": SCORE_THRESHOLD,
        "proposal_top_k": PROPOSAL_TOP_K,
        "random_repeats": RANDOM_REPEATS,
        "seed": SEED,
        "clip_km": CLIP_KM,
        "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
        "tf32": False,
    },
    "baseline": {
        "mean_distance_km": float(baseline_distances_all.mean()),
        "median_distance_km": float(np.median(baseline_distances_all)),
    },
    # The question the paper leaves open: do detected regions matter more than
    # an arbitrary region of the same shape elsewhere in the image?
    # shift_vs_scatter separates a contiguity effect from a content effect.
    # Nothing here is tuned, so all active images are valid; the holdout subset
    # is reported too because Table 1 is a holdout result.
    "subsets": {
        subset_name: {
            "images": int(len(subset)),
            "arms": {arm: arm_summary(arm, subset) for arm in ("union", "shift", "scatter")},
            "contrasts": {
                metric: {
                    f"{left}_vs_{right}": contrast(left, right, subset, metric)
                    for left, right in (("union", "shift"), ("union", "scatter"), ("shift", "scatter"))
                }
                for metric in ("delta", "displacement")
            },
        }
        for subset_name, subset in (
            ("active", active),
            ("holdout_active", active[active["split"] == "holdout"]),
        )
    },
    "runtime_seconds": time.perf_counter() - run_start,
}
SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
print("SUMMARY", json.dumps(summary, indent=2))
print(f"\nWrote {PER_IMAGE_PATH}\nWrote {SUMMARY_PATH}")
