# GeoProbe

Interactive attention intervention for explainable image geolocation.

Reference implementation for the MMM 2027 demonstration paper *GeoProbe:
Interactive Attention Intervention for Explainable Image Geolocation*.

GeoProbe adds a region-conditioned, key-indexed bias to the pre-softmax
attention logits of GeoCLIP's CLIP ViT-L/14 vision tower, independently per
layer, and reports how far the predicted GPS coordinate moves. No training, no
gradients, no architectural change: the baseline and the intervened prediction
are two forward passes of identical weights.

Setting every bias to zero is an exact no-op, so the displayed baseline is the
unmodified model rather than an approximation.

## Repository layout

| Path | Contents |
|---|---|
| `backend/` | FastAPI service: loads GeoCLIP once, patches the 24 vision layers, serves predictions and WeDetect-Uni proposals |
| `frontend/` | React (Vite) UI: upload, region selection, per-layer `a`/`b` controls, map comparison |
| `experiments/` | Dataset-scale evaluation on Img2GPS3k that produces the paper's numbers |
| `results/` | Committed run summaries and per-image CSVs |
| `demo-images/` | Pre-screened photographs for the live demonstration |
| `paper/` | LaTeX source of the demonstration paper |
| `docs/HOW_IT_WORKS.md` | Detailed walkthrough of the intervention and the request path |
| `third_party/geo-clip` | Submodule: GeoCLIP fork made compatible with Transformers v4/v5 |

## Quick start

```bash
git clone --recursive https://github.com/Le-Anh-Duy/geoprobe.git
cd geoprobe
```

Already cloned without `--recursive`:

```bash
git submodule update --init --recursive
```

### Backend

```bash
cd backend
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --port 8000
```

The first run downloads CLIP ViT-L/14 (~1.6 GB) into `~/.cache/huggingface`.
The GeoCLIP location encoder, its MLP head and the 100k-point GPS gallery ship
inside the submodule. A GPU is not required: one CPU request returns both
predictions in about eight seconds.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. The backend must be reachable at
`http://127.0.0.1:8000`.

### Prompt-free region proposals (optional)

```bash
python backend/scripts/setup_wedetect_uni.py
```

Downloads the official WeDetect-Base-Uni checkpoint and exports its visual-only
proposal graph to ONNX at `~/.cache/wedetect/wedetect_anything_base.onnx`.
Point `WEDETECT_UNI_ONNX` at another export to override. The UI's
**Generate proposals** button then calls `POST /proposals`; the defaults are an
objectness threshold of `0.4` and an NMS IoU of `0.7`.

WeDetect is GPL-3.0. Its source and checkpoint are fetched at setup time and
deliberately kept outside this repository.

## Reproducing the paper

See [`experiments/README.md`](experiments/README.md).

## Demonstration

`demo-images/` holds photographs screened with
`backend/scripts/screen_demo_images.py`, which reports how far each WeDetect
proposal moves the top-1 prediction. Roughly a quarter of images with proposals
do not move the prediction at all, which is fine in aggregate and poor on
stage, so candidates are chosen in advance.

Basemap tiles and reverse-geocoded place names come from public map services
and are cartographic context only; they are not part of model inference, and
inference itself runs offline.

## Citation

```bibtex
@inproceedings{le2027geoprobe,
  title     = {GeoProbe: Interactive Attention Intervention for Explainable
               Image Geolocation},
  author    = {Le, Anh-Duy and Tran, Gia-Huy and Luu, Duc-Tuan and
               Tran, Anh-Duy and Dang-Nguyen, Duc-Tien},
  booktitle = {Proceedings of the 33rd International Conference on Multimedia
               Modeling (MMM)},
  year      = {2027}
}
```

## Licences

This repository is MIT (see `LICENSE`). GeoCLIP is MIT. WeDetect is GPL-3.0 and
is not distributed here.
