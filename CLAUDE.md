# BorzikPieces — CLAUDE.md

Project context for Claude Code. Auto-loaded at session start.

---

## What This Project Is

An **MLOps pipeline** using [Domino Workflow](https://github.com/Tauffer-Consulting/domino) (Airflow-based DAG UI) that wraps a radiology MRI brain lesion segmentation project into reusable, containerized "Pieces". The pipeline orchestrates data loading → EDA → preprocessing → model training → inference entirely through a drag-and-drop web interface.

**Two pipelines:**
- **Radiology** (active): Pituitary/brain lesion segmentation from NIfTI MRI scans
- **Histopathology** (active): 2D semantic segmentation of H&E stained tissue images (beetle/pituitary dataset, 5 classes)

**Domino repo (GitHub):** https://github.com/Tauffer-Consulting/domino  
**Piece repo owner:** `patrikborzik2426` (GitHub user, see `config.toml`)  
**GHCR images:** `ghcr.io/patrikborzik2426/borzikpieces:<version>-group0` / `-group1`

---

## Project Layout

```
BorzikPieces/
├── pieces/                           # All Domino pieces
│   │
│   │  ── Radiology pipeline ──────────────────────────────────────────
│   ├── NiftiDataLoaderPiece/         # Discovers NIfTI image+mask pairs
│   ├── DataSplitPiece/               # Train/val/test split (SubjectInfo list)
│   ├── NiftiPreprocessingPiece/      # Normalize, resize, save as .npy
│   ├── PituitaryDatasetPiece/        # Merges preprocessed splits, creates dataset config
│   ├── ModelTrainingPiece/           # 3D MONAI UNet/SwinUNETR training
│   ├── ModelInferencePiece/          # Inference + confidence visualization
│   ├── NiftiEDAPiece/                # Comprehensive 8-phase EDA (SK text)
│   ├── NiftiVisualizationPiece/      # Standalone NIfTI grid visualizer
│   │
│   │  ── Histopathology pipeline ────────────────────────────────────
│   ├── HistoDataLoaderPiece/         # Scans image/mask dirs → SampleInfo list
│   ├── HistoEDAPiece/                # Class pixel dist, gallery, HTML report
│   ├── HistoPatchExtractorPiece/     # Sliding-window patch extraction with fg filter
│   ├── HistoDataSplitPiece/          # Train/val/test split (SampleInfo list)
│   ├── HistoTrainingPiece/           # 2D SMP UNet/UNet++/FPN/DeepLabV3+, albumentations
│   ├── HistoValidationPiece/         # Dice, IoU, pixel acc, confusion matrix, HTML report
│   ├── HistoInferencePiece/          # Predictions + comparison figures, optional Dice
│   │
│   │  ── Examples ───────────────────────────────────────────────────
│   ├── GenerativeShapesPiece/        # Example piece (shapes generator)
│   └── HelloWorldPiece/              # Example piece (hello world)
│
├── dependencies/
│   ├── Dockerfile_base               # Light: nibabel, matplotlib, scipy, pandas, tqdm
│   ├── Dockerfile_torch              # Heavy: + torch==2.1.2, monai==1.3.0,
│   │                                 #          albumentations==1.3.1, smp==0.3.3
│   ├── requirements.txt              # Base requirements
│   └── requirements_torch.txt        # Torch + MONAI + albumentations + SMP
├── Histo/                            # Original histopathology reference code (not a piece)
│   ├── config.py                     # Config class (class mapping, paths, hyperparams)
│   ├── beetle_dataset.py             # BeetleDataset (RGB mask → class index)
│   ├── data.py                       # setup_dataloaders (train/val split)
│   ├── trainer.py                    # Trainer class (train/validate loop)
│   ├── metrics.py                    # DiceAccumulator
│   ├── helpers.py                    # setup_device, prepare_batch, save_rgb_masks
│   └── main.py                       # Entry point
├── tp-radiology-adonema/             # Original radiology project (reference, not a piece)
│   └── tp_radiology_adonema/         # Source code + checkpoints + configs
├── data/paired/{images,masks}/       # Sample NIfTI data (sub-001 to sub-050)
├── airflow/                          # Airflow dags/logs/plugins (auto-created by docker-compose)
├── docker-compose.yaml               # Full Domino stack (Airflow + REST API + Frontend)
├── config.toml                       # Piece repository metadata
├── .domino/compiled_metadata.json    # Auto-generated — DO NOT edit manually
└── current_status.md                 # Task tracker
```

---

## Workflow Architecture (Radiology)

```
NiftiDataLoaderPiece          ← images_path, masks_path from shared storage
│
├─► NiftiEDAPiece             ← parallel branch, analysis only
│     └─► NiftiVisualizationPiece
│
└─► DataSplitPiece            ← splits subjects into train/val/test
      ├─► NiftiPreprocessingPiece [train]  → output_dir: .../preprocessed/train
      ├─► NiftiPreprocessingPiece [val]    → output_dir: .../preprocessed/val
      └─► NiftiPreprocessingPiece [test]   → output_dir: .../preprocessed/test
            └─► PituitaryDatasetPiece      ← receives all 3 preprocessed sets
                  └─► ModelTrainingPiece   ← 3D MONAI UNet, patch-based
                        └─► ModelInferencePiece
```

**Critical config rule:** Each of the 3 NiftiPreprocessingPiece instances MUST have a different `output_dir` path. Default is `.../preprocessed/train` — change the val and test instances to `.../preprocessed/val` and `.../preprocessed/test`.

---

## Workflow Architecture (Histopathology)

```
HistoDataLoaderPiece          ← images_path, masks_path
│  outputs: samples (List[SampleInfo]), images_path, masks_path
│
├─► HistoEDAPiece             ← parallel branch, analysis + HTML report only
│     inputs: images_path, masks_path, class_mapping_json, class_names
│
└─► HistoPatchExtractorPiece  ← OPTIONAL: tile large slides into patches
│     inputs: samples, patch_size, stride, min_foreground_ratio
│     outputs: samples (one SampleInfo per patch)
│     skip this piece and connect DataLoader → DataSplit directly
│     if images are already patch-sized (e.g. beetle dataset)
│
└─► HistoDataSplitPiece       ← train / val / test split
      inputs: samples, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15
      outputs: train_samples, val_samples, test_samples
      │
      ├─► HistoTrainingPiece  ← inputs: train_samples, val_samples
      │     SMP architecture, albumentations augmentation, AdamW + ReduceLROnPlateau
      │     outputs: model_path, best_model_path + arch pass-through fields
      │
      │     └─► HistoValidationPiece
      │           inputs: val_samples (from DataSplit)
      │                   model_path / arch fields (from Training)
      │           outputs: Dice, IoU, pixel_accuracy, confusion matrix, HTML report
      │
      └─► HistoInferencePiece ← inputs: test_samples (from DataSplit)
                                         model_path / arch fields (from Training)
            outputs: RGB prediction masks, comparison figures, optional Dice
```

### SampleInfo — shared data contract across the histopathology pipeline
Every histo piece that passes data uses `SampleInfo`:
```python
class SampleInfo(BaseModel):
    name: str           # filename with extension, e.g. "image_001.png"
    image_path: str     # absolute path to image file
    mask_path: Optional[str] = None  # absolute path to RGB mask
```
Each piece defines its own copy (Domino serialises via JSON so the field names just need to match).

### HistoTrainingPiece — pass-through outputs
After training, these fields are emitted so downstream pieces can auto-wire without re-entering values:

| Output field | Connects to |
|---|---|
| `model_path` / `best_model_path` | `HistoValidationPiece.model_path`, `HistoInferencePiece.model_path` |
| `class_mapping_json` | Validation + Inference |
| `model_architecture` | Validation + Inference |
| `encoder_name` | Validation + Inference |
| `num_classes` | Validation + Inference |
| `image_height` / `image_width` | Validation + Inference |

### HistoTrainingPiece — `dry_run` checkbox
Works identically to the radiology `ModelTrainingPiece.dry_run`:
- `dry_run=True` → `epochs=1`, `batch_size=4`, first 8 train samples only
- Use it to verify full pipeline wiring before a real training run.

### HistoPatchExtractorPiece — foreground filter
Large WSI images contain mostly background. Set `min_foreground_ratio=0.05` to discard patches where less than 5% of mask pixels are non-background. Background is detected by matching `background_rgb` (default `[0,0,0]`) against the mask. Set `min_foreground_ratio=0.0` to keep all patches.

### Class mapping (default — 5-class beetle/pituitary dataset)
```json
{"0": [0,0,0], "1": [128,128,128], "2": [0,255,0], "3": [255,0,0], "4": [0,0,255]}
```
| ID | Name | RGB |
|----|------|-----|
| 0 | unannotated | (0, 0, 0) black |
| 1 | other | (128,128,128) grey |
| 2 | non-invasive epithelium | (0,255,0) green |
| 3 | invasive epithelium | (255,0,0) red |
| 4 | necrosis | (0,0,255) blue |

---

## Data

### Radiology

| Location | Description |
|----------|-------------|
| `./data/paired/images/` | Sample NIfTI images (sub-001..sub-050) |
| `./data/paired/masks/` | Corresponding segmentation masks (binary: 0=bg, 1=lesion) |
| `./tp-radiology-adonema/tp_radiology_adonema/` | Full reference project (trainers, configs, notebooks) |

The radiology reference project uses a **2D/2.5D slice-based SMP UNet++** approach (ResNet34 encoder, pretrained on ImageNet, albumentations). The current pieces use a **3D MONAI UNet/SwinUNETR** patch-based approach. Both are scientifically valid for this data.

### Histopathology

| Location | Description |
|----------|-------------|
| `./Histo/` | Original reference code (not a piece — used as implementation blueprint) |
| Shared storage | Copy images to `/home/shared_storage/histo_data/images/` and masks to `/home/shared_storage/histo_data/masks/` |

The `Histo/` reference code uses CrossEntropyLoss + AdamW with a custom UNET and BeetleDataset (RGB-mask → class-index conversion). The histo pieces port this logic using **segmentation-models-pytorch** (SMP) for flexible architecture choice and **albumentations** for augmentation. The `filtered_dataset/` directory from the reference config maps to `histo_data/` in shared storage.

### Bundled test dataset — MoNuSeg (downloaded 2026-05-03)

`domino_data/histo_data/` already contains the **MoNuSeg** (Multi-organ Nucleus Segmentation) dataset, ready to use.

| Property | Value |
|----------|-------|
| Source | HuggingFace `RationAI/MoNuSeg` (CC BY-NC-SA 4.0) |
| Images | 51 H&E tiles, 1000×1000 px, TCGA archive (7 organ types) |
| Masks | 2-class RGB: background `[0,0,0]`, nucleus `[0,255,0]` |
| Setup script | `python3 setup_histo_data.py` (re-downloads if needed) |

**Piece configuration for MoNuSeg:**
```
class_mapping_json  = {"0": [0,0,0], "1": [0,255,0]}
class_names         = ["background", "nucleus"]
num_classes         = 2
```

**Pipeline wiring for MoNuSeg** (images are 1000×1000 → patch extraction required):
```
HistoDataLoaderPiece  → images_path=/home/shared_storage/histo_data/images
                        masks_path=/home/shared_storage/histo_data/masks
HistoPatchExtractorPiece → patch_size=256, stride=128, min_foreground_ratio=0.05
HistoDataSplitPiece
HistoTrainingPiece    → image_height=256, image_width=256, num_classes=2
```

**Expected patch count:** ~49 patches per image × 51 images ≈ 2 500 patches (after 5% foreground filter).

**To re-download the dataset:**
```bash
python3 setup_histo_data.py
```

**For the 5-class beetle/pituitary dataset** (your original data), update these fields:
```
class_mapping_json = {"0":[0,0,0],"1":[128,128,128],"2":[0,255,0],"3":[255,0,0],"4":[0,0,255]}
class_names        = ["unannotated","other","non-invasive","invasive","necrosis"]
image_height/width = 512 (or match your tile size)
Skip HistoPatchExtractorPiece if tiles are already patch-sized
```

---

## Piece Anatomy

Every piece has three required files:

```python
# piece.py
from domino.base_piece import BasePiece
from .models import InputModel, OutputModel

class MyPiece(BasePiece):
    def piece_function(self, input_data: InputModel) -> OutputModel:
        # self.logger      → logging
        # self.results_path → directory for tracked output files
        # self.display_result = {"file_type": "json"|"html", "base64_content": ...}
        return OutputModel(...)
```

```python
# models.py
from pydantic import BaseModel, Field
from typing import Optional, List

class InputModel(BaseModel):
    my_param: str = Field(description="...", default="value")

class OutputModel(BaseModel):
    result: str = Field(description="...")
```

```json
// metadata.json
{
  "name": "MyPiece",
  "description": "...",
  "dependency": {"dockerfile": "Dockerfile_base"},
  "tags": ["tag1"],
  "style": {"node_label": "My Piece", "icon_class_name": "fa-solid:cog"}
}
```

Use `Dockerfile_base` for pieces without PyTorch. Use `Dockerfile_torch` for ModelTrainingPiece, ModelInferencePiece, HistoTrainingPiece, HistoValidationPiece, and HistoInferencePiece.

---

## GHCR Authentication — Critical Setup (2026-04-29)

The classic GitHub PAT in `.env` (`GHCR_TOKEN`) belongs to `PatrikBorzik2426`. The `REGISTRY_NAME` in `config.toml` MUST match this GitHub username (lowercase). Previously it was `borzikpieces` (a non-existent account), causing every CI push to fail silently with "owner not found".

**One-time host setup** (must be redone if `~/.docker/config.json` is wiped or on a fresh machine):
```bash
echo "YOUR_GITHUB_PAT" | docker login ghcr.io -u patrikborzik2426 --password-stdin
```
This authenticates the host Docker daemon. The `domino-docker-proxy` service exposes the host's `/var/run/docker.sock`, so piece containers are pulled using the host's stored credentials.

**`.env` must have:**
```
DOMINO_DEFAULT_PIECES_REPOSITORY_TOKEN=YOUR_GITHUB_PAT
GHCR_USERNAME=patrikborzik2426
GHCR_TOKEN=YOUR_GITHUB_PAT
```
`DOMINO_DEFAULT_PIECES_REPOSITORY_TOKEN` is passed to the `domino_rest` service so it can access the GitHub API for piece metadata. If it is empty, Domino UI will show pieces but may fail to authenticate with GHCR.

**After editing `.env`**, restart the REST service to apply:
```bash
docker compose up -d --no-deps domino_rest
```

**To verify images are accessible after a CI run:**
```bash
docker pull ghcr.io/patrikborzik2426/borzikpieces:VERSION-group0
docker pull ghcr.io/patrikborzik2426/borzikpieces:VERSION-group1
```
If pull fails with "not found" even after `docker login`, the CI push itself failed — check the "Publish images" step in the Actions log for "denied" errors.

---

## Known Bugs Fixed

| Date | Piece / Area | Bug | Fix Applied |
|------|-------------|-----|-------------|
| 2026-04-28 | `NiftiEDAPiece` | Duplicate `_generate_html_gallery` method (dead code) | Removed stub |
| 2026-04-28 | `ModelTrainingPiece` | `os.path.exists(None)` crash on missing mask | Added None guard in `_precompute_fg_locations` and `__getitem__` |
| 2026-04-28 | `ModelInferencePiece` | Mask list misaligned with image list for mixed-mask subjects | Keep all mask paths (incl. None) aligned with images; added None guard |
| 2026-04-28 | `NiftiPreprocessingPiece` | Three parallel instances defaulted to same output_dir | Updated default + warning in field description |
| 2026-04-28 | `Dockerfile_torch` | Unpinned `torch`/`monai` versions | Pinned `torch==2.1.2`, `monai[all]==1.3.0`, added `scikit-learn==1.3.2` |
| 2026-04-29 | CI (`validate-and-organize.yml`) | `domino piece publish-images` Python SDK has a hard read timeout — the multi-GB group0 (torch) image consistently timed out mid-upload | Replaced with `docker push` CLI calls in a shell loop with 3 retries and 15 s backoff. The Docker CLI has no read timeout. |
| 2026-05-03 | `Dockerfile_torch` / `requirements_torch.txt` | Missing `albumentations` and `segmentation-models-pytorch` for histo pieces | Added `albumentations==1.3.1` and `segmentation-models-pytorch==0.3.3` to both files |
| 2026-05-05 | CI (`validate-and-organize.yml`) | `docker push` failing silently on every run since histo pieces added — `${{ github.repository_owner }}` expands to `PatrikBorzik2426` (mixed case) but Docker requires all-lowercase image refs. All 3 retry attempts errored with `invalid reference format: repository name must be lowercase`. CI still reported "success" because push failures weren't propagated as job-stopping exit codes. Images 0.3.15–0.3.17 were built but never pushed to GHCR. | Added `OWNER=$(echo "${{ github.repository_owner }}" \| tr '[:upper:]' '[:lower:]')` before the push loop and use `${OWNER}` in the image tag. |
| 2026-05-05 | Domino workflow run | Workflow fails immediately with no visible logs in UI — piece container shows `404 Not Found` pulling the Docker image from GHCR. Root cause is always a CI push failure: the version Domino registered exists as a GitHub release but the image was never actually pushed. Check GHCR tags via `curl -s "https://ghcr.io/token?scope=repository:patrikborzik2426/borzikpieces:pull&service=ghcr.io" \| python3 -c "import sys,json; print(json.load(sys.stdin).get('token',''))" > /tmp/t.txt && curl -s "https://ghcr.io/v2/patrikborzik2426/borzikpieces/tags/list" -H "Authorization: Bearer $(cat /tmp/t.txt)"` and compare against the registered version. |
| 2026-05-05 | `HistoInferencePiece` / `HistoValidationPiece` | `FileNotFoundError: Model not found: /home/shared_storage/histo_models/best_model.pth` — default `model_path` in both pieces was missing the `checkpoints/` subdirectory. `HistoTrainingPiece` saves to `output_dir/checkpoints/best_model.pth` but defaults in Inference and Validation pointed to `output_dir/best_model.pth`. Fails whenever the Training → Inference/Validation `best_model_path` wire is not connected and the user relies on the default. | Fixed default to `/home/shared_storage/histo_models/checkpoints/best_model.pth` in both `HistoInferencePiece/models.py` and `HistoValidationPiece/models.py`. Always wire `best_model_path` from HistoTrainingPiece to avoid relying on the default. |

---

## Notable Features Added

### ModelTrainingPiece — `dry_run` checkbox
`InputModel` has a `dry_run: bool` field (default `False`). When checked in the Domino UI it overrides:
- `epochs = 1`
- `batch_size = 1`
- `samples_per_volume = 1`

This lets the full pipeline be validated end-to-end in minutes without waiting for real training. Use it whenever you want to confirm the piece wiring is correct before a real run.

### Histopathology pipeline — 7-piece implementation (2026-05-03)

Full 2D semantic segmentation pipeline built from the `Histo/` reference code:

| Piece | Docker image | Key role |
|-------|-------------|----------|
| `HistoDataLoaderPiece` | base | Scan dirs → `List[SampleInfo]` |
| `HistoEDAPiece` | base | Class pixel distribution, gallery, HTML report |
| `HistoPatchExtractorPiece` | base | Sliding-window tiling with foreground filter |
| `HistoDataSplitPiece` | base | Train/val/test split of `SampleInfo` list |
| `HistoTrainingPiece` | torch | SMP model training, dry_run support |
| `HistoValidationPiece` | torch | Dice + IoU + pixel acc + confusion matrix |
| `HistoInferencePiece` | torch | Predictions + comparison figures |

**SampleInfo** (`name`, `image_path`, `mask_path`) is the data contract passed between every histo piece — same pattern as radiology's `SubjectInfo`.

**HistoTrainingPiece** emits arch pass-through fields (`model_architecture`, `encoder_name`, `num_classes`, `image_height`, `image_width`, `class_mapping_json`) so HistoValidationPiece and HistoInferencePiece can auto-wire without re-entering values.

**HistoPatchExtractorPiece** is optional — skip it and wire DataLoader → DataSplit directly when images are already patch-sized (e.g. the beetle dataset where each file is already a tile).

### MoNuSeg bundled dataset + setup_histo_data.py (2026-05-03)

`setup_histo_data.py` downloads the **MoNuSeg** dataset from HuggingFace (`RationAI/MoNuSeg`) and converts instance-level nucleus masks to 2-class RGB semantic masks expected by the pipeline. Running it once populates `domino_data/histo_data/` so the stack can be used immediately without sourcing private data.

All 4 pieces that accept `class_mapping_json` (`HistoEDAPiece`, `HistoTrainingPiece`, `HistoValidationPiece`, `HistoInferencePiece`) now default to the MoNuSeg 2-class mapping. `HistoPatchExtractorPiece` now defaults to `patch_size=256`, `stride=128` (appropriate for 1000×1000 MoNuSeg tiles). `HistoTrainingPiece` `image_height`/`image_width` default updated to `256` to match.

### startup.sh — automatic container log dump
Every time `bash startup.sh` runs it dumps logs into `logs/containers/` (gitignored):

| Path | Contents |
|------|----------|
| `logs/containers/<container>.txt` | Last 2000 lines from each of the 5 main containers |
| `logs/containers/airflow_tasks/` | Copies of the 5 most recent Airflow task `attempt=1.log` files (flat filenames) |

Containers captured: `airflow-domino-worker`, `airflow-domino-scheduler`, `airflow-webserver`, `domino-rest`, `domino-frontend`. Containers that are not running are skipped silently.

---

## Setup Guide

See the "How to Set Up" section in `.claude/setup_guide.md` for full step-by-step instructions.

**Quick start:**
```bash
# 0. ONE-TIME GPU setup (run once per machine, already done on this host)
bash setup_gpu.sh
# What it does: installs nvidia-container-toolkit, sets Docker default runtime to nvidia,
# restarts Docker, then brings the stack down/up.

# 1. ONE-TIME GHCR login on the host (persists in ~/.docker/config.json)
echo "YOUR_GITHUB_PAT" | docker login ghcr.io -u patrikborzik2426 --password-stdin

# 2. Prepare shared data storage
mkdir -p domino_data/medical_data/images domino_data/medical_data/masks
cp data/paired/images/*.nii.gz domino_data/medical_data/images/
cp data/paired/masks/*.nii.gz  domino_data/medical_data/masks/

# 3. Create .env (only needed first time — AIRFLOW_UID must be set)
cat > .env <<'EOF'
AIRFLOW_UID=1000
DOMINO_COMPOSE_DEV=
DOMINO_DEFAULT_PIECES_REPOSITORY_TOKEN=YOUR_GITHUB_PAT
DOMINO_CREATE_DEFAULT_USER=true
GHCR_USERNAME=patrikborzik2426
GHCR_TOKEN=YOUR_GITHUB_PAT
EOF

# 4. Start the full stack
docker compose up -d

# 5. Open Domino UI
# Frontend:  http://localhost:3000  (admin@email.com / admin)
# REST API:  http://localhost:8000/docs
# Airflow:   http://localhost:8080  (airflow / airflow)

# 6. Add piece repository via API (first time only — UI also works):
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@email.com","password":"admin"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")
curl -s -X POST "http://localhost:8000/pieces-repositories" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "workspace_id": 1,
    "source": "github",
    "path": "PatrikBorzik2426/BorzikPieces",
    "url": "https://github.com/PatrikBorzik2426/BorzikPieces",
    "version": "LATEST_VERSION_HERE"
  }'

# 7. Import the radiology workflow:
bash import_workflow.sh
# or manually — see "Workflow Import" section below
```

**GPU stack:**
- Host GPU: RTX 4060, 8 GB VRAM, CUDA 13.0 driver
- PyTorch in containers: `torch==2.1.2+cu121` (bundled CUDA 12.1 libs)
- Piece containers get GPU via Docker daemon default NVIDIA runtime
- `use_gpu: true` is already the default in ModelTrainingPiece and ModelInferencePiece

---

## Workflow Import

`radiology_workflow.json` is the canonical workflow definition. Import it into a fresh Domino instance via the REST API:

```bash
# 1. Get a token
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@email.com","password":"admin"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# 2. POST the workflow
curl -s -X POST http://localhost:8000/workspaces/1/workflows \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @radiology_workflow.json | python3 -c "import sys,json; r=json.load(sys.stdin); print('Created workflow ID:', r['id'], 'name:', r['name'])"
```

**When to re-import:**
- After `docker compose down -v` (full reset wipes the database)
- After bumping the piece repo version (delete old repo + workflows, re-add repo, update `radiology_workflow.json` image tags, then re-import)
- After changing `REGISTRY_NAME` in `config.toml`

**When NOT to re-import:**
- Normal restarts (`docker compose down` / `up`) preserve the Domino postgres database — the workflow persists

**Import fails with "Some pieces were not found"?**
The piece repo must be registered in Domino (with the correct version) *before* importing the workflow. The `source_image` fields in the JSON must also match what Domino has indexed. Use the python snippet in "Workflow JSON — Keeping It Current" to update them.

---

## Piece Update Cycle

When you fix a piece and want to deploy the update:

```bash
# 1. Edit pieces/<PieceName>/piece.py or models.py

# 2. Do NOT manually bump config.toml version — CI auto-bumps the patch version on every push

# 3. Commit and push to GitHub
git add pieces/ dependencies/
git commit -m "fix: <description>"
git push
# NOTE: the CI will push its own auto-bump commit back; always pull --rebase before your next push

# 4. GitHub Actions builds and pushes new Docker images to:
#    ghcr.io/patrikborzik2426/borzikpieces:VERSION-group0  (torch — ModelTrainingPiece, ModelInferencePiece)
#    ghcr.io/patrikborzik2426/borzikpieces:VERSION-group1  (base — all other pieces)
#    Push uses docker push CLI (3 retries, 15 s backoff) — NOT the domino SDK which times out on group0.
#    Verify with: docker pull ghcr.io/patrikborzik2426/borzikpieces:VERSION-group0

# 5. Re-register the piece repository with the new version (API — Domino UI also works):
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@email.com","password":"admin"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# First delete any workflows using the repo, then delete the repo itself:
curl -s -X DELETE "http://localhost:8000/workspaces/1/workflows/WORKFLOW_ID" -H "Authorization: Bearer $TOKEN"
curl -s -X DELETE "http://localhost:8000/pieces-repositories/REPO_ID?workspace_id=1" -H "Authorization: Bearer $TOKEN"

# Re-add with new version:
curl -s -X POST "http://localhost:8000/pieces-repositories" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"workspace_id":1,"source":"github","path":"PatrikBorzik2426/BorzikPieces","url":"https://github.com/PatrikBorzik2426/BorzikPieces","version":"NEW_VERSION"}'

# 6. Update radiology_workflow.json source_image fields to match new version, then re-import:
#    (see "Workflow JSON — Keeping It Current" below)
bash import_workflow.sh
```

**Image groups (as assigned by Domino's organize step):**
- `group0` = `Dockerfile_torch` → **ModelTrainingPiece, ModelInferencePiece, HistoTrainingPiece, HistoValidationPiece, HistoInferencePiece**
- `group1` = `Dockerfile_base` → NiftiDataLoaderPiece, DataSplitPiece, NiftiPreprocessingPiece, PituitaryDatasetPiece, NiftiEDAPiece, NiftiVisualizationPiece, HistoDataLoaderPiece, HistoEDAPiece, HistoPatchExtractorPiece, HistoDataSplitPiece, HelloWorldPiece, GenerativeShapesPiece

> **Warning:** The group numbering is the opposite of what you might expect — the *heavier* torch image is group0. The `.domino/compiled_metadata.json` (auto-generated by CI) is the source of truth.

---

## Workflow JSON — Keeping It Current

`radiology_workflow.json` contains hardcoded `source_image` fields. After a version bump or registry change, update them before re-importing:

```python
import json

TORCH_PIECES = {
    'ModelTrainingPiece', 'ModelInferencePiece',
    'HistoTrainingPiece', 'HistoValidationPiece', 'HistoInferencePiece',
}
GROUP0 = 'ghcr.io/patrikborzik2426/borzikpieces:VERSION-group0'
GROUP1 = 'ghcr.io/patrikborzik2426/borzikpieces:VERSION-group1'

for wf_file in ['radiology_workflow.json', 'histo_workflow.json']:
    try:
        d = json.load(open(wf_file))
        for task in d['tasks'].values():
            piece_name = task['piece']['name']
            task['piece']['source_image'] = GROUP0 if piece_name in TORCH_PIECES else GROUP1
        json.dump(d, open(wf_file, 'w'), indent=2)
        print(f'Updated {wf_file}')
    except FileNotFoundError:
        print(f'Skipped {wf_file} (not found)')
```

The Domino REST API does **not** have a PATCH endpoint for piece repositories — the only way to update the version is delete + re-create. You must also delete all workflows that reference the repository before deleting it:

```bash
# Check which repo ID to delete
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8000/pieces-repositories?workspace_id=1" \
  | python3 -c "import sys,json; [print(r['id'], r['name'], r['version']) for r in json.load(sys.stdin)['data']]"

# Check which workflow IDs are active
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8000/workspaces/1/workflows" \
  | python3 -c "import sys,json; [print(w['id'], w['name']) for w in json.load(sys.stdin).get('data',[])]"
```

---

## Local Testing (run_local.py)

Test pieces without Docker, Airflow, or a CI push. The runner imports pieces directly, topologically sorts the DAG, wires outputs to inputs via `{"from": "node.field"}` references, and injects file-system paths so pieces that write files work without `/home/shared_storage` being mounted.

**Full guide:** `LOCAL_TESTING.md`

```bash
# Histo — base pieces only (no torch, ~10 s) — VERIFIED PASSING 2026-05-05
python3 run_local.py local_workflows/histo_monuseg_base_only.json

# Radiology — non-GPU chain (~30 s) — VERIFIED PASSING 2026-05-05
python3 run_local.py local_workflows/radiology_smoke.json

# Full histo pipeline incl. training (torch required)
python3 run_local.py local_workflows/histo_monuseg.json

# Verbose: prints every resolved input and every serialised output per piece
python3 run_local.py local_workflows/histo_monuseg_base_only.json --verbose

# Custom output directory
python3 run_local.py local_workflows/histo_monuseg_base_only.json --results-dir /tmp/mytest
```

**Workflow JSON format (minimal example):**
```json
{
  "name": "my_workflow",
  "results_dir": "/tmp/my_results",
  "pieces": {
    "loader": {
      "piece": "HistoDataLoaderPiece",
      "inputs": { "images_path": "/path/to/images", "masks_path": "/path/to/masks" }
    },
    "split": {
      "piece": "HistoDataSplitPiece",
      "inputs": {
        "samples":     {"from": "loader.samples"},
        "train_ratio": 0.7
      }
    }
  }
}
```

**Key rules:**
- `{"from": "node_id.field"}` — `field` must match exactly the OutputModel field name of the upstream piece. Run with `--verbose` to discover field names.
- Pieces that write to a path defaulting to `/home/shared_storage/...` need that path overridden in the JSON to a writable local path (e.g. `HistoPatchExtractorPiece.output_dir`).
- Failures are non-fatal — other non-dependent pieces still run; summary at the end lists all failures.
- torch pieces (`HistoTrainingPiece`, `HistoValidationPiece`, `HistoInferencePiece`, `ModelTrainingPiece`, `ModelInferencePiece`) will fail locally with `No module named 'torch'` — they only run inside the Docker image.

**Available workflow files:**

| File | Pieces | Torch | Runtime |
|------|--------|-------|---------|
| `local_workflows/histo_monuseg_base_only.json` | DataLoader → EDA → Patches → Split | No | ~10 s |
| `local_workflows/histo_monuseg.json` | Full histo pipeline | Yes | minutes |
| `local_workflows/radiology_smoke.json` | DataLoader → EDA → Split → 3×Preprocess → Dataset | No | ~30 s |

---

## Useful Commands

```bash
# Stack lifecycle
docker compose up -d                  # start
docker compose down                   # stop (data preserved)
docker compose down -v                # full reset (wipes DB volumes)
docker compose logs -f domino_rest    # watch API logs
docker compose logs -f airflow-domino-worker  # watch worker logs

# Check GPU is visible inside the worker
docker exec airflow-domino-worker nvidia-smi

# Check shared storage is mounted correctly
docker exec airflow-domino-worker ls /home/shared_storage/medical_data/

# Tail a running piece's logs (find container name first)
docker ps | grep piece
docker logs -f <piece-container-name>

# API shortcuts
TOKEN=$(curl -s -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"admin@email.com","password":"admin"}' \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['access_token'])")

# List workflows
curl -s -H "Authorization: Bearer $TOKEN" http://localhost:8000/workspaces/1/workflows \
  | python3 -c "import sys,json; [print(w['id'], w['name'], w['status']) for w in json.load(sys.stdin)['data']]"

# List piece repositories
curl -s -H "Authorization: Bearer $TOKEN" "http://localhost:8000/pieces-repositories?workspace_id=1" \
  | python3 -c "import sys,json; [print(r['id'], r['name'], r['version']) for r in json.load(sys.stdin)['data']]"
```
