# BorzikPieces — CLAUDE.md

Project context for Claude Code. Auto-loaded at session start.

---

## What This Project Is

An **MLOps pipeline** using [Domino Workflow](https://github.com/Tauffer-Consulting/domino) (Airflow-based DAG UI) that wraps a radiology MRI brain lesion segmentation project into reusable, containerized "Pieces". The pipeline orchestrates data loading → EDA → preprocessing → model training → inference entirely through a drag-and-drop web interface.

**Two pipelines planned:**
- **Radiology** (active): Pituitary/brain lesion segmentation from NIfTI MRI scans
- **Histopathology** (pending): No data provided yet

**Domino repo (GitHub):** https://github.com/Tauffer-Consulting/domino  
**Piece repo owner:** `borzikpieces` (see `config.toml`)

---

## Project Layout

```
BorzikPieces/
├── pieces/                         # All Domino pieces
│   ├── NiftiDataLoaderPiece/       # Discovers NIfTI image+mask pairs
│   ├── DataSplitPiece/             # Train/val/test split
│   ├── NiftiPreprocessingPiece/    # Normalize, resize, save as .npy
│   ├── PituitaryDatasetPiece/      # Merges preprocessed splits, creates dataset config
│   ├── ModelTrainingPiece/         # 3D MONAI UNet/SwinUNETR training
│   ├── ModelInferencePiece/        # Inference + confidence visualization
│   ├── NiftiEDAPiece/              # Comprehensive 8-phase EDA (SK text)
│   ├── NiftiVisualizationPiece/    # Standalone NIfTI grid visualizer
│   ├── GenerativeShapesPiece/      # Example piece (shapes generator)
│   └── HelloWorldPiece/            # Example piece (hello world)
├── dependencies/
│   ├── Dockerfile_base             # Light image: nibabel, matplotlib, scipy, pandas, tqdm
│   ├── Dockerfile_torch            # Heavy image: + torch==2.1.2, monai[all]==1.3.0
│   ├── requirements.txt            # Base requirements (matches Dockerfile_base)
│   └── requirements_torch.txt      # Torch requirements
├── tp-radiology-adonema/           # Original radiology project (reference, not a piece)
│   └── tp_radiology_adonema/       # Source code + checkpoints + configs
├── data/paired/{images,masks}/     # Sample NIfTI data (sub-001 to sub-050)
├── airflow/                        # Airflow dags/logs/plugins (auto-created by docker-compose)
├── docker-compose.yaml             # Full Domino stack (Airflow + REST API + Frontend)
├── config.toml                     # Piece repository metadata
├── .domino/compiled_metadata.json  # Auto-generated — DO NOT edit manually
└── current_status.md               # Task tracker
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

## Data

| Location | Description |
|----------|-------------|
| `./data/paired/images/` | Sample NIfTI images (sub-001..sub-050) |
| `./data/paired/masks/` | Corresponding segmentation masks (binary: 0=bg, 1=lesion) |
| `./tp-radiology-adonema/tp_radiology_adonema/` | Full reference project (trainers, configs, notebooks) |

The radiology reference project uses a **2D/2.5D slice-based SMP UNet++** approach (ResNet34 encoder, pretrained on ImageNet, albumentations). The current pieces use a **3D MONAI UNet/SwinUNETR** patch-based approach. Both are scientifically valid for this data.

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

Use `Dockerfile_base` for pieces without PyTorch. Use `Dockerfile_torch` for ModelTrainingPiece and ModelInferencePiece.

---

## Known Bugs Fixed (2026-04-28)

| Piece | Bug | Fix Applied |
|-------|-----|-------------|
| `NiftiEDAPiece` | Duplicate `_generate_html_gallery` method (dead code) | Removed stub |
| `ModelTrainingPiece` | `os.path.exists(None)` crash on missing mask | Added None guard in `_precompute_fg_locations` and `__getitem__` |
| `ModelInferencePiece` | Mask list misaligned with image list for mixed-mask subjects | Keep all mask paths (incl. None) aligned with images; added None guard |
| `NiftiPreprocessingPiece` | Three parallel instances defaulted to same output_dir | Updated default + warning in field description |
| `Dockerfile_torch` | Unpinned `torch`/`monai` versions | Pinned `torch==2.1.2`, `monai[all]==1.3.0`, added `scikit-learn==1.3.2` |

---

## Setup Guide

See the "How to Set Up" section in `.claude/setup_guide.md` for full step-by-step instructions.

**Quick start:**
```bash
# 0. ONE-TIME GPU setup (run once per machine, already done on this host)
bash setup_gpu.sh
# What it does: installs nvidia-container-toolkit, sets Docker default runtime to nvidia,
# restarts Docker, then brings the stack down/up.

# 1. Prepare shared data storage
mkdir -p domino_data/medical_data/images domino_data/medical_data/masks
cp data/paired/images/*.nii.gz domino_data/medical_data/images/
cp data/paired/masks/*.nii.gz  domino_data/medical_data/masks/

# 2. Start the full stack
echo -e "AIRFLOW_UID=$(id -u)" > .env   # only needed first time
docker compose up -d

# 3. Open Domino UI
# Frontend:  http://localhost:3000  (admin@email.com / admin)
# REST API:  http://localhost:8000/docs
# Airflow:   http://localhost:8080  (airflow / airflow)

# 4. Install the piece repository in the UI (first time only):
# Settings → Piece Repositories → paste GitHub URL → select version 0.3.8 → Add

# 5. Import the radiology workflow (first time, or after a full reset):
bash import_workflow.sh
# or manually via curl — see "Workflow Import" section below
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
- When piece configs change significantly
- After bumping the piece repo version and updating the workflow

**When NOT to re-import:**
- Normal restarts (`docker compose down` / `up`) preserve the database — the workflow persists

---

## Piece Update Cycle

When you fix a piece and want to deploy the update:

```bash
# 1. Edit pieces/<PieceName>/piece.py or models.py

# 2. Bump version in config.toml
#    version = "0.3.9"   (or whatever next version)

# 3. Commit and push to GitHub
git add pieces/ config.toml dependencies/
git commit -m "fix: <description>"
git push

# 4. GitHub Actions will build and push new Docker images automatically
#    (images tagged: ghcr.io/borzikpieces/borzikpieces:<version>-group0  /group1 etc.)

# 5. In Domino UI → Settings → Piece Repositories → borzikpieces → update version
#    Domino will pull the new images on the next workflow run
```

**Image groups:**
- `group0`: `Dockerfile_base` — NiftiDataLoaderPiece, DataSplitPiece, NiftiPreprocessingPiece, PituitaryDatasetPiece, NiftiEDAPiece, NiftiVisualizationPiece
- `group1`: `Dockerfile_torch` — ModelTrainingPiece, ModelInferencePiece

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
