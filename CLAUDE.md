# BorzikPieces — CLAUDE.md

Project context for Claude Code. Auto-loaded at session start.

---

## What This Project Is

An **MLOps pipeline** using [Domino Workflow](https://github.com/Tauffer-Consulting/domino) (Airflow-based DAG UI) that wraps a radiology MRI brain lesion segmentation project into reusable, containerized "Pieces". The pipeline orchestrates data loading → EDA → preprocessing → model training → inference entirely through a drag-and-drop web interface.

**Two pipelines planned:**
- **Radiology** (active): Pituitary/brain lesion segmentation from NIfTI MRI scans
- **Histopathology** (pending): No data provided yet

**Domino repo (GitHub):** https://github.com/Tauffer-Consulting/domino  
**Piece repo owner:** `patrikborzik2426` (GitHub user, see `config.toml`)  
**GHCR images:** `ghcr.io/patrikborzik2426/borzikpieces:<version>-group0` / `-group1`

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

---

## Notable Features Added

### ModelTrainingPiece — `dry_run` checkbox
`InputModel` has a `dry_run: bool` field (default `False`). When checked in the Domino UI it overrides:
- `epochs = 1`
- `batch_size = 1`
- `samples_per_volume = 1`

This lets the full pipeline be validated end-to-end in minutes without waiting for real training. Use it whenever you want to confirm the piece wiring is correct before a real run.

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
- `group0` = `Dockerfile_torch` → **ModelTrainingPiece, ModelInferencePiece**
- `group1` = `Dockerfile_base` → NiftiDataLoaderPiece, DataSplitPiece, NiftiPreprocessingPiece, PituitaryDatasetPiece, NiftiEDAPiece, NiftiVisualizationPiece, HelloWorldPiece, GenerativeShapesPiece

> **Warning:** The group numbering is the opposite of what you might expect — the *heavier* torch image is group0. The `.domino/compiled_metadata.json` (auto-generated by CI) is the source of truth.

---

## Workflow JSON — Keeping It Current

`radiology_workflow.json` contains hardcoded `source_image` fields. After a version bump or registry change, update them before re-importing:

```python
import json

IMAGE_MAP = {
    'ModelInferencePiece': 'ghcr.io/patrikborzik2426/borzikpieces:VERSION-group0',
    'ModelTrainingPiece':  'ghcr.io/patrikborzik2426/borzikpieces:VERSION-group0',
}
DEFAULT_IMAGE = 'ghcr.io/patrikborzik2426/borzikpieces:VERSION-group1'

d = json.load(open('radiology_workflow.json'))
for task in d['tasks'].values():
    piece_name = task['piece']['name']
    task['piece']['source_image'] = IMAGE_MAP.get(piece_name, DEFAULT_IMAGE)
json.dump(d, open('radiology_workflow.json', 'w'), indent=2)
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
