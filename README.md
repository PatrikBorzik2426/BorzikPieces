# BorzikPieces

Domino piece repository for two medical-imaging segmentation pipelines:

- **Radiology** — pituitary gland / brain lesion segmentation from 3D NIfTI MRI data
- **Histopathology** — tissue / cell instance segmentation from 2D whole-slide image patches

Pieces are deployed by [Domino Workflow](https://github.com/Tauffer-Consulting/domino) and orchestrated by Apache Airflow.

---

## Repository structure

```
pieces/                  individual Domino pieces (one folder per piece)
dependencies/
  Dockerfile_base        light image — no ML framework
  Dockerfile_torch       heavy image — PyTorch + MONAI / SMP
data/
  paired/images/         sample NIfTI images  (sub-001 … sub-050)
  paired/masks/          sample NIfTI masks
domino_data/             shared storage volume root (git-ignored)
local_workflows/         workflow JSONs for run_local.py
airflow/                 Airflow DAG / log directory (managed by Domino)
config.toml              release configuration — bump VERSION to publish
docker-compose.yaml      full Domino stack (Airflow + Domino REST + Frontend)
startup.sh               boot helper: copies sample data, starts stack, checks GPU
setup_gpu.sh             one-time NVIDIA Container Toolkit installer
run_local.py             local workflow runner — no Docker/Airflow required
```

---

## Pieces

### Radiology (NIfTI / 3D)

| Piece | Description | Image |
|---|---|---|
| `NiftiDataLoaderPiece` | Discovers and pairs NIfTI image/mask files | base |
| `NiftiEDAPiece` | EDA with statistics and HTML gallery report | base |
| `NiftiVisualizationPiece` | Axial/sagittal/coronal slice visualisation with mask overlay | base |
| `NiftiPreprocessingPiece` | Z-score / min-max / percentile normalisation, optional resize, saves as NumPy | base |
| `DataSplitPiece` | Train / val / test split with configurable ratios | base |
| `PituitaryDatasetPiece` | Assembles PyTorch-compatible dataset config from preprocessed splits | base |
| `ModelTrainingPiece` | 3D MONAI patch-based UNet / SwinUNETR training | torch |
| `ModelInferencePiece` | Inference with confidence visualisation and Dice/IoU metrics | torch |

### Histopathology (2D patches)

| Piece | Description | Image |
|---|---|---|
| `HistoDataLoaderPiece` | Discovers and validates paired image/mask files | base |
| `HistoPatchExtractorPiece` | Sliding-window patch tiling with foreground-ratio filter | base |
| `HistoDataSplitPiece` | Train / val / test split for patch lists | base |
| `HistoEDAPiece` | Per-class pixel distributions and sample gallery HTML report | base |
| `HistoTrainingPiece` | 2D segmentation (UNet / UNet++ / FPN / DeepLabV3+) via SMP, AdamW, Dice+CE loss | torch |
| `HistoValidationPiece` | Per-class Dice, IoU, pixel accuracy, confusion matrix, HTML report | torch |
| `HistoInferencePiece` | Inference with RGB prediction masks and optional Dice scores | torch |

---

## Docker images

Two images are published to **GitHub Container Registry (GHCR)** on every release:

| Image | Tag | Contents |
|---|---|---|
| `ghcr.io/patrikborzik2426/borzikpieces` | `<version>-base` | nibabel, matplotlib, scipy, scikit-image |
| `ghcr.io/patrikborzik2426/borzikpieces` | `<version>-torch` | torch 2.1.2+cu121, monai[all] 1.3.0, SMP, albumentations, scikit-learn |

Images are rebuilt automatically by GitHub Actions whenever `config.toml` is updated. Current version: **0.3.21**.

---

## Prerequisites

- Docker with Compose V2
- Linux host (required for `AIRFLOW_UID`)
- GPU (recommended): NVIDIA GPU with CUDA 12.x driver and `nvidia-container-toolkit`

### One-time GPU setup (new machine)

```bash
bash setup_gpu.sh
```

This installs `nvidia-container-toolkit`, sets NVIDIA as the Docker default runtime, and restarts Docker.

---

## Quick start

### 1. Start the full stack

```bash
bash startup.sh
```

The script:
1. Creates `.env` with `AIRFLOW_UID`
2. Copies sample NIfTI data into `domino_data/medical_data/`
3. Authenticates Docker to GHCR (if `GHCR_TOKEN` / `GHCR_USERNAME` are set in `.env`)
4. Runs `docker compose up -d`
5. Waits for the REST API and Frontend to respond
6. Checks GPU availability inside the worker container

| Service | URL | Credentials |
|---|---|---|
| Domino Frontend | http://localhost:3000 | admin@email.com / admin |
| Domino REST API | http://localhost:8000/docs | — |
| Airflow | http://localhost:8080 | airflow / airflow |

### 2. Register the piece repository in Domino

1. Open **Settings → Piece Repositories**
2. Paste `https://github.com/PatrikBorzik2426/BorzikPieces`
3. Select version tag (e.g. `0.3.21`) → **Add**

Domino pulls the Docker images and registers all pieces (a few minutes). To add a second repository (multi-repo setup), repeat these steps for the other repo's URL — pieces from all repos appear together in the workflow builder.

### 3. Import and run a workflow

```bash
bash import_workflow.sh          # imports radiology_domino_workflow.json via REST API
```

Or manually: **Workflows → Import** → select a workflow JSON.

Input data must be present under `domino_data/medical_data/` on the host (mounted to `/home/shared_storage/` inside workers) before running.

---

## Local testing (no Docker / Airflow)

`run_local.py` runs pieces directly in Python using the same workflow JSON format exported from Domino:

```bash
python run_local.py local_workflows/radiology_smoke.json
python run_local.py local_workflows/histo_monuseg.json --results-dir /tmp/test
```

Pieces execute in topological order; upstream outputs wire to downstream inputs automatically via `{"from": "node_id.field_name"}` references. Pre-built workflow JSONs are in `local_workflows/`.

---

## Releasing a new version

Bump `VERSION` in `config.toml`:

```toml
VERSION = "0.3.22"
```

Pushing this change to `master` triggers GitHub Actions to:
1. Rebuild and push both Docker images tagged `0.3.22-base` and `0.3.22-torch`
2. Regenerate `.domino/compiled_metadata.json`
3. Create git tag `0.3.22` and a GitHub Release

---

## Shared storage conventions

| Host path | Container path | Usage |
|---|---|---|
| `domino_data/medical_data/images/` | `/home/shared_storage/medical_data/images/` | Radiology NIfTI images |
| `domino_data/medical_data/masks/` | `/home/shared_storage/medical_data/masks/` | Radiology NIfTI masks |
| `domino_data/histo_data/images/` | `/home/shared_storage/histo_data/images/` | Histopathology images |
| `domino_data/histo_data/masks/` | `/home/shared_storage/histo_data/masks/` | Histopathology masks |

---

## License

See [LICENSE](LICENSE).
