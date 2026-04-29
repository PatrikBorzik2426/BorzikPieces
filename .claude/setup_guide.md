# Setup Guide — BorzikPieces Domino MLOps Pipeline

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| Python | 3.9+ | system |
| Docker Engine | 20.0+ | https://docs.docker.com/engine/install/ |
| Docker Compose V2 | 2.0+ | included with Docker Desktop |
| Git | any | system |
| NVIDIA Container Toolkit | latest | see Section 0 below (GPU) |

Optional but recommended:
- GitHub account + personal access token (for installing pieces from private repos)

---

## 0. Enable GPU Support (One-Time Host Setup)

The machine has an **RTX 4060 (8 GB VRAM, CUDA 13.0 driver)**. Run these steps ONCE to give Docker access to the GPU.

### Step 1 — Install NVIDIA Container Toolkit

```bash
# Add NVIDIA package repository
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### Step 2 — Configure Docker to use NVIDIA as default runtime

This is critical for Domino: piece execution containers are created by the Docker daemon via the docker-proxy, not docker-compose. Setting the default runtime ensures ALL containers (including piece containers) get GPU access.

```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

### Step 3 — Verify

```bash
# Should show GPU info
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

### Step 4 — (Optional) Increase shared memory for PyTorch DataLoader

```bash
# If you see "Bus error" or DataLoader crashes, add this to /etc/docker/daemon.json:
# { "default-shm-size": "2g" }
sudo systemctl restart docker
```

> After this setup, the `docker-compose.yaml` `airflow-worker` service will automatically receive GPU access via the `deploy.resources.reservations.devices` config already in place.

---

---

## 1. Start the Domino Stack Locally

The `docker-compose.yaml` in the project root starts all required services.

```bash
# Set Airflow user ID (required on Linux to avoid permission issues)
echo -e "AIRFLOW_UID=$(id -u)" > .env

# Start all services in background
docker compose up -d

# Watch logs until healthy (takes 3-5 minutes on first run)
docker compose logs -f domino_rest
```

**Services started:**

| Service | URL | Credentials |
|---------|-----|-------------|
| Domino Frontend | http://localhost:3000 | admin / admin (set on first login) |
| Domino REST API | http://localhost:8000 | — |
| Airflow Webserver | http://localhost:8080 | airflow / airflow |
| Airflow Postgres | localhost:5432 | airflow / airflow |
| Domino Postgres | localhost:5433 | postgres / postgres |

```bash
# Stop everything
docker compose down

# Stop and remove all data (full reset)
docker compose down -v
```

---

## 2. Install This Piece Repository in Domino

1. Push this repo to GitHub (or use an existing GitHub URL)
2. Open Domino Frontend → **http://localhost:3000**
3. Log in (first login creates your account)
4. Go to **Settings → Piece Repositories**
5. Paste the GitHub URL of this repository
6. Select the version (matches `VERSION` in `config.toml`)
7. Click **Add Repository to Workspace**

Domino will pull the Docker images and install the pieces. This takes several minutes on first install.

> If the repo is private, provide a GitHub token at startup:
> `DOMINO_DEFAULT_PIECES_REPOSITORY_TOKEN=ghp_xxx docker compose up -d`

---

## 3. Mount the Radiology Data

The workflow reads data from `/home/shared_storage/` inside the worker container.

The `airflow-worker` service mounts `${PWD}/domino_data → /home/shared_storage`.

```bash
# Create the shared storage directory
mkdir -p domino_data/medical_data/images
mkdir -p domino_data/medical_data/masks

# Copy the sample data
cp data/paired/images/*.nii.gz domino_data/medical_data/images/
cp data/paired/masks/*.nii.gz  domino_data/medical_data/masks/

# Or use the full tp-radiology-adonema data
cp tp-radiology-adonema/tp_radiology_adonema/data/paired/images/*.nii.gz \
   domino_data/medical_data/images/
cp tp-radiology-adonema/tp_radiology_adonema/data/paired/masks/*.nii.gz \
   domino_data/medical_data/masks/
```

---

## 4. Build the Workflow in Domino UI

### Step-by-step in the Frontend

1. **New Workflow** → give it a name (e.g., "Radiology Segmentation")
2. Drag pieces from the left panel onto the canvas in this order:

```
NiftiDataLoaderPiece
    ↓
DataSplitPiece
    ↓ (three outputs: train_subjects, val_subjects, test_subjects)
NiftiPreprocessingPiece × 3
    ↓ (all three outputs → PituitaryDatasetPiece)
PituitaryDatasetPiece
    ↓
ModelTrainingPiece
    ↓
ModelInferencePiece
```

And separately:
```
NiftiDataLoaderPiece → NiftiEDAPiece → NiftiVisualizationPiece
```

3. **Configure each piece** (click on it to open the config panel):

### NiftiDataLoaderPiece
```
images_path:  /home/shared_storage/medical_data/images
masks_path:   /home/shared_storage/medical_data/masks
file_pattern: *.nii.gz
```

### DataSplitPiece
```
train_ratio: 0.70
val_ratio:   0.15
test_ratio:  0.15
random_seed: 42
```

### NiftiPreprocessingPiece (TRAIN instance)
```
output_dir:   /home/shared_storage/medical_data/preprocessed/train
normalization: percentile
save_as_numpy: true
```

### NiftiPreprocessingPiece (VAL instance)
```
output_dir:   /home/shared_storage/medical_data/preprocessed/val
normalization: percentile
save_as_numpy: true
```

### NiftiPreprocessingPiece (TEST instance)
```
output_dir:   /home/shared_storage/medical_data/preprocessed/test
normalization: percentile
save_as_numpy: true
```

### PituitaryDatasetPiece
```
batch_size:    2
num_workers:   0
shuffle_train: true
```

### ModelTrainingPiece
```
model_architecture: unet
num_classes:        2          # binary: background + lesion
epochs:             50
batch_size:         2
learning_rate:      0.0001
patch_size:         64
samples_per_volume: 20
use_gpu:            true       # set false if no GPU
output_dir:         /home/shared_storage/models
num_workers:        0          # keep 0 in Docker to avoid shared-memory issues
```

### ModelInferencePiece
```
model_architecture: unet
num_classes:        2
output_dir:         /home/shared_storage/inference_results
save_visualizations: true
save_predictions:    true
num_workers:         0
```

4. **Save** the workflow → **Run**

---

## 5. Developing / Updating Pieces

### Local piece testing (without Domino)

```bash
cd /home/borzito/code/BorzikPieces

# Install the domino SDK
pip install domino-py

# Run a piece's test file
python -m pytest pieces/NiftiDataLoaderPiece/test_nifti_data_loader_piece.py -v
```

### Updating a piece

1. Edit `pieces/<PieceName>/piece.py` or `models.py`
2. Bump `VERSION` in `config.toml` (Domino uses this to detect new releases)
3. Push to GitHub
4. In Domino UI → Settings → Piece Repositories → update version

### Rebuilding Docker images manually

```bash
# Build the base image
docker build -f dependencies/Dockerfile_base -t borzikpieces-base:local .

# Build the torch image
docker build -f dependencies/Dockerfile_torch -t borzikpieces-torch:local .
```

---

## 6. Troubleshooting

### Domino worker can't find data files
- Check that `domino_data/` exists in the project root
- Verify the worker container sees it: `docker exec airflow-domino-worker ls /home/shared_storage`

### Out of shared memory (PyTorch DataLoader)
- Set `num_workers: 0` in any piece that uses DataLoader (ModelTrainingPiece, ModelInferencePiece)
- Docker containers have limited `/dev/shm` by default

### Piece fails with import error (torch/monai not found)
- Verify `metadata.json` says `"dockerfile": "Dockerfile_torch"` (not `Dockerfile_base`)
- Rebuild: `docker compose pull` or re-install the piece repository

### NiftiPreprocessingPiece overwrites data
- Three parallel instances MUST have different `output_dir` values:
  - train → `/home/shared_storage/.../preprocessed/train`
  - val   → `/home/shared_storage/.../preprocessed/val`
  - test  → `/home/shared_storage/.../preprocessed/test`

### Airflow webserver at localhost:8080 shows no DAGs
- DAGs are generated by Domino when you run a workflow from the frontend
- Check the scheduler logs: `docker compose logs airflow-domino-scheduler`

### First run takes very long
- Docker images are ~5-10 GB; allow 10-20 min on first pull
- Subsequent runs use the cached images

---

## 7. Current Status (as of 2026-04-28)

### Radiology Pipeline
| Component | Status | Notes |
|-----------|--------|-------|
| NiftiDataLoaderPiece | ✅ Ready | Loads NIfTI pairs, outputs SubjectInfo list |
| DataSplitPiece | ✅ Ready | Random/sequential split with seed |
| NiftiPreprocessingPiece | ✅ Ready | z-score/minmax/percentile norm, numpy save |
| PituitaryDatasetPiece | ✅ Ready | Merges splits, creates dataset config |
| ModelTrainingPiece | ✅ Ready (3D MONAI) | UNet + SwinUNETR, patch-based, early stopping |
| ModelInferencePiece | ✅ Ready | Confidence maps, Dice metric, visualizations |
| NiftiEDAPiece | ✅ Ready | 8-phase EDA, HTML report, Slovak analysis text |
| NiftiVisualizationPiece | ✅ Ready | Standalone NIfTI grid viewer |
| Docker images | ✅ Ready | Dockerfile_base + Dockerfile_torch (pinned) |
| docker-compose.yaml | ✅ Ready | Full Domino stack |
| Domino workflow file | ⬜ TODO | Must be built in the UI and saved as JSON |

### Histopathology Pipeline
| Component | Status | Notes |
|-----------|--------|-------|
| All pieces | ⬜ TODO | No data provided yet |

### Architecture note
The reference radiology project (`tp-radiology-adonema`) uses 2D/2.5D SMP UNet++ with ResNet34 encoder.
The current pieces use 3D MONAI UNet/SwinUNETR (patch-based). This is a **valid and complementary approach** —
3D methods better capture volumetric context for small structures like the pituitary gland.

If 2D SMP approach is preferred, ModelTrainingPiece would need to be updated to use
`segmentation_models_pytorch` + `albumentations` instead of MONAI.
