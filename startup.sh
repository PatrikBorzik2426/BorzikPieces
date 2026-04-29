#!/bin/bash
# startup.sh — Boot the BorzikPieces / Domino stack after a fresh machine restart.
# Run from the project root: bash startup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ── 1. Sanity checks ────────────────────────────────────────────────────────

info "Checking Docker daemon..."
if ! docker info &>/dev/null; then
  error "Docker is not running. Start it first: sudo systemctl start docker"
  exit 1
fi

info "Checking NVIDIA runtime..."
if ! docker info 2>/dev/null | grep -q "nvidia"; then
  warn "NVIDIA runtime not detected. GPU pieces may fail."
  warn "If this is the first boot on a new machine, run: bash setup_gpu.sh"
fi

# ── 2. Create .env if missing ────────────────────────────────────────────────

if [[ ! -f .env ]]; then
  info "Creating .env (AIRFLOW_UID)..."
  echo "AIRFLOW_UID=$(id -u)" > .env
else
  info ".env already present, skipping."
fi

# ── 3. Populate shared storage if empty ─────────────────────────────────────

IMG_DEST="domino_data/medical_data/images"
MSK_DEST="domino_data/medical_data/masks"
IMG_SRC="data/paired/images"
MSK_SRC="data/paired/masks"

mkdir -p "$IMG_DEST" "$MSK_DEST"

IMG_COUNT=$(find "$IMG_DEST" -name "*.nii.gz" 2>/dev/null | wc -l)
MSK_COUNT=$(find "$MSK_DEST" -name "*.nii.gz" 2>/dev/null | wc -l)

if [[ "$IMG_COUNT" -eq 0 ]]; then
  if [[ -d "$IMG_SRC" ]]; then
    info "Copying NIfTI images to shared storage..."
    cp "$IMG_SRC"/*.nii.gz "$IMG_DEST"/
  else
    warn "No images in $IMG_SRC — shared storage will be empty."
  fi
else
  info "Shared storage already has $IMG_COUNT images, skipping copy."
fi

if [[ "$MSK_COUNT" -eq 0 ]]; then
  if [[ -d "$MSK_SRC" ]]; then
    info "Copying NIfTI masks to shared storage..."
    cp "$MSK_SRC"/*.nii.gz "$MSK_DEST"/
  else
    warn "No masks in $MSK_SRC — shared storage will be empty."
  fi
else
  info "Shared storage already has $MSK_COUNT masks, skipping copy."
fi

# ── 4. Authenticate Docker to GHCR ──────────────────────────────────────────

if [[ -f .env ]]; then
  export $(grep -E '^GHCR_(USERNAME|TOKEN)=' .env | xargs) 2>/dev/null || true
fi

if [[ -n "${GHCR_TOKEN:-}" && -n "${GHCR_USERNAME:-}" ]]; then
  info "Logging Docker into ghcr.io as ${GHCR_USERNAME}..."
  echo "$GHCR_TOKEN" | docker login ghcr.io -u "$GHCR_USERNAME" --password-stdin \
    && info "Docker authenticated to ghcr.io." \
    || warn "Docker login failed — piece image pulls may be denied."
else
  warn "GHCR_TOKEN / GHCR_USERNAME not set in .env — skipping docker login."
fi

# ── 5. Start the stack ───────────────────────────────────────────────────────

info "Starting Domino stack (docker compose up -d)..."
docker compose up -d

# ── 6. Wait for REST API ─────────────────────────────────────────────────────

API="http://localhost:8000"
info "Waiting for REST API at $API ..."
MAX_WAIT=120
ELAPSED=0
until curl -sf "$API/health" &>/dev/null || curl -sf "$API/docs" &>/dev/null; do
  if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    warn "REST API did not respond within ${MAX_WAIT}s — it may still be initialising."
    warn "Try again in a moment: curl $API/health"
    break
  fi
  printf "."
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done
[[ $ELAPSED -lt $MAX_WAIT ]] && echo "" && info "REST API is up."

# ── 7. Wait for Frontend ─────────────────────────────────────────────────────

FRONTEND="http://localhost:3000"
info "Waiting for Frontend at $FRONTEND ..."
ELAPSED=0
until curl -sf "$FRONTEND" &>/dev/null; do
  if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    warn "Frontend did not respond within ${MAX_WAIT}s — it may still be starting."
    break
  fi
  printf "."
  sleep 5
  ELAPSED=$((ELAPSED + 5))
done
[[ $ELAPSED -lt $MAX_WAIT ]] && echo "" && info "Frontend is up."

# ── 8. Check GPU inside worker ───────────────────────────────────────────────

info "Checking GPU inside airflow-domino-worker..."
if docker exec airflow-domino-worker nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; then
  info "GPU is accessible inside the worker."
else
  warn "nvidia-smi not available inside worker — GPU pieces will run on CPU."
fi

# ── 9. Dump container logs to files ──────────────────────────────────────────

LOG_DIR="logs/containers"
mkdir -p "$LOG_DIR"

CONTAINERS=(
  airflow-domino-worker
  airflow-domino-scheduler
  airflow-webserver
  domino-rest
  domino-frontend
)

info "Dumping container logs to $LOG_DIR/ ..."
for name in "${CONTAINERS[@]}"; do
  out="$LOG_DIR/${name}.txt"
  # --no-log-prefix keeps lines clean; tail last 2000 lines so files stay manageable
  if docker logs "$name" --tail 2000 2>&1 > "$out"; then
    lines=$(wc -l < "$out")
    info "  $name → $out ($lines lines)"
  else
    warn "  $name — container not running, skipping."
    rm -f "$out"
  fi
done

# Also snapshot the most recent Airflow task logs (last 5 unique tasks)
TASK_LOG_DIR="$LOG_DIR/airflow_tasks"
mkdir -p "$TASK_LOG_DIR"
mapfile -t RECENT_TASK_LOGS < <(
  find airflow/logs -name "attempt=1.log" \
    ! -path "*/dag_processor_manager/*" \
    ! -path "*/scheduler/*" \
    2>/dev/null \
    | xargs ls -t 2>/dev/null \
    | head -5
)
for src in "${RECENT_TASK_LOGS[@]}"; do
  # Build a flat filename from the path components
  flat=$(echo "$src" | sed 's|airflow/logs/||; s|/|__|g')
  cp "$src" "$TASK_LOG_DIR/$flat"
  info "  task log → $TASK_LOG_DIR/$flat"
done

# ── 10. Summary ───────────────────────────────────────────────────────────────

echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  Stack is up — open in your browser:  ${NC}"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo "  Frontend : http://localhost:3000   (admin@email.com / admin)"
echo "  REST API : http://localhost:8000/docs"
echo "  Airflow  : http://localhost:8080   (airflow / airflow)"
echo ""
echo "  Shared storage: domino_data/medical_data/"
echo "    images: $(find "$IMG_DEST" -name '*.nii.gz' 2>/dev/null | wc -l) files"
echo "    masks : $(find "$MSK_DEST" -name '*.nii.gz' 2>/dev/null | wc -l) files"
echo ""
echo "  Container logs: $LOG_DIR/"
echo "  If this is a fresh DB (after 'docker compose down -v'), import the workflow:"
echo "    bash import_workflow.sh"
echo ""
echo "  Container status:"
docker compose ps --format "table {{.Name}}\t{{.Status}}"
