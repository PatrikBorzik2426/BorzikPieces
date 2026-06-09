**From:** Patrik Borzik <zaluzar00@gmail.com>
**To:**
**Subject:** Re: Docker containers and Domino workflow JSON

---

Hi,

Happy to answer both questions.

**1. Docker containers / registry**

Yes, we do have Docker containers built for all Domino pieces. We maintain two images:

- `Dockerfile_base` — lightweight image for non-ML pieces (data loading, preprocessing, EDA, visualization); dependencies: nibabel, matplotlib, scipy, scikit-image
- `Dockerfile_torch` — heavy image for ML pieces (training, inference); torch==2.1.2+cu121, monai[all]==1.3.0, scikit-learn

Both images are published to **GitHub Container Registry (GHCR)** under:

```
ghcr.io/patrikborzik2426/borzikpieces:<version>-<group>
```

We are not using Harbor. The images are versioned alongside the piece repository (currently `0.3.21`) and rebuilt automatically via GitHub Actions whenever `config.toml` is updated. If Harbor integration is required for WP5 infrastructure, we can push to an additional registry without changes to the pieces themselves — just an extra step in the CI pipeline.

**2. Domino workflow JSON**

I have attached the exported Domino workflow JSONs for both use cases:

- `radiology_domino_workflow.json` — the radiology pipeline (pituitary/brain lesion MRI segmentation)
- `histo_domino_workflow.json` — the histopathology pipeline (still a work in progress, pending data for full end-to-end validation)

You are correct that the export is done via the Domino GUI (Workflows → Export), and the result is a self-contained JSON that includes piece schemas, input configurations, and the DAG topology. No manual edits were made to either file.

**Note on current status:** Both workflows are currently experiencing failures that I am actively working on. The histopathology workflow was failing at the data loading step because the data directory (`/home/shared_storage/histo_data/images` and `/home/shared_storage/histo_data/masks`) did not exist on the host machine — the workflow has no mechanism to fetch data automatically, so the input data must be placed into the shared storage volume before the workflow is run. This is a general requirement for both pipelines: the data loader pieces expect the imaging data to already be present at a configured local path on the machine running Domino. For the radiology workflow, the training step is failing with an error about the channel dimension being of length 1, which I am still investigating. I have bumped the piece repository to version **0.3.21** (already reflected in the attached JSONs), which targets these issues. I will follow up once the fixes are verified.

Please let me know if any further information is needed to build the AWPL representation.

Best regards,
Patrik Borzik

---

## 2.1.3 AI Modules for Applications

- GitLab/Hub code repository link(s)
- Git tag + GitLab Release link
- Reference to: Container images / Helm charts / Packages
- Documentation locations (if there are any, if not process the doc into the readme of BorzikPieces)

### 2.1.3.1 How to run

- Broader context descriptions
    - for complex setups (multiple repositories)
- GitLab repository Quickstart link(s)
- Run steps (for multiple repostitories)
- Expectec output / success criteria

---

## 2.1.3 AI Modules for Applications — Answers

**GitHub repository**
https://github.com/PatrikBorzik2426/BorzikPieces
(Note: we are on GitHub, not GitLab.)

**Git tag + GitHub Release link**
Current release: `0.3.21`
https://github.com/PatrikBorzik2426/BorzikPieces/releases/tag/0.3.21

**Container images**
Two Docker images are published to GitHub Container Registry (GHCR) on every release:

| Image | Tag pattern | Purpose |
|---|---|---|
| `ghcr.io/patrikborzik2426/borzikpieces` | `0.3.21-base` | Non-ML pieces (data loading, preprocessing, EDA, visualization) |
| `ghcr.io/patrikborzik2426/borzikpieces` | `0.3.21-torch` | ML pieces (training, inference) — torch 2.1.2+cu121, MONAI 1.3.0 |

No Helm charts — Domino handles piece container scheduling internally via Airflow. No separate package registry; pieces are installed directly through the Domino UI.

**Documentation**
Full documentation is in the repository `README.md`:
https://github.com/PatrikBorzik2426/BorzikPieces/blob/master/README.md

Covers: piece catalogue (radiology + histopathology), Docker images, prerequisites, quick-start (full stack and local runner), multi-repo registration in Domino, shared storage conventions, and release process.

---

### 2.1.3.1 How to Run — Answers

**Broader context — complex setups (multiple repositories)**
Domino natively supports multiple piece repositories registered simultaneously. Each repository is independent: its own GitHub URL, version tag, and Docker images. Pieces from all registered repos appear together in the workflow builder. BorzikPieces is currently a single repo but is structured to split (independent `config.toml`, versioning, and Dockerfiles), so a second repo (e.g. a partner's pieces) can be added without any changes here.

**GitHub repository Quickstart**
https://github.com/PatrikBorzik2426/BorzikPieces — no dedicated quickstart page yet; setup steps are currently only in the internal `.claude/setup_guide.md`.

**Run steps (for multiple repositories)**

*Option A — Full Domino stack (production-equivalent):*
1. Start the Domino stack: `docker compose up -d` (frontend: http://localhost:3000, Airflow: http://localhost:8080)
2. In Domino UI → **Settings → Piece Repositories** → paste each repository's GitHub URL → select version tag → **Add**. Repeat for each additional repo.
3. Domino pulls the Docker images and registers all pieces (takes a few minutes per repo).
4. Open **Workflows** → build or import a workflow JSON → **Run**.
5. Input data must be pre-placed under `./domino_data/` on the host (mounted to `/home/shared_storage/` inside workers).

*Option B — Local runner (no Docker/Airflow required, fastest for development):*
```bash
python run_local.py local_workflows/radiology_smoke.json
python run_local.py local_workflows/histo_monuseg.json --results-dir /tmp/test
```
Pieces from any registered repo can be referenced by name in the workflow JSON; outputs wire between pieces automatically.

**Expected output / success criteria**
- *Domino stack:* all tasks in the Airflow DAG show status **success** (green). The Domino UI workflow view shows each piece node with a result card (HTML gallery for visualization pieces, JSON summary for others).
- *Local runner:* script exits with `✓ Workflow completed` and a results directory populated with per-piece output files.
- *End-to-end radiology pipeline:* `ModelInferencePiece` produces a segmentation mask NIfTI file and a dice/IoU score summary without error.
- *End-to-end histopathology pipeline:* `HistoInferencePiece` produces patch-level predictions without error (requires MoNuSeg data in shared storage).