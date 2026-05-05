# Local Workflow Testing

Run Domino pieces directly in Python — no Docker, no Airflow — using a JSON file that mirrors the workflow wiring.

## Why

Iterating via the full Domino stack (commit → CI build → push to GHCR → re-register repo → run workflow) takes ~10 minutes per cycle. The local runner lets you verify piece logic and data wiring in seconds, on the same machine, before pushing anything.

## Quick start

```bash
# Histo — base pieces only (no torch required, ~10 s)
python3 run_local.py local_workflows/histo_monuseg_base_only.json

# Histo — full pipeline including training (torch required, slow)
python3 run_local.py local_workflows/histo_monuseg.json

# Radiology — non-GPU chain (no torch required, ~30 s)
python3 run_local.py local_workflows/radiology_smoke.json

# Any workflow with verbose input/output logging
python3 run_local.py local_workflows/histo_monuseg_base_only.json --verbose

# Custom output directory
python3 run_local.py local_workflows/histo_monuseg_base_only.json --results-dir /tmp/mytest
```

## Verified outputs (2026-05-05)

### `histo_monuseg_base_only.json` — 4/4 passed

| Step | Piece | Key output |
|------|-------|-----------|
| 1/4 | `HistoDataLoaderPiece` | 51 image/mask pairs found |
| 2/4 | `HistoEDAPiece` | HTML report; 74.7% background / 25.3% nucleus |
| 3/4 | `HistoPatchExtractorPiece` | 459 patches extracted (256×256, stride=256) |
| 4/4 | `HistoDataSplitPiece` | train=321 / val=68 / test=70 |

### `radiology_smoke.json` — 7/7 passed

| Step | Piece | Key output |
|------|-------|-----------|
| 1/7 | `NiftiDataLoaderPiece` | 50 subjects loaded |
| 2/7 | `NiftiEDAPiece` | EDA report + 6 visualizations |
| 3/7 | `DataSplitPiece` | train=35 / val=7 / test=8 |
| 4/7 | `NiftiPreprocessingPiece` [train] | 35 processed, 0 failed |
| 5/7 | `NiftiPreprocessingPiece` [val] | 7 processed, 0 failed |
| 6/7 | `NiftiPreprocessingPiece` [test] | 8 processed, 0 failed |
| 7/7 | `PituitaryDatasetPiece` | dataset config written |

## Workflow JSON format

```json
{
  "name": "my_workflow",
  "description": "Optional description shown in the header",
  "results_dir": "/tmp/my_results",
  "pieces": {
    "node_id": {
      "piece": "PieceName",
      "inputs": {
        "static_field": "value",
        "wired_field":  {"from": "upstream_node_id.output_field_name"}
      }
    }
  }
}
```

- **`node_id`** — arbitrary string key; used as the target of `{"from": "..."}` references and as the subdirectory name under `results_dir/shared_storage/`.
- **`{"from": "node.field"}`** — replaced at runtime with the serialised value of `field` from the `OutputModel` of `node`. Mirrors exactly how Domino passes data between pieces.
- **`results_dir`** — where all piece outputs land. Defaults to a temp dir if omitted. Each piece gets `results_dir/shared_storage/<node_id>/results/`.
- Piece execution order is determined automatically by topological sort — write nodes in any order in the JSON.

## Available workflow files

| File | Pipeline | Torch needed | Approx. runtime |
|------|----------|-------------|-----------------|
| `histo_monuseg_base_only.json` | Histo: DataLoader → EDA → Patches → Split | No | ~10 s |
| `histo_monuseg.json` | Histo: full pipeline incl. training | Yes | minutes |
| `radiology_smoke.json` | Radiology: DataLoader → EDA → Split → 3×Preprocess → Dataset | No | ~30 s |

## How the runner works

`run_local.py` does four things:

1. **Topological sort** — reads `{"from": "..."}` references to build a DAG, then sorts it with Kahn's algorithm so every node runs after all its dependencies.
2. **Dynamic import** — imports `pieces/<PieceName>/piece.py` and `models.py` at runtime; no registry or metadata.json needed.
3. **Path injection** — `BasePiece.__init__` does not set `results_path` when called directly (that only happens inside `run_piece_function` which assumes `/home/shared_storage` is Docker-mounted). The runner injects `results_path`, `xcom_path`, and `report_path` pointing under `results_dir` after each piece is constructed.
4. **Output serialisation** — Pydantic OutputModel instances are converted to plain dicts (matching what Domino does via XCom JSON) before being stored and passed to downstream pieces.

Failures are non-fatal: a failed piece stores an empty output dict and downstream pieces that don't depend on it still run. The full failure list is printed at the end.

## Writing a new workflow JSON

1. Pick a `node_id` for each piece (e.g. `"loader"`, `"split"`, `"training"`).
2. For each input field:
   - Use a plain value (`"value"`, `42`, `true`) if it is static.
   - Use `{"from": "node_id.field_name"}` if it comes from an upstream piece — `field_name` must match exactly the field name in that piece's `OutputModel`.
3. Any field with a default in the piece's `InputModel` can be omitted from the JSON.
4. If a piece writes files to a path that defaults to `/home/shared_storage/...`, override that field in the JSON to a writable local path (e.g. `"/tmp/my_test/patches"`).
5. Run with `--verbose` to see every resolved input and every serialised output — useful for finding the exact output field names.

### Finding output field names

```bash
python3 -c "
import sys; sys.path.insert(0, 'pieces')
from HistoTrainingPiece.models import OutputModel
print([f for f in OutputModel.model_fields])
"
```

Or run with `--verbose` and read the `Outputs:` block printed after each piece.
