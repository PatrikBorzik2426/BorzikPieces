"""
Local workflow runner — executes Domino pieces directly in Python without Docker or Airflow.

Reads a workflow JSON that defines pieces, their inputs, and how outputs wire between them.
Pieces run in topological order; outputs from upstream pieces are automatically passed to
downstream inputs via {"from": "node_id.output_field"} references.

Usage:
    python run_local.py local_workflows/histo_monuseg.json
    python run_local.py local_workflows/radiology_smoke.json
    python run_local.py local_workflows/histo_monuseg.json --results-dir /tmp/mytest

Workflow JSON format:
    {
      "name": "my_workflow",
      "description": "Optional description",
      "results_dir": "/tmp/local_test",   // optional, defaults to a temp dir
      "pieces": {
        "loader": {
          "piece": "HistoDataLoaderPiece",
          "inputs": {
            "images_path": "/path/to/images",
            "masks_path": "/path/to/masks"
          }
        },
        "split": {
          "piece": "HistoDataSplitPiece",
          "inputs": {
            "samples": {"from": "loader.samples"},
            "train_ratio": 0.7
          }
        }
      }
    }

Upstream references use {"from": "node_id.field_name"} where field_name is the output
field name from the referenced piece's OutputModel. The value is passed as-is (already
serialised to JSON-compatible Python types, matching how Domino transfers data).
"""

import argparse
import json
import logging
import os
import sys
import tempfile
from collections import defaultdict, deque
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_local")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "pieces"))

GREEN  = "\033[0;32m"
YELLOW = "\033[1;33m"
RED    = "\033[0;31m"
BOLD   = "\033[1m"
NC     = "\033[0m"


# ── helpers ──────────────────────────────────────────────────────────────────

def _serialize(value: Any) -> Any:
    """Recursively convert Pydantic models → dicts so they can cross piece boundaries."""
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    return value


def _resolve_inputs(node_inputs: dict, outputs: dict[str, dict]) -> dict:
    """
    Replace {"from": "node.field"} references with the actual upstream output value.
    Raises KeyError if the referenced node or field hasn't been produced yet.
    """
    resolved = {}
    for key, spec in node_inputs.items():
        if isinstance(spec, dict) and "from" in spec and len(spec) == 1:
            ref = spec["from"]
            node_id, field = ref.split(".", 1)
            if node_id not in outputs:
                raise KeyError(f"Node '{node_id}' has not run yet (referenced by input '{key}')")
            if field not in outputs[node_id]:
                raise KeyError(
                    f"Node '{node_id}' has no output field '{field}'. "
                    f"Available: {list(outputs[node_id].keys())}"
                )
            resolved[key] = outputs[node_id][field]
        else:
            resolved[key] = spec
    return resolved


def _topo_sort(nodes: dict) -> list[str]:
    """Kahn's algorithm — returns node IDs in execution order."""
    deps: dict[str, set] = {nid: set() for nid in nodes}
    for nid, cfg in nodes.items():
        for spec in cfg.get("inputs", {}).values():
            if isinstance(spec, dict) and "from" in spec and len(spec) == 1:
                upstream_node = spec["from"].split(".")[0]
                deps[nid].add(upstream_node)

    in_degree = {n: len(d) for n, d in deps.items()}
    rdeps: dict[str, list] = defaultdict(list)
    for n, d in deps.items():
        for u in d:
            rdeps[u].append(n)

    queue = deque(n for n, deg in in_degree.items() if deg == 0)
    order = []
    while queue:
        n = queue.popleft()
        order.append(n)
        for child in rdeps[n]:
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    if len(order) != len(nodes):
        cycle = set(nodes) - set(order)
        raise ValueError(f"Cycle detected in workflow — nodes involved: {cycle}")
    return order


def _import_piece(piece_name: str):
    """Dynamically import PieceClass and InputModel from pieces/<piece_name>/."""
    import importlib
    piece_mod  = importlib.import_module(f"{piece_name}.piece")
    models_mod = importlib.import_module(f"{piece_name}.models")
    piece_cls  = getattr(piece_mod, piece_name)
    input_cls  = getattr(models_mod, "InputModel")
    return piece_cls, input_cls


def _setup_piece_paths(piece_obj: Any, base_dir: str, node_id: str) -> None:
    """
    BasePiece.__init__ does NOT set results_path — that happens inside run_piece_function
    which reads from /home/shared_storage (a Docker-only mount). When calling piece_function
    directly we must inject the paths ourselves so pieces that write files don't crash.
    """
    from pathlib import Path
    shared = os.path.join(base_dir, "shared_storage")
    piece_obj.workflow_shared_storage_path = shared
    piece_obj.results_path = os.path.join(shared, node_id, "results")
    piece_obj.xcom_path    = os.path.join(shared, node_id, "xcom")
    piece_obj.report_path  = os.path.join(shared, node_id, "report")
    for p in (piece_obj.results_path, piece_obj.xcom_path, piece_obj.report_path):
        Path(p).mkdir(parents=True, exist_ok=True)


# ── runner ───────────────────────────────────────────────────────────────────

def run_workflow(workflow_path: str, results_dir: str | None = None, verbose: bool = False):
    with open(workflow_path) as f:
        wf = json.load(f)

    name        = wf.get("name", os.path.basename(workflow_path))
    description = wf.get("description", "")
    nodes       = wf["pieces"]

    results_dir = results_dir or wf.get("results_dir") or tempfile.mkdtemp(prefix="domino_local_")
    os.makedirs(results_dir, exist_ok=True)

    # env vars BasePiece reads on __init__
    os.environ["DOMINO_RESULTS_PATH"]        = results_dir
    os.environ.setdefault("DOMINO_PIECE_INPUT_FILE",   "")
    os.environ.setdefault("DOMINO_PIECE_SECRETS_FILE", "")

    from domino.schemas.deploy_mode import DeployModeType
    piece_kwargs = dict(
        deploy_mode=DeployModeType.dry_run,
        task_id="local_test",
        dag_id="local_workflow",
    )

    print(f"\n{BOLD}{'═' * 62}{NC}")
    print(f"{BOLD}  {name}{NC}")
    if description:
        print(f"  {description}")
    print(f"  Results → {results_dir}")
    print(f"{BOLD}{'═' * 62}{NC}")

    order    = _topo_sort(nodes)
    outputs  = {}   # node_id → {field: value}
    failures = []

    for step_num, node_id in enumerate(order, 1):
        cfg        = nodes[node_id]
        piece_name = cfg["piece"]
        label      = f"{step_num}/{len(order)}  {BOLD}{piece_name}{NC}  [{node_id}]"
        print(f"\n{GREEN}▶  {label}{NC}")

        try:
            piece_cls, input_cls = _import_piece(piece_name)
            resolved = _resolve_inputs(cfg.get("inputs", {}), outputs)

            if verbose:
                log.info("Resolved inputs: %s", json.dumps(resolved, indent=2, default=str))

            input_obj = input_cls(**resolved)
            piece_obj = piece_cls(**piece_kwargs)
            _setup_piece_paths(piece_obj, results_dir, node_id)
            result    = piece_obj.piece_function(input_obj)
            serialized = _serialize(result)
            outputs[node_id] = serialized

            if verbose:
                log.info("Outputs: %s", json.dumps(serialized, indent=2, default=str))

            # Print a short summary of scalar outputs
            summary_fields = {
                k: v for k, v in serialized.items()
                if not isinstance(v, (list, dict)) or (isinstance(v, list) and len(v) <= 5)
            }
            for k, v in summary_fields.items():
                print(f"   {GREEN}✓{NC}  {k} = {v}")

        except Exception as exc:
            print(f"   {RED}✗  {piece_name} FAILED: {exc}{NC}")
            if verbose:
                import traceback
                traceback.print_exc()
            failures.append((node_id, piece_name, exc))
            # Don't abort — let non-dependent downstream nodes still run if possible
            outputs[node_id] = {}

    # ── summary ──────────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'═' * 62}{NC}")
    if failures:
        print(f"{RED}  {len(failures)} piece(s) FAILED:{NC}")
        for nid, pname, exc in failures:
            print(f"  {RED}✗  [{nid}] {pname}: {exc}{NC}")
        print(f"{BOLD}{'═' * 62}{NC}\n")
        return False
    else:
        print(f"{GREEN}  All {len(order)} pieces passed.{NC}")
        print(f"  Results: {results_dir}")
        print(f"{BOLD}{'═' * 62}{NC}\n")
        return True


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a local_workflows/*.json workflow without Docker or Airflow"
    )
    parser.add_argument("workflow", help="Path to the workflow JSON file")
    parser.add_argument("--results-dir", help="Directory for piece outputs (default: temp dir)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full resolved inputs and outputs for each piece")
    args = parser.parse_args()

    ok = run_workflow(args.workflow, results_dir=args.results_dir, verbose=args.verbose)
    sys.exit(0 if ok else 1)
