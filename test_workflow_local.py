"""
Local workflow integration test — mirrors radiology_workflow.json without Docker/Airflow.

Runs the full non-GPU chain:
  NiftiDataLoader → DataSplit → PreprocessTrain/Val/Test → PituitaryDataset → (EDA + Viz in parallel)

ModelTraining and ModelInference are skipped unless --with-torch is passed and torch is installed.

Usage:
  pip install nibabel scipy pandas matplotlib seaborn tqdm plotly Pillow
  python test_workflow_local.py
  python test_workflow_local.py --with-torch    # also run training (slow, needs GPU packages)
"""

import argparse
import logging
import os
import sys
import tempfile
from domino.schemas.deploy_mode import DeployModeType

PIECE_KWARGS = dict(
    deploy_mode=DeployModeType.dry_run,
    task_id="local_test",
    dag_id="local_workflow",
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")

# Point imports at the pieces directory
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "pieces"))

# BasePiece reads these env vars on __init__
_tmp_results = tempfile.mkdtemp(prefix="domino_test_")
os.environ.setdefault("DOMINO_RESULTS_PATH", _tmp_results)
os.environ.setdefault("DOMINO_PIECE_INPUT_FILE", "")
os.environ.setdefault("DOMINO_PIECE_SECRETS_FILE", "")

# Local data paths
IMAGES_PATH = os.path.join(REPO_ROOT, "data", "paired", "images")
MASKS_PATH  = os.path.join(REPO_ROOT, "data", "paired", "masks")

GREEN = "\033[0;32m"
YELLOW = "\033[1;33m"
RED   = "\033[0;31m"
NC    = "\033[0m"

def step(name): print(f"\n{GREEN}{'─'*60}{NC}\n{GREEN}▶  {name}{NC}")
def ok(msg):    print(f"{GREEN}✓  {msg}{NC}")
def warn(msg):  print(f"{YELLOW}⚠  {msg}{NC}")
def fail(msg):  print(f"{RED}✗  {msg}{NC}")

def to_dicts(items):
    """Serialize a list of Pydantic models to dicts — mirrors what Domino does between pieces."""
    return [i.model_dump() if hasattr(i, "model_dump") else i.dict() for i in items]


def run_pipeline(with_torch: bool):
    errors = []

    # ── Step 1: NiftiDataLoader ──────────────────────────────────────────────
    step("1 / 8  NiftiDataLoaderPiece")
    from NiftiDataLoaderPiece.piece import NiftiDataLoaderPiece
    from NiftiDataLoaderPiece.models import InputModel as LoaderInput

    loader_out = NiftiDataLoaderPiece(**PIECE_KWARGS).piece_function(LoaderInput(
        images_path=IMAGES_PATH,
        masks_path=MASKS_PATH,
        file_pattern="*.nii.gz",
    ))
    ok(f"Loaded {loader_out.num_subjects} subjects")
    assert loader_out.num_subjects > 0, "No subjects found — check data/paired/"

    # ── Step 2: DataSplit ────────────────────────────────────────────────────
    step("2 / 8  DataSplitPiece")
    from DataSplitPiece.piece import DataSplitPiece
    from DataSplitPiece.models import InputModel as SplitInput

    split_out = DataSplitPiece(**PIECE_KWARGS).piece_function(SplitInput(
        subjects=to_dicts(loader_out.subjects),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42,
        split_strategy="random",
    ))
    ok(f"Train: {split_out.train_count}  Val: {split_out.val_count}  Test: {split_out.test_count}")
    assert split_out.train_count > 0

    # ── Steps 3–5: NiftiPreprocessing (train / val / test) ──────────────────
    from NiftiPreprocessingPiece.piece import NiftiPreprocessingPiece
    from NiftiPreprocessingPiece.models import InputModel as PrepInput

    prep_results = {}
    for split_name, subjects in [
        ("train", split_out.train_subjects),
        ("val",   split_out.val_subjects),
        ("test",  split_out.test_subjects),
    ]:
        step(f"3-5 / 8  NiftiPreprocessingPiece — {split_name}")
        out_dir = os.path.join(_tmp_results, "preprocessed", split_name)
        os.makedirs(out_dir, exist_ok=True)
        prep_out = NiftiPreprocessingPiece(**PIECE_KWARGS).piece_function(PrepInput(
            subjects=to_dicts(subjects),
            output_dir=out_dir,
            normalization="percentile",
            lower_percentile=1.0,
            upper_percentile=99.0,
            save_as_numpy=True,
        ))
        ok(f"{split_name}: processed={prep_out.num_processed}  failed={prep_out.num_failed}")
        if prep_out.num_failed > 0:
            warn(f"{prep_out.num_failed} subject(s) failed preprocessing in {split_name} split")
            errors.append(f"preprocessing failures in {split_name}")
        prep_results[split_name] = prep_out

    # ── Step 6: PituitaryDataset ─────────────────────────────────────────────
    step("6 / 8  PituitaryDatasetPiece")
    from PituitaryDatasetPiece.piece import PituitaryDatasetPiece
    from PituitaryDatasetPiece.models import InputModel as DatasetInput

    dataset_out = PituitaryDatasetPiece(**PIECE_KWARGS).piece_function(DatasetInput(
        train_subjects=to_dicts(prep_results["train"].preprocessed_subjects),
        val_subjects=to_dicts(prep_results["val"].preprocessed_subjects),
        test_subjects=to_dicts(prep_results["test"].preprocessed_subjects),
        batch_size=2,
        num_workers=0,
        shuffle_train=True,
    ))
    ok(f"Dataset config: {dataset_out.dataset_config_path}")
    ok(f"Total subjects: {len(dataset_out.subjects)}")

    # ── Step 7a: NiftiEDA (parallel branch from loader) ─────────────────────
    step("7a / 8  NiftiEDAPiece  (parallel branch)")
    try:
        from NiftiEDAPiece.piece import NiftiEDAPiece
        from NiftiEDAPiece.models import InputModel as EDAInput

        eda_out = NiftiEDAPiece(**PIECE_KWARGS).piece_function(EDAInput(
            subjects=to_dicts(loader_out.subjects),
            max_subjects=10,    # limit to first 10 for speed
        ))
        ok(f"EDA complete — report: {getattr(eda_out, 'report_path', 'n/a')}")
    except Exception as e:
        warn(f"EDA failed (non-fatal): {e}")
        errors.append(f"EDA: {e}")

    # ── Step 7b: NiftiVisualization (parallel branch from loader) ────────────
    step("7b / 8  NiftiVisualizationPiece  (parallel branch)")
    try:
        from NiftiVisualizationPiece.piece import NiftiVisualizationPiece
        from NiftiVisualizationPiece.models import InputModel as VizInput

        viz_out = NiftiVisualizationPiece(**PIECE_KWARGS).piece_function(VizInput(
            images_path=IMAGES_PATH,
            masks_path=MASKS_PATH,
            file_pattern="*.nii.gz",
            max_subjects=6,
            view_plane="axial",
        ))
        ok(f"Visualization complete — output: {getattr(viz_out, 'output_path', 'n/a')}")
    except Exception as e:
        warn(f"Visualization failed (non-fatal): {e}")
        errors.append(f"Visualization: {e}")

    # ── Step 8: ModelTraining + Inference (GPU, optional) ────────────────────
    if with_torch:
        step("8 / 8  ModelTrainingPiece  (use_gpu=False for local test)")
        try:
            from ModelTrainingPiece.piece import ModelTrainingPiece
            from ModelTrainingPiece.models import InputModel as TrainInput, ModelArchitecture

            model_out_dir = os.path.join(_tmp_results, "models")
            os.makedirs(model_out_dir, exist_ok=True)
            train_out = ModelTrainingPiece(**PIECE_KWARGS).piece_function(TrainInput(
                subjects=to_dicts(dataset_out.subjects),
                dataset_config_path=dataset_out.dataset_config_path,
                data_root=REPO_ROOT,
                output_dir=model_out_dir,
                model_architecture=ModelArchitecture.UNET,
                num_classes=2,
                epochs=2,           # smoke-test only
                batch_size=1,
                patch_size=32,
                samples_per_volume=5,
                use_gpu=False,
            ))
            ok(f"Training complete — best model: {train_out.best_model_path}")

            step("8 / 8  ModelInferencePiece")
            from ModelInferencePiece.piece import ModelInferencePiece
            from ModelInferencePiece.models import InputModel as InferInput

            infer_out_dir = os.path.join(_tmp_results, "inference")
            os.makedirs(infer_out_dir, exist_ok=True)
            infer_out = ModelInferencePiece(**PIECE_KWARGS).piece_function(InferInput(
                subjects=to_dicts(train_out.validation_subjects),
                model_path=train_out.best_model_path,
                model_architecture=train_out.model_architecture,
                num_classes=train_out.num_classes,
                patch_size=train_out.patch_size,
                output_dir=infer_out_dir,
                save_visualizations=True,
                save_predictions=True,
                use_gpu=False,
            ))
            ok(f"Inference complete — results: {infer_out.output_dir}")
        except Exception as e:
            fail(f"Training/Inference failed: {e}")
            errors.append(f"training: {e}")
    else:
        step("8 / 8  ModelTraining + Inference  → SKIPPED (pass --with-torch to enable)")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{GREEN}{'═'*60}{NC}")
    if errors:
        print(f"{YELLOW}Pipeline finished with {len(errors)} warning(s):{NC}")
        for e in errors:
            print(f"  {YELLOW}• {e}{NC}")
    else:
        print(f"{GREEN}All steps passed.{NC}")
    print(f"{GREEN}Results written to: {_tmp_results}{NC}")
    print(f"{GREEN}{'═'*60}{NC}\n")
    return len(errors) == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Local radiology workflow integration test")
    parser.add_argument("--with-torch", action="store_true",
                        help="Also run ModelTraining + Inference (requires nibabel+torch+monai)")
    args = parser.parse_args()

    if not os.path.isdir(IMAGES_PATH):
        fail(f"Images directory not found: {IMAGES_PATH}")
        fail("Run from the repo root and make sure data/paired/images/ exists.")
        sys.exit(1)

    success = run_pipeline(with_torch=args.with_torch)
    sys.exit(0 if success else 1)
