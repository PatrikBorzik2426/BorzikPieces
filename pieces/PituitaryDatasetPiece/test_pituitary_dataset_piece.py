from domino.testing import piece_dry_run
from pieces.PituitaryDatasetPiece.models import InputModel, PreprocessedSubject


def create_preprocessed_subjects(n, prefix="train"):
    """Create n dummy preprocessed subjects"""
    return [
        PreprocessedSubject(
            subject_id=f"{prefix}-sub-{i:03d}",
            original_image_path=f"/data/images/{prefix}-sub-{i:03d}.nii.gz",
            preprocessed_image_path=f"/data/preprocessed/images/{prefix}-sub-{i:03d}.npy",
            preprocessed_mask_path=f"/data/preprocessed/masks/{prefix}-sub-{i:03d}.npy",
            original_shape=[256, 256, 128],
            preprocessed_shape=[128, 128, 64],
            image_stats={"original": {"mean": 100}, "normalized": {"mean": 0}}
        )
        for i in range(n)
    ]


def test_create_train_dataset():
    """Test creating training dataset configuration"""
    train_subjects = create_preprocessed_subjects(10, "train")
    
    inp = InputModel(
        train_subjects=train_subjects,
        batch_size=2
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    output = piece_dry_run("PituitaryDatasetPiece", input_data)
    
    assert output["train_info"] is not None
    assert output["train_info"]["num_samples"] == 10
    assert output["train_info"]["batch_size"] == 2
    assert output["train_info"]["num_batches"] == 5


def test_create_all_splits():
    """Test creating all dataset splits"""
    train = create_preprocessed_subjects(10, "train")
    val = create_preprocessed_subjects(3, "val")
    test = create_preprocessed_subjects(2, "test")
    
    inp = InputModel(
        train_subjects=train,
        val_subjects=val,
        test_subjects=test,
        batch_size=2
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    output = piece_dry_run("PituitaryDatasetPiece", input_data)
    
    assert output["train_info"]["num_samples"] == 10
    assert output["val_info"]["num_samples"] == 3
    assert output["test_info"]["num_samples"] == 2


def test_batch_size_config():
    """Test that batch size is configurable"""
    train_subjects = create_preprocessed_subjects(5, "train")
    
    inp = InputModel(
        train_subjects=train_subjects,
        batch_size=4
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    output = piece_dry_run("PituitaryDatasetPiece", input_data)
    
    assert output["train_info"]["batch_size"] == 4
    assert output["train_info"]["num_batches"] == 2  # 5 samples / 4 batch_size = 2 batches
