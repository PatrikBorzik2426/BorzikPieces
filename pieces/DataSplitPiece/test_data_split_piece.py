from domino.testing import piece_dry_run
from pieces.DataSplitPiece.models import InputModel, SubjectInfo, SplitStrategy


def create_subjects(n):
    """Create n dummy subjects"""
    return [
        SubjectInfo(
            subject_id=f"sub-{i:03d}",
            image_path=f"/data/images/sub-{i:03d}.nii.gz",
            mask_path=f"/data/masks/sub-{i:03d}.nii.gz"
        )
        for i in range(n)
    ]


def test_basic_split():
    """Test basic 70/15/15 split"""
    subjects = create_subjects(100)
    
    inp = InputModel(
        subjects=subjects,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    output = piece_dry_run("DataSplitPiece", input_data)
    
    assert output["train_count"] == 70
    assert output["val_count"] == 15
    assert output["test_count"] == 15
    assert output["total_count"] == 100


def test_reproducibility():
    """Test that same seed produces same split"""
    subjects = create_subjects(50)
    
    inp1 = InputModel(subjects=subjects, random_seed=123)
    input_data1 = inp1.model_dump() if hasattr(inp1, "model_dump") else inp1.dict()
    output1 = piece_dry_run("DataSplitPiece", input_data1)
    
    inp2 = InputModel(subjects=subjects, random_seed=123)
    input_data2 = inp2.model_dump() if hasattr(inp2, "model_dump") else inp2.dict()
    output2 = piece_dry_run("DataSplitPiece", input_data2)
    
    assert [s["subject_id"] for s in output1["train_subjects"]] == [s["subject_id"] for s in output2["train_subjects"]]


def test_sequential_split():
    """Test sequential (non-random) split"""
    subjects = create_subjects(10)
    
    inp = InputModel(
        subjects=subjects,
        train_ratio=0.5,
        val_ratio=0.3,
        test_ratio=0.2,
        split_strategy=SplitStrategy.SEQUENTIAL
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    output = piece_dry_run("DataSplitPiece", input_data)
    
    # First 5 should be train
    assert output["train_subjects"][0]["subject_id"] == "sub-000"
    assert output["train_count"] == 5


def test_small_dataset():
    """Test with small dataset"""
    subjects = create_subjects(5)
    
    inp = InputModel(
        subjects=subjects,
        train_ratio=0.6,
        val_ratio=0.2,
        test_ratio=0.2
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    output = piece_dry_run("DataSplitPiece", input_data)
    
    assert output["total_count"] == 5
    assert output["train_count"] + output["val_count"] + output["test_count"] == 5
