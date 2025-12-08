from domino.testing import piece_dry_run
from pieces.NiftiPreprocessingPiece.models import InputModel, SubjectInfo, NormalizationMethod
import numpy as np


def test_normalization_methods():
    """Test that different normalization methods can be specified"""
    subjects = [
        SubjectInfo(
            subject_id="test-001",
            image_path="/data/test.nii.gz",
            mask_path="/data/test_mask.nii.gz"
        )
    ]
    
    # Test with zscore normalization
    inp = InputModel(
        subjects=subjects,
        output_dir="/tmp/test_output",
        normalization=NormalizationMethod.ZSCORE,
        save_as_numpy=True
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    
    # Note: This will fail without actual data files, but tests the model validation
    assert input_data["normalization"] == "zscore"
    assert input_data["save_as_numpy"] is True


def test_percentile_parameters():
    """Test percentile normalization parameters"""
    subjects = [
        SubjectInfo(
            subject_id="test-001",
            image_path="/data/test.nii.gz"
        )
    ]
    
    inp = InputModel(
        subjects=subjects,
        output_dir="/tmp/test_output",
        normalization=NormalizationMethod.PERCENTILE,
        lower_percentile=2.0,
        upper_percentile=98.0
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    
    assert input_data["normalization"] == "percentile"
    assert input_data["lower_percentile"] == 2.0
    assert input_data["upper_percentile"] == 98.0
