from domino.testing import piece_dry_run
from pieces.NiftiDataLoaderPiece.models import InputModel


def test_model_validation():
    """Test that InputModel validates correctly"""
    # Test valid input with both images and masks
    inp = InputModel(
        images_path="/data/paired/images",
        masks_path="/data/paired/masks",
        file_pattern="*.nii.gz"
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    
    assert input_data["images_path"] == "/data/paired/images"
    assert input_data["masks_path"] == "/data/paired/masks"
    assert input_data["file_pattern"] == "*.nii.gz"


def test_file_pattern_customization():
    """Test that custom file patterns are accepted"""
    inp = InputModel(
        images_path="/data/test",
        masks_path=None,
        file_pattern="*_T1.nii.gz"
    )
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    
    assert input_data["file_pattern"] == "*_T1.nii.gz"
    assert input_data["masks_path"] is None
