from domino.testing import piece_dry_run
from pieces.NiftiDataLoaderPiece.models import InputModel
import pytest
import os
import tempfile
import gzip


def create_dummy_nifti(filepath):
    """Create a dummy .nii.gz file for testing"""
    # Create minimal NIfTI-like content (just header placeholder)
    dummy_content = b"NIFTI_DUMMY_CONTENT"
    with gzip.open(filepath, 'wb') as f:
        f.write(dummy_content)


def test_basic_loading():
    """Test basic NIfTI file discovery"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        images_dir = os.path.join(tmpdir, "images")
        masks_dir = os.path.join(tmpdir, "masks")
        os.makedirs(images_dir)
        os.makedirs(masks_dir)
        
        # Create dummy files
        for i in range(3):
            create_dummy_nifti(os.path.join(images_dir, f"sub-{i:03d}.nii.gz"))
            create_dummy_nifti(os.path.join(masks_dir, f"sub-{i:03d}.nii.gz"))
        
        inp = InputModel(
            images_path=images_dir,
            masks_path=masks_dir,
            file_pattern="*.nii.gz"
        )
        input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
        output = piece_dry_run("NiftiDataLoaderPiece", input_data)
        
        assert output["num_subjects"] == 3
        assert len(output["subjects"]) == 3
        assert all(s["mask_path"] is not None for s in output["subjects"])


def test_images_only():
    """Test loading without masks"""
    with tempfile.TemporaryDirectory() as tmpdir:
        images_dir = os.path.join(tmpdir, "images")
        os.makedirs(images_dir)
        
        for i in range(2):
            create_dummy_nifti(os.path.join(images_dir, f"sub-{i:03d}.nii.gz"))
        
        inp = InputModel(
            images_path=images_dir,
            masks_path=None,
            file_pattern="*.nii.gz"
        )
        input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
        output = piece_dry_run("NiftiDataLoaderPiece", input_data)
        
        assert output["num_subjects"] == 2
        assert all(s["mask_path"] is None for s in output["subjects"])
