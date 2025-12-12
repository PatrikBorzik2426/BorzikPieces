import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import InputModel, SubjectInfo
import tempfile


def test_nifti_eda_piece():
    """Test NiftiEDAPiece with minimal configuration"""
    
    # Mock subjects - in real testing, these paths would exist
    subjects = [
        SubjectInfo(
            subject_id="test-001",
            image_path="/home/shared_storage/medical_data/images/sub-001.nii.gz",
            mask_path="/home/shared_storage/medical_data/masks/sub-001.nii.gz"
        )
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        input_data = InputModel(
            subjects=subjects,
            output_dir=tmpdir,
            max_subjects=10,
            generate_3d_plots=False,
            num_sample_slices=5
        )
        
        # Verify input model
        assert len(input_data.subjects) == 1
        assert input_data.max_subjects == 10
        assert input_data.generate_3d_plots is False
        assert input_data.num_sample_slices == 5
        
        # Note: Full execution test requires actual NIfTI files
        # This test validates the input model structure


if __name__ == "__main__":
    test_nifti_eda_piece()
    print("NiftiEDAPiece tests passed!")
