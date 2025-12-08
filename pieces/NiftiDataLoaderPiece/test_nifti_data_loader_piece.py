import pytest
import os
import tempfile
import gzip
from pieces.NiftiDataLoaderPiece.piece import NiftiDataLoaderPiece
from pieces.NiftiDataLoaderPiece.models import InputModel


class TestNiftiDataLoaderPiece:
    """Tests for NiftiDataLoaderPiece"""

    def setup_method(self):
        """Set up test fixtures"""
        self.piece = NiftiDataLoaderPiece()
        
    def create_dummy_nifti(self, filepath):
        """Create a dummy .nii.gz file for testing"""
        # Create minimal NIfTI-like content (just header placeholder)
        dummy_content = b"NIFTI_DUMMY_CONTENT"
        with gzip.open(filepath, 'wb') as f:
            f.write(dummy_content)

    def test_basic_loading(self):
        """Test basic NIfTI file discovery"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test structure
            images_dir = os.path.join(tmpdir, "images")
            masks_dir = os.path.join(tmpdir, "masks")
            os.makedirs(images_dir)
            os.makedirs(masks_dir)
            
            # Create dummy files
            for i in range(3):
                self.create_dummy_nifti(os.path.join(images_dir, f"sub-{i:03d}.nii.gz"))
                self.create_dummy_nifti(os.path.join(masks_dir, f"sub-{i:03d}.nii.gz"))
            
            input_data = InputModel(
                images_path=images_dir,
                masks_path=masks_dir,
                file_pattern="*.nii.gz"
            )
            
            output = self.piece.piece_function(input_data)
            
            assert output.num_subjects == 3
            assert len(output.subjects) == 3
            assert all(s.mask_path is not None for s in output.subjects)

    def test_images_only(self):
        """Test loading without masks"""
        with tempfile.TemporaryDirectory() as tmpdir:
            images_dir = os.path.join(tmpdir, "images")
            os.makedirs(images_dir)
            
            for i in range(2):
                self.create_dummy_nifti(os.path.join(images_dir, f"sub-{i:03d}.nii.gz"))
            
            input_data = InputModel(
                images_path=images_dir,
                masks_path=None,
                file_pattern="*.nii.gz"
            )
            
            output = self.piece.piece_function(input_data)
            
            assert output.num_subjects == 2
            assert all(s.mask_path is None for s in output.subjects)

    def test_missing_directory(self):
        """Test error handling for missing directory"""
        input_data = InputModel(
            images_path="/nonexistent/path",
            masks_path=None
        )
        
        with pytest.raises(ValueError, match="Images directory not found"):
            self.piece.piece_function(input_data)

    def test_empty_directory(self):
        """Test error handling for empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_data = InputModel(
                images_path=tmpdir,
                masks_path=None
            )
            
            with pytest.raises(ValueError, match="No NIfTI files found"):
                self.piece.piece_function(input_data)
