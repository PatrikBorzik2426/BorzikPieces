import pytest
import os
import json
import tempfile
from pieces.PituitaryDatasetPiece.piece import PituitaryDatasetPiece
from pieces.PituitaryDatasetPiece.models import InputModel, PreprocessedSubject


class TestPituitaryDatasetPiece:
    """Tests for PituitaryDatasetPiece"""

    def setup_method(self):
        """Set up test fixtures"""
        self.piece = PituitaryDatasetPiece()
        # Set a temporary results path
        self.piece.results_path = tempfile.mkdtemp()

    def create_preprocessed_subjects(self, n, prefix="train"):
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

    def test_create_train_dataset(self):
        """Test creating training dataset configuration"""
        train_subjects = self.create_preprocessed_subjects(10, "train")
        
        input_data = InputModel(
            train_subjects=train_subjects,
            batch_size=2
        )
        
        output = self.piece.piece_function(input_data)
        
        assert output.train_info is not None
        assert output.train_info.num_samples == 10
        assert output.train_info.batch_size == 2
        assert output.train_info.num_batches == 5

    def test_create_all_splits(self):
        """Test creating all dataset splits"""
        train = self.create_preprocessed_subjects(10, "train")
        val = self.create_preprocessed_subjects(3, "val")
        test = self.create_preprocessed_subjects(2, "test")
        
        input_data = InputModel(
            train_subjects=train,
            val_subjects=val,
            test_subjects=test,
            batch_size=2
        )
        
        output = self.piece.piece_function(input_data)
        
        assert output.train_info.num_samples == 10
        assert output.val_info.num_samples == 3
        assert output.test_info.num_samples == 2

    def test_config_file_created(self):
        """Test that configuration file is created"""
        train_subjects = self.create_preprocessed_subjects(5, "train")
        
        input_data = InputModel(
            train_subjects=train_subjects,
            batch_size=2
        )
        
        output = self.piece.piece_function(input_data)
        
        assert os.path.exists(output.dataset_config_path)
        
        with open(output.dataset_config_path, 'r') as f:
            config = json.load(f)
        
        assert config["batch_size"] == 2
        assert "pytorch_code" in config
        assert len(config["train"]["subjects"]) == 5
