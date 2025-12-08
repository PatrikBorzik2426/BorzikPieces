import pytest
from pieces.DataSplitPiece.piece import DataSplitPiece
from pieces.DataSplitPiece.models import InputModel, SubjectInfo, SplitStrategy


class TestDataSplitPiece:
    """Tests for DataSplitPiece"""

    def setup_method(self):
        """Set up test fixtures"""
        self.piece = DataSplitPiece()
        
    def create_subjects(self, n):
        """Create n dummy subjects"""
        return [
            SubjectInfo(
                subject_id=f"sub-{i:03d}",
                image_path=f"/data/images/sub-{i:03d}.nii.gz",
                mask_path=f"/data/masks/sub-{i:03d}.nii.gz"
            )
            for i in range(n)
        ]

    def test_basic_split(self):
        """Test basic 70/15/15 split"""
        subjects = self.create_subjects(100)
        
        input_data = InputModel(
            subjects=subjects,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            random_seed=42
        )
        
        output = self.piece.piece_function(input_data)
        
        assert output.train_count == 70
        assert output.val_count == 15
        assert output.test_count == 15
        assert output.total_count == 100

    def test_reproducibility(self):
        """Test that same seed produces same split"""
        subjects = self.create_subjects(50)
        
        input_data1 = InputModel(subjects=subjects, random_seed=123)
        input_data2 = InputModel(subjects=subjects, random_seed=123)
        
        output1 = self.piece.piece_function(input_data1)
        output2 = self.piece.piece_function(input_data2)
        
        assert [s.subject_id for s in output1.train_subjects] == [s.subject_id for s in output2.train_subjects]

    def test_sequential_split(self):
        """Test sequential (non-random) split"""
        subjects = self.create_subjects(10)
        
        input_data = InputModel(
            subjects=subjects,
            train_ratio=0.5,
            val_ratio=0.3,
            test_ratio=0.2,
            split_strategy=SplitStrategy.SEQUENTIAL
        )
        
        output = self.piece.piece_function(input_data)
        
        # First 5 should be train
        assert output.train_subjects[0].subject_id == "sub-000"
        assert output.train_count == 5

    def test_small_dataset(self):
        """Test with small dataset"""
        subjects = self.create_subjects(5)
        
        input_data = InputModel(
            subjects=subjects,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        output = self.piece.piece_function(input_data)
        
        assert output.total_count == 5
        assert output.train_count + output.val_count + output.test_count == 5
