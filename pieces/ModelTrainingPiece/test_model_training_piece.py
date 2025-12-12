import pytest
import sys
import os
import json
import tempfile

# Add pieces directory to path
pieces_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pieces_dir not in sys.path:
    sys.path.insert(0, pieces_dir)

from ModelTrainingPiece.models import InputModel, ModelArchitecture


def test_model_training_piece():
    """Test ModelTrainingPiece with minimal configuration"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock dataset config
        dataset_config = {
            "train": {
                "subjects": [
                    {"subject_id": "sub-001", "image_path": "/path/to/img1.nii.gz"},
                    {"subject_id": "sub-002", "image_path": "/path/to/img2.nii.gz"}
                ]
            },
            "val": {
                "subjects": [
                    {"subject_id": "sub-003", "image_path": "/path/to/img3.nii.gz"}
                ]
            }
        }
        
        config_path = os.path.join(tmpdir, "dataset_config.json")
        with open(config_path, 'w') as f:
            json.dump(dataset_config, f)
        
        input_data = InputModel(
            dataset_config_path=config_path,
            data_root="/home/shared_storage/medical_data",
            output_dir=tmpdir,
            model_architecture=ModelArchitecture.UNET,
            num_classes=6,
            epochs=2,  # Minimal for testing
            batch_size=2,
            learning_rate=1e-4,
            patch_size=64,
            samples_per_volume=5,
            use_gpu=False  # Use CPU for testing
        )
        
        # Verify input model
        assert input_data.model_architecture == ModelArchitecture.UNET
        assert input_data.epochs == 2
        assert input_data.batch_size == 2
        assert input_data.patch_size == 64
        assert input_data.num_classes == 6
        assert input_data.use_augmentation is True
        assert input_data.foreground_oversample == 0.9
        
        # Note: Full execution test requires actual NIfTI files and significant compute
        # This test validates the input model structure


if __name__ == "__main__":
    test_model_training_piece()
    print("ModelTrainingPiece tests passed!")
