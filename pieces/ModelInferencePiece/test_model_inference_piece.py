import pytest
import sys
import os
import tempfile

# Add pieces directory to path
pieces_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if pieces_dir not in sys.path:
    sys.path.insert(0, pieces_dir)

# Import only models to avoid heavy dependencies
from ModelInferencePiece.models import InputModel, OutputModel, InferenceMetrics, ModelArchitecture


def test_model_inference_input_model():
    """Test ModelInferencePiece input model validation"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Mock model path
        model_path = os.path.join(tmpdir, "model.pth")
        
        # Test with image paths
        input_data = InputModel(
            model_path=model_path,
            model_architecture=ModelArchitecture.UNET,
            num_classes=6,
            patch_size=64,
            image_paths=["/path/to/img1.nii.gz", "/path/to/img2.nii.gz"],
            mask_paths=["/path/to/mask1.nii.gz", "/path/to/mask2.nii.gz"],
            num_samples=2,
            output_dir=tmpdir,
            use_gpu=False
        )
        
        # Verify input model
        assert input_data.model_architecture == ModelArchitecture.UNET
        assert input_data.num_classes == 6
        assert input_data.patch_size == 64
        assert len(input_data.image_paths) == 2
        assert len(input_data.mask_paths) == 2
        assert input_data.num_samples == 2
        assert input_data.save_visualizations is True
        assert input_data.save_predictions is False


def test_inference_metrics_model():
    """Test InferenceMetrics model"""
    metrics = InferenceMetrics(
        subject_id="sub-001",
        dice_score=0.85,
        mean_confidence=0.92,
        max_confidence=0.99,
        min_confidence=0.65,
        class_confidences={"class_0": 0.91, "class_1": 0.93}
    )
    
    assert metrics.subject_id == "sub-001"
    assert metrics.dice_score == 0.85
    assert metrics.mean_confidence == 0.92
    assert len(metrics.class_confidences) == 2


def test_output_model():
    """Test OutputModel"""
    metrics = [
        InferenceMetrics(
            subject_id="sub-001",
            dice_score=0.85,
            mean_confidence=0.92,
            max_confidence=0.99,
            min_confidence=0.65,
            class_confidences={"class_0": 0.91}
        )
    ]
    
    output = OutputModel(
        output_dir="/tmp/results",
        num_samples_processed=1,
        mean_dice_score=0.85,
        mean_confidence=0.92,
        inference_metrics=metrics,
        summary_report="Test summary"
    )
    
    assert output.num_samples_processed == 1
    assert output.mean_dice_score == 0.85
    assert len(output.inference_metrics) == 1


if __name__ == "__main__":
    test_model_inference_input_model()
    test_inference_metrics_model()
    test_output_model()
    print("ModelInferencePiece tests passed!")
