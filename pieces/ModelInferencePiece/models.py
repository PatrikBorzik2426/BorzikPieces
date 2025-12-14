from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict
from enum import Enum


class ModelArchitecture(str, Enum):
    """Supported model architectures"""
    UNET = "unet"
    SWIN_UNETR = "swin_unetr"


class SubjectInfo(BaseModel):
    """Information about a single subject for inference"""
    subject_id: str
    image_path: str
    mask_path: Optional[str] = None


class InferenceMetrics(BaseModel):
    """Metrics for a single inference sample"""
    subject_id: str
    dice_score: float
    mean_confidence: float
    max_confidence: float
    min_confidence: float
    class_confidences: Dict[str, float]


class InputModel(BaseModel):
    """
    Model Inference Piece Input Model
    """
    model_config = ConfigDict(protected_namespaces=())
    
    # Model configuration
    model_path: str = Field(
        description="Path to the trained model (.pth file)"
    )
    model_architecture: ModelArchitecture = Field(
        description="Model architecture used during training",
        default=ModelArchitecture.UNET
    )
    num_classes: int = Field(
        description="Number of segmentation classes (including background)",
        default=6,
        ge=2
    )
    patch_size: int = Field(
        description="Size of 3D patches used during training",
        default=64,
        ge=32,
        le=128
    )
    
    # Input data - can come from upstream piece or file paths
    subjects: Optional[List[SubjectInfo]] = Field(
        description="List of subjects from upstream piece. If provided, image_paths and mask_paths are ignored.",
        default=None
    )
    image_paths: Optional[List[str]] = Field(
        description="List of paths to NIfTI image files for inference",
        default=None
    )
    mask_paths: Optional[List[str]] = Field(
        description="Optional list of paths to ground truth masks for metric calculation",
        default=None
    )
    
    # Inference configuration
    num_samples: int = Field(
        description="Number of samples to run inference on (0 = all)",
        default=5,
        ge=0,
        le=100
    )
    samples_per_volume: int = Field(
        description="Number of patches to extract per volume",
        default=5,
        ge=1,
        le=50
    )
    
    # Output configuration
    output_dir: str = Field(
        description="Directory to save inference results and visualizations",
        default="/home/shared_storage/inference_results"
    )
    save_predictions: bool = Field(
        description="Whether to save prediction masks as NIfTI files",
        default=False
    )
    save_visualizations: bool = Field(
        description="Whether to save visualization images",
        default=True
    )
    
    # Device configuration
    use_gpu: bool = Field(
        description="Whether to use GPU if available",
        default=True
    )
    batch_size: int = Field(
        description="Batch size for inference",
        default=4,
        ge=1,
        le=32
    )


class OutputModel(BaseModel):
    """
    Model Inference Piece Output Model
    """
    output_dir: str = Field(
        description="Directory containing all inference results"
    )
    num_samples_processed: int = Field(
        description="Total number of samples processed"
    )
    mean_dice_score: Optional[float] = Field(
        description="Average Dice score across all samples (if masks provided)",
        default=None
    )
    mean_confidence: float = Field(
        description="Average confidence score across all samples"
    )
    inference_metrics: List[InferenceMetrics] = Field(
        description="Detailed metrics for each sample"
    )
    visualization_dir: Optional[str] = Field(
        description="Directory containing visualization images",
        default=None
    )
    predictions_dir: Optional[str] = Field(
        description="Directory containing saved prediction masks",
        default=None
    )
    summary_report: str = Field(
        description="Text summary of inference results"
    )
