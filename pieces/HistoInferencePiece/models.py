from pydantic import BaseModel, Field
from typing import List, Optional


# MoNuSeg 2-class mapping (default — matches the bundled dataset in domino_data/histo_data/)
# For the 5-class beetle/pituitary dataset use:
#   {"0":[0,0,0],"1":[128,128,128],"2":[0,255,0],"3":[255,0,0],"4":[0,0,255]}
DEFAULT_CLASS_MAPPING = '{"0": [0,0,0], "1": [0,255,0]}'
DEFAULT_CLASS_NAMES = ["background", "nucleus"]


class SampleInfo(BaseModel):
    name: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    test_samples: List[SampleInfo] = Field(
        description="Test set from HistoDataSplitPiece (masks optional — Dice computed if present)"
    )
    model_path: str = Field(
        description="Connect best_model_path from HistoTrainingPiece",
        default="/home/shared_storage/histo_models/checkpoints/best_model.pth"
    )
    output_dir: str = Field(
        description="Directory to save prediction masks and comparison visualizations",
        default="/home/shared_storage/histo_inference"
    )
    class_mapping_json: str = Field(
        description="Connect class_mapping_json from HistoTrainingPiece",
        default=DEFAULT_CLASS_MAPPING
    )
    class_names: List[str] = Field(
        description="Connect class_names from HistoTrainingPiece or re-enter",
        default=DEFAULT_CLASS_NAMES
    )
    model_architecture: str = Field(
        description="Connect model_architecture from HistoTrainingPiece",
        default="unet"
    )
    encoder_name: str = Field(
        description="Connect encoder_name from HistoTrainingPiece",
        default="resnet34"
    )
    num_classes: int = Field(
        description="Connect num_classes from HistoTrainingPiece",
        default=5,
        ge=2
    )
    image_height: int = Field(
        description="Connect image_height from HistoTrainingPiece",
        default=256,
        ge=64,
        le=2048
    )
    image_width: int = Field(
        description="Connect image_width from HistoTrainingPiece",
        default=256,
        ge=64,
        le=2048
    )
    save_comparison_images: bool = Field(
        description="Save side-by-side comparison images (input | prediction | ground truth if available)",
        default=True
    )
    max_comparison_images: int = Field(
        description="Maximum number of comparison images to save",
        default=20,
        ge=1,
        le=500
    )
    use_gpu: bool = Field(
        description="Use GPU if available",
        default=True
    )


class OutputModel(BaseModel):
    predictions_dir: str = Field(description="Directory containing RGB prediction mask files")
    visualization_dir: str = Field(description="Directory containing comparison figures")
    num_images_processed: int = Field(description="Number of images inference was run on")
    mean_dice: Optional[float] = Field(
        description="Mean Dice score (only set when test_samples include mask_path)",
        default=None
    )
    dice_per_class: Optional[List[float]] = Field(
        description="Per-class Dice scores (only set when masks available)",
        default=None
    )
    inference_summary: str = Field(description="Human-readable inference summary")
