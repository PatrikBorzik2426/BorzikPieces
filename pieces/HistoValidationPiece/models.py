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
    val_samples: List[SampleInfo] = Field(
        description="Validation set from HistoDataSplitPiece"
    )
    model_path: str = Field(
        description="Path to best_model_path or model_path from HistoTrainingPiece",
        default="/home/shared_storage/histo_models/checkpoints/best_model.pth"
    )
    output_dir: str = Field(
        description="Directory to save validation report and visualizations",
        default="/home/shared_storage/histo_validation"
    )
    class_mapping_json: str = Field(
        description="JSON class mapping — connect class_mapping_json from HistoTrainingPiece",
        default=DEFAULT_CLASS_MAPPING
    )
    class_names: List[str] = Field(
        description="Class names in class_id order — connect class_names from HistoTrainingPiece or re-enter",
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
    max_viz_images: int = Field(
        description="Maximum number of comparison images to save in the report",
        default=16,
        ge=1,
        le=200
    )
    use_gpu: bool = Field(
        description="Use GPU if available",
        default=True
    )


class OutputModel(BaseModel):
    mean_dice: float = Field(description="Mean Dice score across all classes")
    dice_per_class: List[float] = Field(description="Per-class Dice scores")
    mean_iou: float = Field(description="Mean IoU (Jaccard) across all classes")
    iou_per_class: List[float] = Field(description="Per-class IoU scores")
    pixel_accuracy: float = Field(description="Overall pixel accuracy")
    report_path: str = Field(description="Path to the HTML validation report")
    validation_summary: str = Field(description="Human-readable validation summary")
