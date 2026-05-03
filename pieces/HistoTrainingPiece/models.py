from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


# MoNuSeg 2-class mapping (default — matches the bundled dataset in domino_data/histo_data/)
# For the 5-class beetle/pituitary dataset use:
#   {"0":[0,0,0],"1":[128,128,128],"2":[0,255,0],"3":[255,0,0],"4":[0,0,255]}
DEFAULT_CLASS_MAPPING = '{"0": [0,0,0], "1": [0,255,0]}'
DEFAULT_CLASS_NAMES = ["background", "nucleus"]


class ModelArchitecture(str, Enum):
    UNET = "unet"
    UNET_PLUS_PLUS = "unetplusplus"
    FPN = "fpn"
    DEEPLABV3_PLUS = "deeplabv3plus"


class SampleInfo(BaseModel):
    name: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    train_samples: List[SampleInfo] = Field(
        description="Training set from HistoDataSplitPiece"
    )
    val_samples: List[SampleInfo] = Field(
        description="Validation set from HistoDataSplitPiece (used for per-epoch Dice tracking)"
    )
    output_dir: str = Field(
        description="Directory to save trained models, checkpoints, and training logs",
        default="/home/shared_storage/histo_models"
    )
    class_mapping_json: str = Field(
        description=(
            'JSON mapping class_id (str) to RGB list. '
            'MoNuSeg 2-class: {"0":[0,0,0],"1":[0,255,0]} | '
            '5-class beetle: {"0":[0,0,0],"1":[128,128,128],"2":[0,255,0],"3":[255,0,0],"4":[0,0,255]}'
        ),
        default=DEFAULT_CLASS_MAPPING
    )
    class_names: List[str] = Field(
        description="Class names ordered by class_id",
        default=DEFAULT_CLASS_NAMES
    )
    model_architecture: ModelArchitecture = Field(
        description="2D segmentation architecture (segmentation-models-pytorch)",
        default=ModelArchitecture.UNET
    )
    encoder_name: str = Field(
        description="Backbone encoder (e.g. resnet34, resnet50, efficientnet-b3)",
        default="resnet34"
    )
    encoder_weights: str = Field(
        description="Pretrained encoder weights. Use 'imagenet' or 'none'.",
        default="imagenet"
    )
    image_height: int = Field(
        description="Height to resize images/masks before training. Match your patch_size — default 256 for MoNuSeg patches, 512 for beetle/pituitary.",
        default=256,
        ge=64,
        le=2048
    )
    image_width: int = Field(
        description="Width to resize images/masks before training. Match your patch_size — default 256 for MoNuSeg patches, 512 for beetle/pituitary.",
        default=256,
        ge=64,
        le=2048
    )
    num_epochs: int = Field(
        description="Number of training epochs",
        default=100,
        ge=1,
        le=500
    )
    batch_size: int = Field(
        description="Training batch size",
        default=16,
        ge=1,
        le=64
    )
    learning_rate: float = Field(
        description="Initial learning rate",
        default=1e-3,
        gt=0.0,
        lt=1.0
    )
    use_augmentation: bool = Field(
        description="Apply training-time augmentation (flips, brightness/contrast, hue/saturation)",
        default=True
    )
    lr_scheduler_patience: int = Field(
        description="Patience for ReduceLROnPlateau",
        default=10,
        ge=1
    )
    early_stopping_patience: int = Field(
        description="Epochs without improvement before stopping (0 = disabled)",
        default=20,
        ge=0
    )
    save_checkpoint_interval: int = Field(
        description="Save a numbered checkpoint every N epochs",
        default=5,
        ge=1
    )
    max_saved_val_images: int = Field(
        description="Max prediction images saved per validation epoch",
        default=10,
        ge=0
    )
    num_workers: int = Field(
        description="DataLoader workers (keep 0 inside Docker to avoid shared-memory issues)",
        default=0,
        ge=0
    )
    random_seed: int = Field(
        description="Random seed for reproducibility",
        default=42
    )
    use_gpu: bool = Field(
        description="Use GPU if available",
        default=True
    )
    dry_run: bool = Field(
        description="Quick smoke-test: forces 1 epoch and 4 samples so the full pipeline can be validated without real training",
        default=False
    )


class EpochMetrics(BaseModel):
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    mean_dice: Optional[float] = None
    dice_per_class: Optional[List[float]] = None
    learning_rate: float


class OutputModel(BaseModel):
    model_path: str = Field(description="Path to the final model .pth file")
    checkpoint_dir: str = Field(description="Directory containing all epoch checkpoints")
    best_model_path: str = Field(description="Path to the best model (highest val mean Dice)")
    best_val_dice: float = Field(description="Best mean Dice score achieved")
    best_epoch: int = Field(description="Epoch of the best model")
    final_train_loss: float = Field(description="Training loss at the last epoch")
    total_epochs_trained: int = Field(description="Number of epochs completed")
    training_summary: str = Field(description="Human-readable training summary")
    # pass-through fields for HistoValidationPiece and HistoInferencePiece
    num_classes: int
    class_mapping_json: str
    model_architecture: str
    encoder_name: str
    image_height: int
    image_width: int
