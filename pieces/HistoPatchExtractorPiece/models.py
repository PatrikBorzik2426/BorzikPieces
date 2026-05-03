from pydantic import BaseModel, Field
from typing import List, Optional


class SampleInfo(BaseModel):
    name: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    samples: List[SampleInfo] = Field(
        description="List of image/mask pairs from HistoDataLoaderPiece"
    )
    output_dir: str = Field(
        description="Directory where extracted patches will be saved (patches/images and patches/masks subdirs are created automatically)",
        default="/home/shared_storage/histo_patches"
    )
    patch_size: int = Field(
        description="Side length (pixels) of each square patch extracted from the image",
        default=512,
        ge=64,
        le=2048
    )
    stride: int = Field(
        description="Step size (pixels) between adjacent patches. Use stride < patch_size for overlap. If 0, defaults to patch_size (no overlap).",
        default=0,
        ge=0
    )
    min_foreground_ratio: float = Field(
        description="Minimum fraction of non-background pixels in the MASK patch required to keep it (0.0 = keep all patches, 0.05 = skip mostly-background patches).",
        default=0.05,
        ge=0.0,
        le=1.0
    )
    background_class_id: int = Field(
        description="Class ID that counts as background for the foreground ratio filter",
        default=0
    )
    background_rgb: List[int] = Field(
        description="RGB colour of the background class in the mask (used to detect foreground pixels)",
        default=[0, 0, 0]
    )


class OutputModel(BaseModel):
    samples: List[SampleInfo] = Field(
        description="SampleInfo list for extracted patches — connect to HistoDataSplitPiece"
    )
    num_patches: int = Field(
        description="Total number of patches extracted (after foreground filtering)"
    )
    patches_images_path: str = Field(
        description="Directory containing patch image files"
    )
    patches_masks_path: str = Field(
        description="Directory containing patch mask files"
    )
    extraction_summary: str = Field(
        description="Human-readable extraction summary"
    )
