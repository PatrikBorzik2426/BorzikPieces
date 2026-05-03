from pydantic import BaseModel, Field
from typing import List, Optional


class SampleInfo(BaseModel):
    """A single image/mask pair in the histopathology dataset."""
    name: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    images_path: str = Field(
        description="Absolute path to the directory containing histopathology images (PNG/JPG/TIF)",
        default="/home/shared_storage/histo_data/images"
    )
    masks_path: str = Field(
        description="Absolute path to the directory containing RGB segmentation masks. Masks must share the same filename as their corresponding image.",
        default="/home/shared_storage/histo_data/masks"
    )
    file_extension: str = Field(
        description="File extension to scan for (e.g. .png, .jpg, .tif)",
        default=".png"
    )


class OutputModel(BaseModel):
    images_path: str = Field(
        description="Validated path to images directory (passed through for HistoEDAPiece)"
    )
    masks_path: str = Field(
        description="Validated path to masks directory (passed through for HistoEDAPiece)"
    )
    samples: List[SampleInfo] = Field(
        description="List of matched image/mask pairs — connect to HistoPatchExtractorPiece or HistoDataSplitPiece"
    )
    num_samples: int = Field(
        description="Total number of matched pairs"
    )
    data_summary: str = Field(
        description="Human-readable dataset summary"
    )
