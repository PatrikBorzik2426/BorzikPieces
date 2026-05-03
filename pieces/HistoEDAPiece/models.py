from pydantic import BaseModel, Field
from typing import List


DEFAULT_CLASS_MAPPING = '{"0": [0,0,0], "1": [128,128,128], "2": [0,255,0], "3": [255,0,0], "4": [0,0,255]}'
DEFAULT_CLASS_NAMES = ["unannotated", "other", "non-invasive", "invasive", "necrosis"]


class InputModel(BaseModel):
    images_path: str = Field(
        description="Path to images directory (connect images_path from HistoDataLoaderPiece)",
        default="/home/shared_storage/histo_data/images"
    )
    masks_path: str = Field(
        description="Path to masks directory (connect masks_path from HistoDataLoaderPiece)",
        default="/home/shared_storage/histo_data/masks"
    )
    class_mapping_json: str = Field(
        description=(
            'JSON string mapping class_id (str) to RGB list. '
            'Default 5-class beetle: {"0":[0,0,0],"1":[128,128,128],"2":[0,255,0],"3":[255,0,0],"4":[0,0,255]}'
        ),
        default=DEFAULT_CLASS_MAPPING
    )
    class_names: List[str] = Field(
        description="Class names ordered by class_id",
        default=DEFAULT_CLASS_NAMES
    )
    max_eda_images: int = Field(
        description="Max images to scan for pixel-level statistics",
        default=100,
        ge=10,
        le=5000
    )
    gallery_samples: int = Field(
        description="Number of image/mask pairs in the visual gallery",
        default=8,
        ge=1,
        le=32
    )
    file_extension: str = Field(
        description="File extension to scan",
        default=".png"
    )


class OutputModel(BaseModel):
    eda_summary: str = Field(
        description="Text summary of EDA findings"
    )
    report_path: str = Field(
        description="Path to the HTML EDA report"
    )
