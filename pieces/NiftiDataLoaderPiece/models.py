from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class InputModel(BaseModel):
    """
    NIfTI Data Loader Piece Input Model
    """
    images_path: str = Field(
        description="Path to the directory containing NIfTI image files (.nii.gz)",
        default="/data/images"
    )
    masks_path: Optional[str] = Field(
        description="Path to the directory containing NIfTI mask/segmentation files (.nii.gz). Optional for inference-only mode.",
        default=None
    )
    file_pattern: str = Field(
        description="Glob pattern to match NIfTI files (e.g., 'sub-*.nii.gz')",
        default="*.nii.gz"
    )


class SubjectInfo(BaseModel):
    """Information about a single subject/scan"""
    subject_id: str
    image_path: str
    mask_path: Optional[str] = None


class OutputModel(BaseModel):
    """
    NIfTI Data Loader Piece Output Model
    """
    subjects: List[SubjectInfo] = Field(
        description="List of discovered subjects with their image and mask paths"
    )
    num_subjects: int = Field(
        description="Total number of subjects found"
    )
    images_dir: str = Field(
        description="Path to images directory"
    )
    masks_dir: Optional[str] = Field(
        description="Path to masks directory (if provided)"
    )
