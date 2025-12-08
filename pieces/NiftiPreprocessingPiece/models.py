from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class NormalizationMethod(str, Enum):
    """Normalization methods for medical images"""
    ZSCORE = "zscore"           # (x - mean) / std
    MINMAX = "minmax"           # Scale to [0, 1]
    PERCENTILE = "percentile"   # Clip to percentiles then scale
    NONE = "none"               # No normalization


class SubjectInfo(BaseModel):
    """Information about a single subject/scan"""
    subject_id: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    """
    NIfTI Preprocessing Piece Input Model
    """
    subjects: List[SubjectInfo] = Field(
        description="List of subjects to preprocess"
    )
    output_dir: str = Field(
        description="Directory to save preprocessed data",
        default="/data/preprocessed"
    )
    normalization: NormalizationMethod = Field(
        description="Normalization method to apply",
        default=NormalizationMethod.ZSCORE
    )
    lower_percentile: float = Field(
        description="Lower percentile for clipping (used with 'percentile' normalization)",
        default=1.0,
        ge=0.0,
        le=50.0
    )
    upper_percentile: float = Field(
        description="Upper percentile for clipping (used with 'percentile' normalization)",
        default=99.0,
        ge=50.0,
        le=100.0
    )
    save_as_numpy: bool = Field(
        description="Save preprocessed data as NumPy arrays (.npy) for faster loading",
        default=True
    )
    target_shape: Optional[List[int]] = Field(
        description="Target shape for resizing (e.g., [128, 128, 64]). None = keep original.",
        default=None
    )


class PreprocessedSubject(BaseModel):
    """Information about a preprocessed subject"""
    subject_id: str
    original_image_path: str
    preprocessed_image_path: str
    preprocessed_mask_path: Optional[str] = None
    original_shape: List[int]
    preprocessed_shape: List[int]
    image_stats: dict


class OutputModel(BaseModel):
    """
    NIfTI Preprocessing Piece Output Model
    """
    preprocessed_subjects: List[PreprocessedSubject] = Field(
        description="List of preprocessed subjects with paths and statistics"
    )
    output_dir: str = Field(
        description="Directory containing preprocessed data"
    )
    num_processed: int = Field(
        description="Number of successfully preprocessed subjects"
    )
    num_failed: int = Field(
        description="Number of subjects that failed preprocessing"
    )
    preprocessing_config: dict = Field(
        description="Configuration used for preprocessing"
    )
