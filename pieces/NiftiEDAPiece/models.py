from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class SubjectInfo(BaseModel):
    """Information about a single subject/scan"""
    subject_id: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    """
    NIfTI EDA Piece Input Model
    """
    subjects: List[SubjectInfo] = Field(
        description="List of subjects to analyze"
    )
    output_dir: str = Field(
        description="Directory to save EDA visualizations and reports",
        default="/home/shared_storage/eda_results"
    )
    max_subjects: int = Field(
        description="Maximum number of subjects to analyze (for performance)",
        default=50,
        ge=1,
        le=200
    )
    generate_3d_plots: bool = Field(
        description="Whether to generate 3D volume visualizations (computationally expensive)",
        default=False
    )
    num_sample_slices: int = Field(
        description="Number of random slices to sample for intensity distribution analysis",
        default=10,
        ge=1,
        le=50
    )


class EDAStatistics(BaseModel):
    """Summary statistics from EDA"""
    total_subjects: int
    analyzed_subjects: int
    volume_shape_mean: List[float]
    volume_shape_std: List[float]
    intensity_mean: float
    intensity_std: float
    mask_coverage_mean: float
    class_distribution: dict


class OutputModel(BaseModel):
    """
    NIfTI EDA Piece Output Model
    """
    statistics: EDAStatistics = Field(
        description="Summary statistics from the analysis"
    )
    report_path: str = Field(
        description="Path to the generated EDA report (HTML or markdown)"
    )
    visualization_dir: str = Field(
        description="Directory containing all generated visualizations"
    )
    num_visualizations: int = Field(
        description="Number of visualization files created"
    )
    analysis_summary: str = Field(
        description="Text summary of key findings"
    )
