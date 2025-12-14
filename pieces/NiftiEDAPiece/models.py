from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
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
        description="[DEPRECATED - ignored] Directory to save EDA visualizations and reports. Files are now automatically saved to Domino's tracked storage.",
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


class AnalysisText(BaseModel):
    """Text description of analysis findings in Slovak (matching research notebook style)"""
    section: str = Field(description="Section name/title")
    finding: str = Field(description="Detailed Slovak text describing the finding and its implications")


class EDAStatistics(BaseModel):
    """Summary statistics from comprehensive EDA"""
    total_subjects: int
    analyzed_subjects: int
    unique_shapes: int = Field(description="Number of unique volume shapes found")
    unique_spacings: int = Field(description="Number of unique voxel spacing protocols")
    num_classes: int = Field(description="Number of unique classes in masks")
    class_distribution: Dict[str, int] = Field(description="Voxel count per class")
    mean_lesion_coverage: float = Field(description="Mean ratio of lesion voxels to total voxels")
    slice_lesion_ratio: float = Field(description="Ratio of slices containing lesions")
    mean_intensity: float = Field(description="Mean intensity across all volumes")
    std_intensity: float = Field(description="Mean standard deviation across all volumes")


class OutputModel(BaseModel):
    """
    NIfTI EDA Piece Output Model
    """
    statistics: EDAStatistics = Field(
        description="Summary statistics from the comprehensive analysis"
    )
    report_path: str = Field(
        description="Path to the generated EDA text report"
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
    analysis_texts: List[Dict[str, Any]] = Field(
        description="Detailed Slovak text descriptions for each analysis section",
        default=[]
    )
