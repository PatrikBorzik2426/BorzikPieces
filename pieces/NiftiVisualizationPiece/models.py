from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any, Union


class SubjectInfo(BaseModel):
    """Information about a single subject/scan"""
    subject_id: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    """
    NIfTI Visualization Piece Input Model
    
    NOTE: This piece must be connected to an upstream piece (DataLoader or DataSplit)
    that outputs List[SubjectInfo]. It visualizes first 10 subjects in a grid.
    """
    # Accept either List[SubjectInfo] or List[dict] to handle Domino's serialization
    subjects: List[Any] = Field(
        description="List of subjects with image and mask paths (from DataLoader or DataSplit). Must be connected in workflow.",
        default=[]
    )
    max_subjects: int = Field(
        description="Maximum number of subjects to visualize",
        default=10,
        ge=1,
        le=20
    )
    slice_index: Optional[int] = Field(
        description="Slice index to visualize. If None, uses middle slice.",
        default=None
    )
    view_plane: Literal["axial", "sagittal", "coronal"] = Field(
        description="Anatomical plane to visualize",
        default="axial"
    )
    show_mask_overlay: bool = Field(
        description="Whether to overlay mask on image",
        default=True
    )
    mask_alpha: float = Field(
        description="Transparency of mask overlay (0.0 = transparent, 1.0 = opaque)",
        default=0.5,
        ge=0.0,
        le=1.0
    )
    color_map: str = Field(
        description="Matplotlib colormap for image (e.g., 'gray', 'viridis', 'bone')",
        default="gray"
    )
    mask_color: str = Field(
        description="Color for mask overlay (e.g., 'red', 'green', 'yellow')",
        default="red"
    )
    grid_columns: int = Field(
        description="Number of columns in visualization grid",
        default=5,
        ge=1,
        le=10
    )


class OutputModel(BaseModel):
    """
    NIfTI Visualization Piece Output Model
    """
    num_subjects: int = Field(
        description="Number of subjects visualized"
    )
    subject_ids: List[str] = Field(
        description="List of subject IDs that were visualized"
    )
    view_plane: str = Field(
        description="Anatomical plane that was visualized"
    )
    grid_size: str = Field(
        description="Grid size (rows x columns)"
    )
    visualization_summary: str = Field(
        description="Summary of visualization"
    )
