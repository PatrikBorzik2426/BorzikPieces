from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class InputModel(BaseModel):
    """
    NIfTI Visualization Piece Input Model
    
    Standalone piece that loads NIfTI files directly from directories
    and creates a grid visualization. No upstream connection required.
    """
    images_path: str = Field(
        description="Path to directory containing NIfTI image files (.nii.gz)",
        default="/home/shared_storage/medical_data/images"
    )
    masks_path: Optional[str] = Field(
        description="Path to directory containing NIfTI mask files (.nii.gz). Optional.",
        default="/home/shared_storage/medical_data/masks"
    )
    file_pattern: str = Field(
        description="Glob pattern to match NIfTI files (e.g., '*.nii.gz')",
        default="*.nii.gz"
    )
    max_subjects: int = Field(
        description="Maximum number of subjects to visualize",
        default=6,
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
    grid_columns: int = Field(
        description="Number of columns in visualization grid",
        default=3,
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
