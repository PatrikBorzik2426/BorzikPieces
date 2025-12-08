from pydantic import BaseModel, Field
from typing import Optional, List, Literal


class SubjectInfo(BaseModel):
    """Information about a single subject/scan"""
    subject_id: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    """
    NIfTI Visualization Piece Input Model
    """
    subject: SubjectInfo = Field(
        description="Subject to visualize"
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
    figure_width: int = Field(
        description="Figure width in pixels",
        default=800,
        ge=400,
        le=2000
    )
    figure_height: int = Field(
        description="Figure height in pixels",
        default=600,
        ge=300,
        le=2000
    )


class OutputModel(BaseModel):
    """
    NIfTI Visualization Piece Output Model
    """
    subject_id: str = Field(
        description="Subject ID that was visualized"
    )
    slice_index: int = Field(
        description="Slice index that was visualized"
    )
    view_plane: str = Field(
        description="Anatomical plane that was visualized"
    )
    image_shape: List[int] = Field(
        description="Shape of the full 3D volume"
    )
    slice_shape: List[int] = Field(
        description="Shape of the 2D slice"
    )
    has_mask: bool = Field(
        description="Whether mask was available and displayed"
    )
    visualization_path: str = Field(
        description="Path to saved visualization PNG"
    )
