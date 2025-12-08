from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class SplitStrategy(str, Enum):
    """Strategy for splitting data"""
    RANDOM = "random"
    SEQUENTIAL = "sequential"


class SubjectInfo(BaseModel):
    """Information about a single subject/scan"""
    subject_id: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    """
    Data Split Piece Input Model
    """
    subjects: List[SubjectInfo] = Field(
        description="List of subjects to split (from NiftiDataLoaderPiece)"
    )
    train_ratio: float = Field(
        description="Proportion of data for training (0.0-1.0)",
        default=0.7,
        ge=0.0,
        le=1.0
    )
    val_ratio: float = Field(
        description="Proportion of data for validation (0.0-1.0)",
        default=0.15,
        ge=0.0,
        le=1.0
    )
    test_ratio: float = Field(
        description="Proportion of data for testing (0.0-1.0). If not specified, uses remaining data.",
        default=0.15,
        ge=0.0,
        le=1.0
    )
    random_seed: Optional[int] = Field(
        description="Random seed for reproducible splits",
        default=42
    )
    split_strategy: SplitStrategy = Field(
        description="Strategy for splitting: 'random' shuffles data, 'sequential' keeps order",
        default=SplitStrategy.RANDOM
    )


class OutputModel(BaseModel):
    """
    Data Split Piece Output Model
    """
    train_subjects: List[SubjectInfo] = Field(
        description="Subjects assigned to training set"
    )
    val_subjects: List[SubjectInfo] = Field(
        description="Subjects assigned to validation set"
    )
    test_subjects: List[SubjectInfo] = Field(
        description="Subjects assigned to test set"
    )
    train_count: int = Field(
        description="Number of training subjects"
    )
    val_count: int = Field(
        description="Number of validation subjects"
    )
    test_count: int = Field(
        description="Number of test subjects"
    )
    total_count: int = Field(
        description="Total number of subjects"
    )
    split_info: dict = Field(
        description="Summary information about the split"
    )
