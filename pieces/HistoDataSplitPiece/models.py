from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum


class SplitStrategy(str, Enum):
    RANDOM = "random"
    SEQUENTIAL = "sequential"


class SampleInfo(BaseModel):
    name: str
    image_path: str
    mask_path: Optional[str] = None


class InputModel(BaseModel):
    samples: List[SampleInfo] = Field(
        description="List of image/mask pairs from HistoDataLoaderPiece or HistoPatchExtractorPiece"
    )
    train_ratio: float = Field(
        description="Fraction of samples assigned to the training set",
        default=0.70,
        ge=0.0,
        le=1.0
    )
    val_ratio: float = Field(
        description="Fraction of samples assigned to the validation set",
        default=0.15,
        ge=0.0,
        le=1.0
    )
    test_ratio: float = Field(
        description="Fraction of samples assigned to the test set (remainder after train+val if ratios don't sum to 1.0)",
        default=0.15,
        ge=0.0,
        le=1.0
    )
    random_seed: int = Field(
        description="Random seed for reproducible splits",
        default=42
    )
    split_strategy: SplitStrategy = Field(
        description="'random' shuffles samples before splitting; 'sequential' preserves filesystem order",
        default=SplitStrategy.RANDOM
    )


class OutputModel(BaseModel):
    train_samples: List[SampleInfo] = Field(
        description="Training set — connect to HistoTrainingPiece"
    )
    val_samples: List[SampleInfo] = Field(
        description="Validation set — connect to HistoTrainingPiece and HistoValidationPiece"
    )
    test_samples: List[SampleInfo] = Field(
        description="Test set — connect to HistoInferencePiece"
    )
    train_count: int
    val_count: int
    test_count: int
    total_count: int
    split_info: dict = Field(
        description="Split metadata (counts, ratios, seed, strategy)"
    )
