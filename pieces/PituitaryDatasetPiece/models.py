from pydantic import BaseModel, Field
from typing import Optional, List


class SubjectInfo(BaseModel):
    """Information about a single subject/scan"""
    subject_id: str
    image_path: str
    mask_path: Optional[str] = None


class PreprocessedSubject(BaseModel):
    """Information about a preprocessed subject"""
    subject_id: str
    original_image_path: str
    preprocessed_image_path: str
    preprocessed_mask_path: Optional[str] = None
    original_shape: List[int]
    preprocessed_shape: List[int]
    image_stats: dict


class InputModel(BaseModel):
    """
    Pituitary Dataset Piece Input Model
    """
    train_subjects: Optional[List[PreprocessedSubject]] = Field(
        description="Preprocessed training subjects",
        default=None
    )
    val_subjects: Optional[List[PreprocessedSubject]] = Field(
        description="Preprocessed validation subjects",
        default=None
    )
    test_subjects: Optional[List[PreprocessedSubject]] = Field(
        description="Preprocessed test subjects",
        default=None
    )
    # Alternative: provide directories directly
    data_dir: Optional[str] = Field(
        description="Directory containing preprocessed 'images' and 'masks' subdirectories",
        default=None
    )
    batch_size: int = Field(
        description="Batch size for DataLoader",
        default=2,
        ge=1
    )
    num_workers: int = Field(
        description="Number of workers for DataLoader",
        default=0,
        ge=0
    )
    shuffle_train: bool = Field(
        description="Whether to shuffle training data",
        default=True
    )


class DatasetInfo(BaseModel):
    """Information about a created dataset"""
    split_name: str
    num_samples: int
    batch_size: int
    num_batches: int
    subject_ids: List[str]


class OutputModel(BaseModel):
    """
    Pituitary Dataset Piece Output Model
    """
    train_info: Optional[DatasetInfo] = Field(
        description="Information about training dataset"
    )
    val_info: Optional[DatasetInfo] = Field(
        description="Information about validation dataset"
    )
    test_info: Optional[DatasetInfo] = Field(
        description="Information about test dataset"
    )
    data_dir: str = Field(
        description="Directory containing the data"
    )
    dataset_config_path: str = Field(
        description="Path to saved dataset configuration JSON file"
    )
