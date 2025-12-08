from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, DatasetInfo
import os
import json
import base64
import traceback
import numpy as np


class PituitaryDatasetPiece(BasePiece):
    """
    A piece that creates PyTorch-compatible dataset configurations for 
    pituitary gland segmentation.
    
    This piece prepares dataset metadata and configurations that can be used
    to create PyTorch DataLoaders for training medical image segmentation models.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            batch_size = input_data.batch_size
            num_workers = input_data.num_workers
            shuffle_train = input_data.shuffle_train
            
            self.logger.info("Creating dataset configuration for pituitary segmentation")
            
            train_info = None
            val_info = None
            test_info = None
            data_dir = input_data.data_dir or "/data/preprocessed"
            
            # Process training subjects
            if input_data.train_subjects:
                train_subjects = input_data.train_subjects
                num_batches = (len(train_subjects) + batch_size - 1) // batch_size
                train_info = DatasetInfo(
                    split_name="train",
                    num_samples=len(train_subjects),
                    batch_size=batch_size,
                    num_batches=num_batches,
                    subject_ids=[s.subject_id for s in train_subjects]
                )
                self.logger.info(f"Training set: {len(train_subjects)} samples, {num_batches} batches")
            
            # Process validation subjects
            if input_data.val_subjects:
                val_subjects = input_data.val_subjects
                num_batches = (len(val_subjects) + batch_size - 1) // batch_size
                val_info = DatasetInfo(
                    split_name="val",
                    num_samples=len(val_subjects),
                    batch_size=batch_size,
                    num_batches=num_batches,
                    subject_ids=[s.subject_id for s in val_subjects]
                )
                self.logger.info(f"Validation set: {len(val_subjects)} samples, {num_batches} batches")
            
            # Process test subjects
            if input_data.test_subjects:
                test_subjects = input_data.test_subjects
                num_batches = (len(test_subjects) + batch_size - 1) // batch_size
                test_info = DatasetInfo(
                    split_name="test",
                    num_samples=len(test_subjects),
                    batch_size=batch_size,
                    num_batches=num_batches,
                    subject_ids=[s.subject_id for s in test_subjects]
                )
                self.logger.info(f"Test set: {len(test_subjects)} samples, {num_batches} batches")
            
            # Create dataset configuration
            config = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "shuffle_train": shuffle_train,
                "data_dir": data_dir,
                "train": {
                    "subjects": [s.model_dump() for s in input_data.train_subjects] if input_data.train_subjects else [],
                    "num_samples": len(input_data.train_subjects) if input_data.train_subjects else 0
                },
                "val": {
                    "subjects": [s.model_dump() for s in input_data.val_subjects] if input_data.val_subjects else [],
                    "num_samples": len(input_data.val_subjects) if input_data.val_subjects else 0
                },
                "test": {
                    "subjects": [s.model_dump() for s in input_data.test_subjects] if input_data.test_subjects else [],
                    "num_samples": len(input_data.test_subjects) if input_data.test_subjects else 0
                },
                "pytorch_dataset_class": "PituitarySegmentationDataset",
                "pytorch_code": self._get_dataset_code()
            }
            
            # Save configuration
            config_dir = getattr(self, "results_path", "/tmp")
            config_path = os.path.join(config_dir, "dataset_config.json")
            os.makedirs(config_dir, exist_ok=True)
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Dataset configuration saved to {config_path}")
            
            # Set display result
            summary = {
                "train_samples": train_info.num_samples if train_info else 0,
                "val_samples": val_info.num_samples if val_info else 0,
                "test_samples": test_info.num_samples if test_info else 0,
                "batch_size": batch_size,
                "config_path": config_path
            }
            summary_text = json.dumps(summary, indent=2)
            base64_content = base64.b64encode(summary_text.encode("utf-8")).decode("utf-8")
            self.display_result = {
                "file_type": "json",
                "base64_content": base64_content
            }
            
            return OutputModel(
                train_info=train_info,
                val_info=val_info,
                test_info=test_info,
                data_dir=data_dir,
                dataset_config_path=config_path
            )
            
        except Exception as e:
            self.logger.error(f"Error in PituitaryDatasetPiece: {e}")
            print("[PituitaryDatasetPiece] Exception in piece_function:")
            traceback.print_exc()
            raise
    
    def _get_dataset_code(self) -> str:
        """Return the PyTorch Dataset class code as a string"""
        return '''
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PituitarySegmentationDataset(Dataset):
    """
    PyTorch Dataset for Pituitary Gland Segmentation.
    
    Loads preprocessed numpy arrays (images and masks) for 3D medical image segmentation.
    """
    
    def __init__(self, subjects, transform=None):
        """
        Args:
            subjects: List of subject dictionaries with 'subject_id', 
                     'preprocessed_image_path', 'preprocessed_mask_path'
            transform: Optional transform to apply to image and mask
        """
        self.subjects = subjects
        self.transform = transform
    
    def __len__(self):
        return len(self.subjects)
    
    def __getitem__(self, idx):
        subject = self.subjects[idx]
        subject_id = subject["subject_id"]
        
        # Load image
        img_path = subject["preprocessed_image_path"]
        if img_path.endswith(".npy"):
            img = np.load(img_path).astype(np.float32)
        else:
            import nibabel as nib
            img = nib.load(img_path).get_fdata().astype(np.float32)
        
        # Load mask if available
        mask_path = subject.get("preprocessed_mask_path")
        if mask_path and os.path.exists(mask_path):
            if mask_path.endswith(".npy"):
                msk = np.load(mask_path).astype(np.int64)
            else:
                import nibabel as nib
                msk = nib.load(mask_path).get_fdata().astype(np.int64)
        else:
            msk = np.zeros_like(img, dtype=np.int64)
        
        # Apply transforms
        if self.transform:
            img, msk = self.transform(img, msk)
        
        # Convert to tensors: [C, D, H, W] format
        img = torch.tensor(img).unsqueeze(0).float()  # [1, D, H, W]
        msk = torch.tensor(msk).long()                # [D, H, W]
        
        return img, msk, subject_id


def create_dataloaders(config_path, batch_size=None, num_workers=None):
    """
    Utility function to create DataLoaders from saved configuration.
    
    Args:
        config_path: Path to dataset_config.json
        batch_size: Override batch size (optional)
        num_workers: Override num_workers (optional)
    
    Returns:
        dict with 'train', 'val', 'test' DataLoaders
    """
    import json
    from torch.utils.data import DataLoader
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    bs = batch_size or config["batch_size"]
    nw = num_workers if num_workers is not None else config["num_workers"]
    
    loaders = {}
    
    if config["train"]["num_samples"] > 0:
        train_ds = PituitarySegmentationDataset(config["train"]["subjects"])
        loaders["train"] = DataLoader(
            train_ds, batch_size=bs, shuffle=config["shuffle_train"], num_workers=nw
        )
    
    if config["val"]["num_samples"] > 0:
        val_ds = PituitarySegmentationDataset(config["val"]["subjects"])
        loaders["val"] = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)
    
    if config["test"]["num_samples"] > 0:
        test_ds = PituitarySegmentationDataset(config["test"]["subjects"])
        loaders["test"] = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=nw)
    
    return loaders
'''
