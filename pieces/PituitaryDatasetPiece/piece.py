from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, DatasetInfo
import os
import json
import base64
import traceback


class PituitaryDatasetPiece(BasePiece):
    """
    A piece that creates PyTorch-compatible dataset configurations for 
    pituitary gland segmentation.
    
    This piece prepares dataset metadata and configurations that can be used
    to create PyTorch DataLoaders for training medical image segmentation models.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            print("=" * 60)
            print("Starting PituitaryDatasetPiece execution")
            print("=" * 60)
            self.logger.info("=" * 60)
            self.logger.info("Starting PituitaryDatasetPiece execution")
            self.logger.info("=" * 60)
            
            batch_size = input_data.batch_size
            num_workers = input_data.num_workers
            shuffle_train = input_data.shuffle_train
            
            self.logger.info(f"Input configuration:")
            self.logger.info(f"  - Batch size: {batch_size}")
            self.logger.info(f"  - Num workers: {num_workers}")
            self.logger.info(f"  - Shuffle train: {shuffle_train}")
            self.logger.info(f"  - Train subjects: {len(input_data.train_subjects) if input_data.train_subjects else 0}")
            self.logger.info(f"  - Val subjects: {len(input_data.val_subjects) if input_data.val_subjects else 0}")
            self.logger.info(f"  - Test subjects: {len(input_data.test_subjects) if input_data.test_subjects else 0}")
            
            self.logger.info("Creating dataset configuration for pituitary segmentation")
            
            train_info = None
            val_info = None
            test_info = None
            data_dir = input_data.data_dir or "/data/preprocessed"
            
            # Process training subjects
            if input_data.train_subjects:
                train_subjects = input_data.train_subjects
                num_batches = (len(train_subjects) + batch_size - 1) // batch_size
                self.logger.info(f"Processing training set:")
                self.logger.info(f"  - Samples: {len(train_subjects)}")
                self.logger.info(f"  - Batches: {num_batches}")
                self.logger.info(f"  - Samples per batch: ~{len(train_subjects)/num_batches:.1f}")
                train_info = DatasetInfo(
                    split_name="train",
                    num_samples=len(train_subjects),
                    batch_size=batch_size,
                    num_batches=num_batches,
                    subject_ids=[s.subject_id for s in train_subjects]
                )
            
            # Process validation subjects
            if input_data.val_subjects:
                val_subjects = input_data.val_subjects
                num_batches = (len(val_subjects) + batch_size - 1) // batch_size
                self.logger.info(f"Processing validation set:")
                self.logger.info(f"  - Samples: {len(val_subjects)}")
                self.logger.info(f"  - Batches: {num_batches}")
                val_info = DatasetInfo(
                    split_name="val",
                    num_samples=len(val_subjects),
                    batch_size=batch_size,
                    num_batches=num_batches,
                    subject_ids=[s.subject_id for s in val_subjects]
                )
            
            # Process test subjects
            if input_data.test_subjects:
                test_subjects = input_data.test_subjects
                num_batches = (len(test_subjects) + batch_size - 1) // batch_size
                self.logger.info(f"Processing test set:")
                self.logger.info(f"  - Samples: {len(test_subjects)}")
                self.logger.info(f"  - Batches: {num_batches}")
                test_info = DatasetInfo(
                    split_name="test",
                    num_samples=len(test_subjects),
                    batch_size=batch_size,
                    num_batches=num_batches,
                    subject_ids=[s.subject_id for s in test_subjects]
                )
            
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
            
            self.logger.info(f"Saving dataset configuration...")
            self.logger.debug(f"  Config directory: {config_dir}")
            self.logger.debug(f"  Config file: dataset_config.json")
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.info(f"Dataset configuration saved successfully")
            self.logger.info(f"  Path: {config_path}")
            self.logger.info(f"  Size: {os.path.getsize(config_path) / 1024:.2f} KB")
            
            total_samples = (train_info.num_samples if train_info else 0) + \
                           (val_info.num_samples if val_info else 0) + \
                           (test_info.num_samples if test_info else 0)
            
            self.logger.info("Dataset configuration summary:")
            self.logger.info(f"  - Total samples: {total_samples}")
            self.logger.info(f"  - Train: {train_info.num_samples if train_info else 0}")
            self.logger.info(f"  - Val: {val_info.num_samples if val_info else 0}")
            self.logger.info(f"  - Test: {test_info.num_samples if test_info else 0}")
            self.logger.info("=" * 60)
            self.logger.info("PituitaryDatasetPiece execution completed successfully")
            self.logger.info("=" * 60)
            
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
