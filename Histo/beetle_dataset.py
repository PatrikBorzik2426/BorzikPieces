import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset


class BeetleDataset(Dataset):
    """
    Custom dataset for beetle semantic segmentation with RGB mask support.
    """

    def __init__(self, image_dir: str, mask_dir: str, rgb_to_class_map: dict, 
                 image_transform=None, mask_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.images = sorted(os.listdir(image_dir))
        self.rgb_to_class_map = rgb_to_class_map

    def __len__(self):
        return len(self.images)

    def rgb_to_class(self, mask_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB mask to class indices."""
        mask_class = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
        for rgb, class_idx in self.rgb_to_class_map.items():
            rgb_mask = np.all(mask_rgb == np.array(rgb), axis=2)
            mask_class[rgb_mask] = class_idx
        return mask_class

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)

        image = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask = self.rgb_to_class(mask_rgb)

        if self.image_transform and self.mask_transform:
            augmented = self.image_transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask