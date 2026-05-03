"""Helper utility functions."""
import os
import torch
import numpy as np
from PIL import Image


def setup_device():
    """Setup and display device information (GPU or CPU)."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using device: {device}")
        print(f"Number of GPUs available: {num_gpus}")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print(f"Using device: {device}")
    return device


def create_directories(config):
    """Create output directories if they don't exist."""
    for directory in [config.checkpoint_dir, config.log_dir, config.pred_dir]:
        os.makedirs(directory, exist_ok=True)


def prepare_batch(images, masks, device, num_classes):
    """Prepare batch by transposing images and moving to device."""
    if images.dim() == 4:
        images = images.permute(0, 3, 1, 2).float()
    images = images.to(device)
    masks = masks.to(device)
    
    # Validate mask values
    assert masks.min() >= 0, f"Mask contains negative values: {masks.min()}"
    assert masks.max() < num_classes, f"Mask contains values >= {num_classes}: {masks.max()}"
    
    return images, masks


def save_rgb_masks(masks, class_mapping, save_dir, prefix, epoch=None, batch_idx=0, num_samples=2):
    """Save masks as RGB images.
    
    Args:
        masks: Tensor of class indices [B, H, W]
        class_mapping: Dict mapping class_id -> RGB tuple
        save_dir: Directory to save images
        prefix: Prefix for filename (e.g., 'pred' or 'gt')
        epoch: Optional epoch number for predictions
        batch_idx: Batch index
        num_samples: Number of samples to save from batch
    """
    mask_array = masks.cpu().numpy()
    for i in range(min(num_samples, mask_array.shape[0])):
        mask_img = mask_array[i]
        H, W = mask_img.shape
        mask_rgb = np.zeros((H, W, 3), dtype=np.uint8)
        
        for class_id, rgb in class_mapping.items():
            class_mask = mask_img == class_id
            mask_rgb[class_mask] = rgb
        
        if epoch is not None:
            save_path = f'{save_dir}/{prefix}_epoch{epoch+1}_batch{batch_idx}_sample{i}.png'
        else:
            save_path = f'{save_dir}/{prefix}_batch{batch_idx}_sample{i}.png'
        
        Image.fromarray(mask_rgb).save(save_path)
