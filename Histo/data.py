"""Dataset utilities and dataloader setup."""
import torch
from torch.utils.data import DataLoader, random_split
from datasets.beetle_dataset import BeetleDataset


def setup_dataloaders(config):
    """Setup train and validation dataloaders.
    
    Args:
        config: Config object with dataset parameters
        
    Returns:
        tuple: (train_loader, val_loader, dataset_info)
    """
    # Load full dataset
    full_dataset = BeetleDataset(
        image_dir=config.image_dir,
        mask_dir=config.mask_dir,
        rgb_to_class_map=config.rgb_to_class_map
    )
    
    # Compute split sizes
    dataset_size = len(full_dataset)
    train_size = int(config.train_val_split * dataset_size)
    val_size = dataset_size - train_size
    
    # Reproducible split
    generator = torch.Generator().manual_seed(0)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    dataset_info = {
        'total_size': dataset_size,
        'train_size': train_size,
        'val_size': val_size
    }
    
    return train_loader, val_loader, dataset_info
