"""Main training script for semantic segmentation."""
import torch
import torch.nn as nn

from configs import Config
from models.unet import UNET
from trainers import Trainer
from utils.helpers import setup_device, create_directories
from utils.data import setup_dataloaders


def main():
    """Main training pipeline."""
    # Setup
    device = setup_device()
    config = Config()
    create_directories(config)
    
    # Setup dataloaders
    train_loader, val_loader, dataset_info = setup_dataloaders(config)
    print(f"\nDataset loaded:")
    print(f"  Total: {dataset_info['total_size']}")
    print(f"  Train: {dataset_info['train_size']}")
    print(f"  Validation: {dataset_info['val_size']}")
    
    # Initialize model
    model = UNET(out_channels=config.num_classes).to(device)
    
    # Use DataParallel for multi-GPU training
    if torch.cuda.device_count() > 1:
        print(f"\nUsing DataParallel with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Initialize trainer
    trainer = Trainer(model, train_loader, val_loader, config, device)
    
    # Load checkpoint if specified
    if config.checkpoint_to_load is not None:
        trainer.load_checkpoint(config.checkpoint_to_load)
    
    # Train
    trainer.train()


if __name__ == '__main__':
    main()

