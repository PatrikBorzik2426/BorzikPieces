"""Configuration class for training parameters and paths."""


class Config:
    """Configuration class for training parameters and paths."""
    
    # Dataset paths
    image_dir = 'filtered_dataset/images'
    mask_dir = 'filtered_dataset/masks'
    
    # Training hyperparameters
    num_epochs = 100
    batch_size = 16
    learning_rate = 0.001
    train_val_split = 0.8  # 80% train, 20% validation
    
    # Output directories
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    pred_dir = 'predictions'
    
    # Resume from checkpoint (set to checkpoint path or None to start fresh)
    checkpoint_to_load = 'checkpoints/model_epoch_6.pth'
    
    # Class mapping (class_id -> RGB tuple)
    class_mapping = {
        0: (0, 0, 0),      # unannotated
        1: (128, 128, 128),    # other
        2: (0, 255, 0),    # non-invasive epithelium
        3: (255, 0, 0),    # invasive epithelium
        4: (0, 0, 255),  # necrosis
    }
    
    # Class names for logging
    class_names = ['unannotated', 'other', 'non-invasive', 'invasive', 'necrosis']
    
    # Validation settings
    max_saved_images = 10  # Maximum number of validation images to save per epoch
    
    @property
    def num_classes(self):
        return len(self.class_mapping)
    
    @property
    def rgb_to_class_map(self):
        """Derive RGB to class mapping from class_mapping."""
        return {rgb: class_id for class_id, rgb in self.class_mapping.items()}
