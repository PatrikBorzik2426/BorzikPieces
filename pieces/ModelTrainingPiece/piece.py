from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, TrainingMetrics, ModelArchitecture
import os
import json
import random
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# MONAI imports
import monai
from monai.networks.nets import UNet, SwinUNETR
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete
from monai.losses import DiceFocalLoss


class PituitaryPatchDataset(Dataset):
    """
    3D Patch-based dataset for pituitary segmentation.
    Implements patch extraction with foreground oversampling and data augmentation.
    """
    
    def __init__(self, root, subjects, patch_size=64, samples_per_volume=10, 
                 is_training=True, fg_oversample=0.9, augment_prob=0.5, logger=None, subject_paths=None):
        self.root = root
        self.subjects = subjects
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.is_training = is_training
        self.fg_oversample = fg_oversample
        self.augment_prob = augment_prob
        self.logger = logger
        self.subject_paths = subject_paths  # Optional: {subject_id: {'image': path, 'mask': path}}
        
        # Use subject_paths if provided, otherwise construct from root
        if subject_paths is None:
            self.img_dir = os.path.join(root, "images")
            self.msk_dir = os.path.join(root, "masks")
        else:
            self.img_dir = None
            self.msk_dir = None
        
        self.fg_locations = self._precompute_fg_locations()
    
    def _log(self, message):
        if self.logger:
            self.logger.info(message)
    
    def _load_volume(self, path):
        """Load volume from either NIfTI or NumPy format"""
        if path.endswith('.npy'):
            return np.load(path)
        else:
            return nib.load(path).get_fdata()
    
    def _precompute_fg_locations(self):
        """Precompute foreground voxel locations for efficient sampling"""
        self._log("Precomputing foreground locations...")
        fg_locs = {}
        for sub in self.subjects:
            # Get mask path from subject_paths or construct it
            if self.subject_paths and sub in self.subject_paths:
                msk_path = self.subject_paths[sub]['mask']
            else:
                msk_path = os.path.join(self.msk_dir, f"{sub}.nii.gz")
            
            if not os.path.exists(msk_path):
                fg_locs[sub] = None
                continue
                
            msk = self._load_volume(msk_path).astype(np.int64)
            fg_coords = np.where(msk > 0)
            
            if len(fg_coords[0]) > 0:
                fg_locs[sub] = np.stack(fg_coords, axis=1)
            else:
                fg_locs[sub] = None
                self._log(f"Warning: {sub} has no foreground!")
        
        return fg_locs
    
    def __len__(self):
        return len(self.subjects) * self.samples_per_volume
    
    def _get_patch_center(self, img_shape, sub):
        """Get patch center with foreground oversampling"""
        fg_locs = self.fg_locations[sub]
        patch_size = self.patch_size
        
        if fg_locs is not None and len(fg_locs) > 0:
            if np.random.rand() < self.fg_oversample:
                # Sample from foreground
                rand_idx = np.random.randint(len(fg_locs))
                center = tuple(fg_locs[rand_idx])
            else:
                # Random sampling with bounds checking
                center = []
                for i, dim_size in enumerate(img_shape):
                    half_patch = patch_size // 2
                    if dim_size > patch_size:
                        # Normal case: sample within valid range
                        c = np.random.randint(half_patch, dim_size - half_patch)
                    else:
                        # Small dimension: center the patch
                        c = dim_size // 2
                    center.append(c)
                center = tuple(center)
        else:
            # Fallback to center if no foreground
            center = tuple(s // 2 for s in img_shape)
        
        return center
    
    def _extract_patch(self, img, center, pad_value=0):
        """Extract 3D patch with padding if necessary"""
        patch_size = self.patch_size
        
        d_start = center[0] - patch_size // 2
        h_start = center[1] - patch_size // 2
        w_start = center[2] - patch_size // 2
        
        d_end = d_start + patch_size
        h_end = h_start + patch_size
        w_end = w_start + patch_size
        
        # Calculate padding
        pad_before = [max(0, -d_start), max(0, -h_start), max(0, -w_start)]
        pad_after = [
            max(0, d_end - img.shape[0]),
            max(0, h_end - img.shape[1]),
            max(0, w_end - img.shape[2])
        ]
        
        # Clip to image bounds
        d_start_clip = max(0, d_start)
        h_start_clip = max(0, h_start)
        w_start_clip = max(0, w_start)
        
        d_end_clip = min(img.shape[0], d_end)
        h_end_clip = min(img.shape[1], h_end)
        w_end_clip = min(img.shape[2], w_end)
        
        # Extract patch
        patch = img[d_start_clip:d_end_clip, 
                   h_start_clip:h_end_clip, 
                   w_start_clip:w_end_clip]
        
        # Apply padding if needed
        if any(p > 0 for p in pad_before + pad_after):
            patch = np.pad(
                patch,
                [(pad_before[i], pad_after[i]) for i in range(3)],
                mode='constant',
                constant_values=pad_value
            )
        
        return patch
    
    def _augment(self, img_patch, msk_patch):
        """Apply data augmentation"""
        if not self.is_training:
            return img_patch, msk_patch
        
        # Random flipping
        for axis in range(3):
            if np.random.rand() > self.augment_prob:
                img_patch = np.flip(img_patch, axis=axis).copy()
                msk_patch = np.flip(msk_patch, axis=axis).copy()
        
        # Random rotation (90 degree increments)
        if np.random.rand() > self.augment_prob:
            k = np.random.randint(1, 4)
            img_patch = np.rot90(img_patch, k=k, axes=(1, 2)).copy()
            msk_patch = np.rot90(msk_patch, k=k, axes=(1, 2)).copy()
        
        # Intensity augmentation
        if np.random.rand() > self.augment_prob:
            # Brightness shift
            img_patch = img_patch + np.random.uniform(-0.2, 0.2)
        
        if np.random.rand() > self.augment_prob:
            # Contrast scaling
            img_patch = img_patch * np.random.uniform(0.8, 1.2)
        
        if np.random.rand() > self.augment_prob:
            # Gaussian noise
            noise = np.random.normal(0, 0.05, img_patch.shape)
            img_patch = img_patch + noise
        
        return img_patch, msk_patch
    
    def __getitem__(self, idx):
        vol_idx = idx // self.samples_per_volume
        sub = self.subjects[vol_idx]
        
        # Get paths from subject_paths or construct them
        if self.subject_paths and sub in self.subject_paths:
            img_path = self.subject_paths[sub]['image']
            msk_path = self.subject_paths[sub]['mask']
        else:
            img_path = os.path.join(self.img_dir, f"{sub}.nii.gz")
            msk_path = os.path.join(self.msk_dir, f"{sub}.nii.gz")
        
        # Load volumes
        img = self._load_volume(img_path).astype(np.float32)
        msk = self._load_volume(msk_path).astype(np.int64)
        
        # Normalize image
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        # Get patch center
        center = self._get_patch_center(img.shape, sub)
        
        # Extract patches
        img_patch = self._extract_patch(img, center, pad_value=0)
        msk_patch = self._extract_patch(msk, center, pad_value=0)
        
        # Apply augmentation
        img_patch, msk_patch = self._augment(img_patch, msk_patch)
        
        # Convert to tensors
        img_patch = torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0)
        msk_patch = torch.tensor(msk_patch, dtype=torch.int64).unsqueeze(0)
        
        return img_patch, msk_patch, sub


class ModelTrainingPiece(BasePiece):
    """
    A piece that trains 3D medical image segmentation models.
    
    Implements:
    - Patch-based training with foreground oversampling
    - Data augmentation
    - UNet and SwinUNETR architectures
    - Dice + Focal loss with class weights
    - Learning rate scheduling
    - Early stopping
    - Comprehensive metrics tracking
    - Model checkpointing
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info("=" * 80)
            self.logger.info("Starting ModelTrainingPiece execution")
            self.logger.info("=" * 80)
            
            # Set random seeds
            random.seed(input_data.random_seed)
            np.random.seed(input_data.random_seed)
            torch.manual_seed(input_data.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(input_data.random_seed)
            
            # Setup device
            if input_data.use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU")
            
            # Create output directories
            os.makedirs(input_data.output_dir, exist_ok=True)
            checkpoint_dir = os.path.join(input_data.output_dir, "checkpoints")
            os.makedirs(checkpoint_dir, exist_ok=True)
            plots_dir = os.path.join(input_data.output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)
            
            # Load or generate dataset configuration
            subject_paths = None  # Will be populated if using upstream subjects
            
            if input_data.subjects is not None:
                # Using upstream subject list - perform train/val split
                self.logger.info(f"Using {len(input_data.subjects)} subjects from upstream piece")
                self.logger.info(f"Performing train/val split: {input_data.train_val_split:.1%} train, {1-input_data.train_val_split:.1%} val")
                
                # Create subject path mapping
                subject_paths = {
                    s.subject_id: {
                        'image': s.image_path,
                        'mask': s.mask_path
                    } for s in input_data.subjects
                }
                
                # Shuffle and split
                all_subjects = [s.subject_id for s in input_data.subjects]
                random.shuffle(all_subjects)
                split_idx = int(len(all_subjects) * input_data.train_val_split)
                train_subjects = all_subjects[:split_idx]
                val_subjects = all_subjects[split_idx:]
                
                # data_root not used when subject_paths is provided, but set for logging
                data_root = "(using direct paths from upstream)"
                    
            elif input_data.dataset_config_path and os.path.exists(input_data.dataset_config_path):
                # Using config file from PituitaryDatasetPiece
                self.logger.info(f"Loading dataset config from: {input_data.dataset_config_path}")
                with open(input_data.dataset_config_path, 'r') as f:
                    dataset_config = json.load(f)
                
                train_subjects = [s['subject_id'] for s in dataset_config['train']['subjects']]
                val_subjects = [s['subject_id'] for s in dataset_config['val']['subjects']]
                data_root = input_data.data_root
            else:
                raise ValueError(
                    "Must provide either 'subjects' from upstream piece or valid 'dataset_config_path'. "
                    f"Got subjects={input_data.subjects is not None}, config_path={input_data.dataset_config_path}"
                )
            
            self.logger.info(f"Training subjects: {len(train_subjects)}")
            self.logger.info(f"Validation subjects: {len(val_subjects)}")
            if subject_paths:
                self.logger.info(f"Using direct file paths from upstream piece")
            else:
                self.logger.info(f"Data root: {data_root}")
            
            # Create datasets
            train_dataset = PituitaryPatchDataset(
                root=data_root if not subject_paths else "/tmp",  # root unused when subject_paths provided
                subjects=train_subjects,
                patch_size=input_data.patch_size,
                samples_per_volume=input_data.samples_per_volume,
                is_training=True,
                fg_oversample=input_data.foreground_oversample,
                augment_prob=input_data.augmentation_probability if input_data.use_augmentation else 0.0,
                logger=self.logger,
                subject_paths=subject_paths
            )
            
            val_dataset = PituitaryPatchDataset(
                root=data_root if not subject_paths else "/tmp",  # root unused when subject_paths provided
                subjects=val_subjects,
                patch_size=input_data.patch_size,
                samples_per_volume=5,  # Fewer samples for validation
                is_training=False,
                fg_oversample=0.5,
                augment_prob=0.0,
                logger=self.logger,
                subject_paths=subject_paths
            )
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=input_data.batch_size,
                shuffle=True,
                num_workers=input_data.num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=input_data.batch_size,
                shuffle=False,
                num_workers=input_data.num_workers,
                pin_memory=(device.type == 'cuda')
            )
            
            self.logger.info(f"Train batches: {len(train_loader)}")
            self.logger.info(f"Val batches: {len(val_loader)}")
            
            # Create model
            self.logger.info(f"Creating model: {input_data.model_architecture.value}")
            if input_data.model_architecture == ModelArchitecture.UNET:
                model = UNet(
                    spatial_dims=3,
                    in_channels=1,
                    out_channels=input_data.num_classes,
                    channels=(16, 32, 64, 128, 256),
                    strides=(2, 2, 2, 2),
                    num_res_units=2,
                ).to(device)
            elif input_data.model_architecture == ModelArchitecture.SWIN_UNETR:
                model = SwinUNETR(
                    img_size=(input_data.patch_size, input_data.patch_size, input_data.patch_size),
                    in_channels=1,
                    out_channels=input_data.num_classes,
                    feature_size=48,
                ).to(device)
            
            # Setup class weights
            if input_data.class_weights:
                class_weights = torch.tensor(input_data.class_weights, dtype=torch.float32).to(device)
                self.logger.info(f"Using custom class weights: {input_data.class_weights}")
            else:
                # Default weights: lower weight for background, higher for foreground
                class_weights = torch.tensor([0.1] + [5.0] * (input_data.num_classes - 1), dtype=torch.float32).to(device)
                self.logger.info(f"Using default class weights: {class_weights.tolist()}")
            
            # Setup loss function
            loss_fn = DiceFocalLoss(
                include_background=False,
                to_onehot_y=True,
                softmax=True,
                gamma=2.0,
                weight=class_weights[1:],  # Exclude background
                lambda_dice=1.0,
                lambda_focal=1.0
            )
            
            # Setup optimizer
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=input_data.learning_rate,
                weight_decay=input_data.weight_decay
            )
            
            # Setup scheduler
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=0.5,
                patience=input_data.lr_scheduler_patience,
                verbose=True
            )
            
            # Setup metrics
            dice_metric = DiceMetric(include_background=False, reduction="mean")
            post_pred = AsDiscrete(argmax=True)
            post_label = AsDiscrete(to_onehot=input_data.num_classes)
            
            # Training loop
            self.logger.info("=" * 80)
            self.logger.info("Starting training loop")
            self.logger.info("=" * 80)
            
            training_history = []
            best_val_dice = 0.0
            best_epoch = 0
            best_model_path = None
            epochs_without_improvement = 0
            
            for epoch in range(input_data.epochs):
                # Training phase
                model.train()
                train_loss = 0.0
                num_train_batches = 0
                
                self.logger.info(f"Epoch {epoch+1}/{input_data.epochs}")
                
                for batch_idx, (img, msk, sid) in enumerate(train_loader):
                    img = img.to(device)
                    msk = msk.to(device)
                    
                    optimizer.zero_grad()
                    preds = model(img)
                    
                    loss = loss_fn(preds, msk)
                    
                    if torch.isnan(loss):
                        self.logger.warning(f"NaN loss detected in batch {batch_idx}, skipping...")
                        continue
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    train_loss += loss.item()
                    num_train_batches += 1
                    
                    if batch_idx % 10 == 0:
                        self.logger.info(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0.0
                
                # Validation phase
                val_loss = None
                val_dice = None
                
                if (epoch + 1) % input_data.eval_interval == 0:
                    model.eval()
                    dice_metric.reset()
                    val_loss_sum = 0.0
                    num_val_batches = 0
                    
                    with torch.no_grad():
                        for img, msk, sid in val_loader:
                            img = img.to(device)
                            msk = msk.to(device)
                            
                            preds = model(img)
                            
                            # Calculate loss
                            loss = loss_fn(preds, msk)
                            val_loss_sum += loss.item()
                            num_val_batches += 1
                            
                            # Calculate Dice
                            preds_post = post_pred(preds)
                            preds_onehot = post_label(preds_post)
                            labels_onehot = post_label(msk)
                            dice_metric(preds_onehot, labels_onehot)
                    
                    val_loss = val_loss_sum / num_val_batches if num_val_batches > 0 else 0.0
                    val_dice = dice_metric.aggregate().item()
                    dice_metric.reset()
                    
                    # Update scheduler
                    scheduler.step(val_dice)
                    
                    # Check for best model
                    if val_dice > best_val_dice:
                        best_val_dice = val_dice
                        best_epoch = epoch + 1
                        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_dice': val_dice,
                            'train_loss': avg_train_loss
                        }, best_model_path)
                        self.logger.info(f"  New best model! Val Dice: {val_dice:.4f}")
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                
                # Log metrics
                current_lr = optimizer.param_groups[0]['lr']
                metrics = TrainingMetrics(
                    epoch=epoch + 1,
                    train_loss=avg_train_loss,
                    val_loss=val_loss,
                    val_dice=val_dice,
                    learning_rate=current_lr
                )
                training_history.append(metrics)
                
                log_msg = f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}"
                if val_loss is not None:
                    log_msg += f", Val Loss={val_loss:.4f}, Val Dice={val_dice:.4f}"
                log_msg += f", LR={current_lr:.2e}"
                self.logger.info(log_msg)
                
                # Save checkpoint
                if (epoch + 1) % input_data.save_checkpoint_interval == 0:
                    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth")
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg_train_loss,
                        'val_dice': val_dice if val_dice else 0.0
                    }, checkpoint_path)
                
                # Early stopping
                if input_data.early_stopping_patience > 0 and epochs_without_improvement >= input_data.early_stopping_patience:
                    self.logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            # Save final model
            final_model_path = os.path.join(input_data.output_dir, "final_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': input_data.model_dump()
            }, final_model_path)
            
            # Generate training plots
            self._generate_training_plots(training_history, plots_dir)
            
            # Create training summary
            training_summary = f"""
Training Summary
================
Model: {input_data.model_architecture.value}
Total Epochs: {len(training_history)}
Best Epoch: {best_epoch}
Best Validation Dice: {best_val_dice:.4f}
Final Training Loss: {training_history[-1].train_loss:.4f}

Dataset:
- Training subjects: {len(train_subjects)}
- Validation subjects: {len(val_subjects)}
- Patch size: {input_data.patch_size}³
- Batch size: {input_data.batch_size}

Hyperparameters:
- Learning rate: {input_data.learning_rate}
- Weight decay: {input_data.weight_decay}
- Samples per volume: {input_data.samples_per_volume}
- Foreground oversample: {input_data.foreground_oversample}
- Augmentation: {input_data.use_augmentation}
"""
            
            # Save training summary
            summary_path = os.path.join(input_data.output_dir, "training_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(training_summary)
            
            # Save training history
            history_path = os.path.join(input_data.output_dir, "training_history.json")
            with open(history_path, 'w') as f:
                json.dump([m.model_dump() for m in training_history], f, indent=2)
            
            self.logger.info(training_summary)
            self.logger.info("=" * 80)
            self.logger.info("Training completed successfully")
            self.logger.info("=" * 80)
            
            # Run inference on validation samples
            inference_dir = os.path.join(input_data.output_dir, "inference_results")
            os.makedirs(inference_dir, exist_ok=True)
            
            if input_data.run_inference:
                self.logger.info("=" * 80)
                self.logger.info("Running inference on validation samples")
                self.logger.info("=" * 80)
                
                # Load best model for inference
                if best_model_path and os.path.exists(best_model_path):
                    checkpoint = torch.load(best_model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    self.logger.info(f"Loaded best model from epoch {best_epoch}")
                
                self._run_inference_and_visualize(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    num_classes=input_data.num_classes,
                    output_dir=inference_dir,
                    num_samples=input_data.num_inference_samples
                )
            
            return OutputModel(
                model_path=final_model_path,
                checkpoint_dir=checkpoint_dir,
                best_model_path=best_model_path or final_model_path,
                best_val_dice=best_val_dice,
                best_epoch=best_epoch,
                final_train_loss=training_history[-1].train_loss,
                total_epochs_trained=len(training_history),
                training_history=training_history,
                training_summary=training_summary.strip(),
                plots_dir=plots_dir,
                inference_results_dir=inference_dir if input_data.run_inference else None
            )
            
        except Exception as e:
            self.logger.error(f"Error in ModelTrainingPiece: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _generate_training_plots(self, history, plots_dir):
        """Generate training visualization plots"""
        epochs = [m.epoch for m in history]
        train_losses = [m.train_loss for m in history]
        val_losses = [m.val_loss for m in history if m.val_loss is not None]
        val_dices = [m.val_dice for m in history if m.val_dice is not None]
        val_epochs = [m.epoch for m in history if m.val_dice is not None]
        
        # Loss plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
        if val_losses:
            ax.plot(val_epochs, val_losses, label='Val Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'loss_curve.png'), dpi=150)
        plt.close()
        
        # Dice plot
        if val_dices:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(val_epochs, val_dices, label='Val Dice', marker='o', markersize=4, color='green')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Dice Score')
            ax.set_title('Validation Dice Score')
            ax.legend()
            ax.grid(True, alpha=0.3)
            best_idx = val_dices.index(max(val_dices))
            ax.axvline(val_epochs[best_idx], color='red', linestyle='--', label=f'Best: {max(val_dices):.4f}')
            ax.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, 'dice_curve.png'), dpi=150)
            plt.close()
    
    def _run_inference_and_visualize(self, model, val_loader, device, num_classes, output_dir, num_samples=5):
        """Run inference on validation samples and visualize results with confidence scores"""
        model.eval()
        
        samples_processed = 0
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        post_pred = AsDiscrete(argmax=True)
        post_label = AsDiscrete(to_onehot=num_classes)
        
        self.logger.info(f"Processing {num_samples} validation samples for inference...")
        
        with torch.no_grad():
            for batch_idx, (img, msk, subject_ids) in enumerate(val_loader):
                if samples_processed >= num_samples:
                    break
                
                img = img.to(device)
                msk = msk.to(device)
                
                # Run inference
                logits = model(img)
                probs = torch.softmax(logits, dim=1)  # Convert to probabilities
                preds = torch.argmax(logits, dim=1)  # Get predictions
                
                # Calculate Dice scores for this batch
                preds_post = post_pred(logits)
                preds_onehot = post_label(preds_post)
                labels_onehot = post_label(msk)
                dice_scores = dice_metric(preds_onehot, labels_onehot)
                
                # Process each sample in batch
                batch_size = img.shape[0]
                for i in range(min(batch_size, num_samples - samples_processed)):
                    subject_id = subject_ids[i] if isinstance(subject_ids, (list, tuple)) else f"sample_{samples_processed}"
                    
                    # Get confidence scores (max probability for each voxel)
                    confidence_map = torch.max(probs[i], dim=0)[0]  # Shape: [D, H, W]
                    mean_confidence = confidence_map.mean().item()
                    max_confidence = confidence_map.max().item()
                    min_confidence = confidence_map.min().item()
                    
                    # Get class-specific confidences
                    class_confidences = {}
                    for class_idx in range(num_classes):
                        class_mask = (preds[i] == class_idx)
                        if class_mask.sum() > 0:
                            class_conf = probs[i, class_idx][class_mask].mean().item()
                            class_confidences[f"class_{class_idx}"] = class_conf
                        else:
                            class_confidences[f"class_{class_idx}"] = 0.0
                    
                    # Calculate Dice score
                    dice_score = dice_scores[i].item() if dice_scores.dim() > 0 else dice_scores.item()
                    
                    # Log confidence information
                    self.logger.info(f"\nInference for {subject_id}:")
                    self.logger.info(f"  Dice Score: {dice_score:.4f}")
                    self.logger.info(f"  Mean Confidence: {mean_confidence:.4f}")
                    self.logger.info(f"  Max Confidence: {max_confidence:.4f}")
                    self.logger.info(f"  Min Confidence: {min_confidence:.4f}")
                    self.logger.info(f"  Per-class Confidences:")
                    for class_name, conf in class_confidences.items():
                        self.logger.info(f"    {class_name}: {conf:.4f}")
                    
                    # Visualize and save
                    self._visualize_inference_result(
                        image=img[i, 0].cpu().numpy(),
                        ground_truth=msk[i, 0].cpu().numpy(),
                        prediction=preds[i].cpu().numpy(),
                        confidence_map=confidence_map.cpu().numpy(),
                        probabilities=probs[i].cpu().numpy(),
                        subject_id=subject_id,
                        dice_score=dice_score,
                        mean_confidence=mean_confidence,
                        class_confidences=class_confidences,
                        output_dir=output_dir,
                        num_classes=num_classes
                    )
                    
                    # Save confidence data to JSON
                    confidence_data = {
                        "subject_id": subject_id,
                        "dice_score": float(dice_score),
                        "mean_confidence": float(mean_confidence),
                        "max_confidence": float(max_confidence),
                        "min_confidence": float(min_confidence),
                        "class_confidences": {k: float(v) for k, v in class_confidences.items()}
                    }
                    
                    confidence_json_path = os.path.join(output_dir, f"{subject_id}_confidence.json")
                    with open(confidence_json_path, 'w') as f:
                        json.dump(confidence_data, f, indent=2)
                    
                    samples_processed += 1
                
                if samples_processed >= num_samples:
                    break
        
        self.logger.info(f"\nInference completed. Results saved to: {output_dir}")
    
    def _visualize_inference_result(self, image, ground_truth, prediction, confidence_map, 
                                   probabilities, subject_id, dice_score, mean_confidence,
                                   class_confidences, output_dir, num_classes):
        """Create comprehensive visualization of inference results"""
        
        # Select middle slice for visualization
        depth = image.shape[0]
        mid_slice = depth // 2
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Image, Ground Truth, Prediction, Confidence Map
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image[mid_slice], cmap='gray')
        ax1.set_title('Input Image', fontsize=12)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(ground_truth[mid_slice], cmap='tab10', vmin=0, vmax=num_classes-1)
        ax2.set_title('Ground Truth', fontsize=12)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(prediction[mid_slice], cmap='tab10', vmin=0, vmax=num_classes-1)
        ax3.set_title(f'Prediction (Dice: {dice_score:.3f})', fontsize=12)
        ax3.axis('off')
        
        ax4 = fig.add_subplot(gs[0, 3])
        im4 = ax4.imshow(confidence_map[mid_slice], cmap='viridis', vmin=0, vmax=1)
        ax4.set_title(f'Confidence Map (μ={mean_confidence:.3f})', fontsize=12)
        ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
        
        # Row 2: Overlay visualizations
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.imshow(image[mid_slice], cmap='gray')
        # Overlay ground truth with transparency
        gt_overlay = np.ma.masked_where(ground_truth[mid_slice] == 0, ground_truth[mid_slice])
        ax5.imshow(gt_overlay, cmap='tab10', alpha=0.5, vmin=0, vmax=num_classes-1)
        ax5.set_title('GT Overlay', fontsize=12)
        ax5.axis('off')
        
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.imshow(image[mid_slice], cmap='gray')
        # Overlay prediction with transparency
        pred_overlay = np.ma.masked_where(prediction[mid_slice] == 0, prediction[mid_slice])
        ax6.imshow(pred_overlay, cmap='tab10', alpha=0.5, vmin=0, vmax=num_classes-1)
        ax6.set_title('Prediction Overlay', fontsize=12)
        ax6.axis('off')
        
        # Error map (differences between GT and prediction)
        ax7 = fig.add_subplot(gs[1, 2])
        error_map = (ground_truth[mid_slice] != prediction[mid_slice]).astype(float)
        im7 = ax7.imshow(error_map, cmap='Reds', vmin=0, vmax=1)
        ax7.set_title('Error Map', fontsize=12)
        ax7.axis('off')
        plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        
        # Confidence histogram
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.hist(confidence_map.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax8.axvline(mean_confidence, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_confidence:.3f}')
        ax8.set_xlabel('Confidence', fontsize=10)
        ax8.set_ylabel('Frequency', fontsize=10)
        ax8.set_title('Confidence Distribution', fontsize=12)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Row 3: Class probability maps for first 4 classes
        for class_idx in range(min(4, num_classes)):
            ax = fig.add_subplot(gs[2, class_idx])
            class_prob_map = probabilities[class_idx, mid_slice]
            im = ax.imshow(class_prob_map, cmap='hot', vmin=0, vmax=1)
            conf_val = class_confidences.get(f"class_{class_idx}", 0.0)
            ax.set_title(f'Class {class_idx} Prob\n(conf={conf_val:.3f})', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add overall title
        fig.suptitle(f'Inference Results: {subject_id}\nDice Score: {dice_score:.4f} | Mean Confidence: {mean_confidence:.4f}',
                    fontsize=14, fontweight='bold')
        
        # Save figure
        output_path = os.path.join(output_dir, f"{subject_id}_inference.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Visualization saved: {output_path}")
