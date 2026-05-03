"""Trainer class for model training and validation."""
import os
import time
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from utils.metrics import DiceAccumulator
from utils.helpers import prepare_batch, save_rgb_masks


class Trainer:
    """Handles model training, validation, and checkpointing."""
    
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Setup optimizer and loss
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Training state
        self.start_epoch = 0
        self.logs = {
            'config': {
                'num_epochs': config.num_epochs,
                'batch_size': config.batch_size,
                'learning_rate': config.learning_rate,
                'device': str(device),
            },
            'losses': [],
            'val_metrics': []
        }
    
    def train_epoch(self, epoch):
        """Train for one epoch.
        
        Returns:
            avg_loss: Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        num_batches = 0
        
        total_batches = len(self.train_loader)
        print(f"\n[Epoch {epoch+1}/{self.config.num_epochs}] Training...")
        
        for batch_idx, (images, masks) in enumerate(self.train_loader):
            images, masks = prepare_batch(images, masks, self.device, self.config.num_classes)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            
            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                current_avg_loss = running_loss / num_batches
                print(f"  Batch [{batch_idx+1}/{total_batches}] - Loss: {current_avg_loss:.4f}")
        
        return running_loss / num_batches
    
    def validate_epoch(self, epoch):
        """Validate for one epoch.
        
        Returns:
            dict: Validation metrics including loss and Dice scores
        """
        print(f"  Validating...")
        self.model.eval()
        
        val_loss = 0.0
        num_batches = 0
        saved_images_count = 0
        
        dice_acc = DiceAccumulator(self.config.num_classes)
        
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(self.val_loader):
                images, masks = prepare_batch(images, masks, self.device, self.config.num_classes)
                
                outputs = self.model(images)
                val_loss += self.criterion(outputs, masks).item()
                num_batches += 1
                
                # Get predictions for Dice calculation
                predictions = torch.argmax(outputs, dim=1)
                dice_acc.update(predictions, masks)
                
                # Save predictions for first N images
                if saved_images_count < self.config.max_saved_images:
                    num_to_save = min(
                        self.config.max_saved_images - saved_images_count,
                        predictions.shape[0]
                    )
                    
                    save_rgb_masks(
                        predictions, self.config.class_mapping,
                        self.config.pred_dir, 'pred', epoch, batch_idx, num_to_save
                    )
                    
                    # Save ground truth only once
                    gt_file = f'{self.config.pred_dir}/gt_batch{batch_idx}_sample0.png'
                    if not os.path.exists(gt_file):
                        save_rgb_masks(
                            masks, self.config.class_mapping,
                            self.config.pred_dir, 'gt', None, batch_idx, num_to_save
                        )
                    
                    saved_images_count += num_to_save
        
        avg_val_loss = val_loss / num_batches if num_batches > 0 else 0
        dice_per_class, mean_dice = dice_acc.compute()
        
        return {
            'val_loss': avg_val_loss,
            'mean_dice': mean_dice,
            'dice_per_class': dice_per_class
        }
    
    def save_checkpoint(self, epoch, loss):
        """Save model checkpoint.
        
        Returns:
            checkpoint_path: Path to saved checkpoint
        """
        checkpoint_path = f'{self.config.checkpoint_dir}/model_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)
        return checkpoint_path
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint and return the starting epoch."""
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return 0
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle DataParallel prefix mismatch
        state_dict = checkpoint['model_state_dict']
        model_is_dp = isinstance(self.model, nn.DataParallel)
        checkpoint_is_dp = any(k.startswith('module.') for k in state_dict)
        
        if checkpoint_is_dp and not model_is_dp:
            # Strip 'module.' prefix
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
        elif not checkpoint_is_dp and model_is_dp:
            # Add 'module.' prefix
            state_dict = {f'module.{k}': v for k, v in state_dict.items()}
        
        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint['epoch']
        
        print(f"Loaded checkpoint from {checkpoint_path}")
        print(f"Resuming training from epoch {self.start_epoch}")
        
        return self.start_epoch
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*80)
        print("Starting Training")
        print("="*80 + "\n")
        
        for epoch in range(self.start_epoch, self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training
            train_start_time = time.time()
            avg_train_loss = self.train_epoch(epoch)
            train_time = time.time() - train_start_time
            
            # Validation
            val_start_time = time.time()
            val_metrics = self.validate_epoch(epoch)
            val_time = time.time() - val_start_time
            
            epoch_time = time.time() - epoch_start_time
            
            # Logging
            self.logs['losses'].append(avg_train_loss)
            self.logs['val_metrics'].append({
                'epoch': epoch + 1,
                **val_metrics,
                'dice_per_class': {i: val_metrics['dice_per_class'][i] 
                                   for i in range(self.config.num_classes)}
            })
            
            # Save checkpoint
            checkpoint_path = self.save_checkpoint(epoch, avg_train_loss)
            
            # Print summary
            self._print_epoch_summary(
                epoch, avg_train_loss, val_metrics,
                train_time, val_time, epoch_time, checkpoint_path
            )
        
        # Save final model and logs
        self._save_final_results()
    
    def _print_epoch_summary(self, epoch, train_loss, val_metrics, 
                            train_time, val_time, epoch_time, checkpoint_path):
        """Print epoch summary with metrics and timing."""
        # GPU memory stats
        gpu_mem_str = ""
        if torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            gpu_mem_reserved = torch.cuda.max_memory_reserved() / 1024**3  # GB
            gpu_mem_str = f" | GPU Mem: {gpu_mem_allocated:.2f}GB/{gpu_mem_reserved:.2f}GB"
            torch.cuda.reset_peak_memory_stats()
        
        # Calculate ETA
        remaining_epochs = self.config.num_epochs - (epoch + 1)
        eta_minutes = (epoch_time * remaining_epochs) / 60
        
        print(f'\n{"="*80}')
        print(f'Epoch [{epoch+1}/{self.config.num_epochs}] Complete')
        print(f'  Train Loss: {train_loss:.4f} | Val Loss: {val_metrics["val_loss"]:.4f}')
        print(f'  Mean Dice: {val_metrics["mean_dice"]:.4f}')
        
        # Print per-class Dice scores
        for class_id, dice in enumerate(val_metrics['dice_per_class']):
            if class_id < len(self.config.class_names):
                print(f'    Class {class_id} ({self.config.class_names[class_id]}): {dice:.4f}')
        
        print(f'  Train Time: {train_time:.1f}s | Val Time: {val_time:.1f}s | Total: {epoch_time:.1f}s')
        print(f'  ETA: {eta_minutes:.1f} minutes ({remaining_epochs} epochs remaining){gpu_mem_str}')
        print(f'  Checkpoint: {checkpoint_path}')
        print(f'{"="*80}')
    
    def _save_final_results(self):
        """Save final model and training logs."""
        print("\n" + "="*80)
        print("Training Complete")
        print("="*80 + "\n")
        
        # Save final model
        final_model_path = f'{self.config.checkpoint_dir}/final_model.pth'
        torch.save(self.model.state_dict(), final_model_path)
        print(f'Final model saved: {final_model_path}')
        
        # Save logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f'{self.config.log_dir}/training_log_{timestamp}.json'
        with open(log_file, 'w') as f:
            json.dump(self.logs, f, indent=2)
        print(f'Logs saved: {log_file}')
