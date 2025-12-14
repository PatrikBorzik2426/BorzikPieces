from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, InferenceMetrics, ModelArchitecture
import os
import json
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# MONAI imports
from monai.networks.nets import UNet, SwinUNETR
from monai.metrics import DiceMetric
from monai.transforms import AsDiscrete


class InferenceDataset(Dataset):
    """Simple dataset for inference on 3D patches"""
    
    def __init__(self, image_paths, mask_paths=None, patch_size=64, samples_per_volume=5, logger=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.patch_size = patch_size
        self.samples_per_volume = samples_per_volume
        self.logger = logger
    
    def _log(self, message):
        if self.logger:
            self.logger.info(message)
    
    def _load_volume(self, path):
        """Load volume from NIfTI or NumPy format"""
        if path.endswith('.npy'):
            return np.load(path)
        else:
            return nib.load(path).get_fdata()
    
    def __len__(self):
        return len(self.image_paths) * self.samples_per_volume
    
    def _extract_center_patch(self, img, patch_size):
        """Extract center patch from volume"""
        d, h, w = img.shape
        d_start = max(0, (d - patch_size) // 2)
        h_start = max(0, (h - patch_size) // 2)
        w_start = max(0, (w - patch_size) // 2)
        
        d_end = min(d, d_start + patch_size)
        h_end = min(h, h_start + patch_size)
        w_end = min(w, w_start + patch_size)
        
        patch = img[d_start:d_end, h_start:h_end, w_start:w_end]
        
        # Pad if necessary
        if patch.shape != (patch_size, patch_size, patch_size):
            padded = np.zeros((patch_size, patch_size, patch_size), dtype=patch.dtype)
            padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
            patch = padded
        
        return patch
    
    def __getitem__(self, idx):
        vol_idx = idx // self.samples_per_volume
        img_path = self.image_paths[vol_idx]
        
        # Load image
        img = self._load_volume(img_path).astype(np.float32)
        
        # Normalize
        p1, p99 = np.percentile(img, [1, 99])
        img = np.clip(img, p1, p99)
        img = (img - img.mean()) / (img.std() + 1e-8)
        
        # Extract patch
        img_patch = self._extract_center_patch(img, self.patch_size)
        img_patch = torch.tensor(img_patch, dtype=torch.float32).unsqueeze(0)
        
        # Load mask if available
        if self.mask_paths and vol_idx < len(self.mask_paths):
            msk_path = self.mask_paths[vol_idx]
            msk = self._load_volume(msk_path).astype(np.int64)
            msk_patch = self._extract_center_patch(msk, self.patch_size)
            msk_patch = torch.tensor(msk_patch, dtype=torch.int64).unsqueeze(0)
        else:
            msk_patch = torch.zeros_like(img_patch, dtype=torch.int64)
        
        # Extract subject ID from path
        subject_id = os.path.basename(img_path).split('.')[0]
        
        return img_patch, msk_patch, subject_id


class ModelInferencePiece(BasePiece):
    """
    A piece that loads trained models and performs inference with confidence visualization.
    
    Features:
    - Loads pre-trained UNet or SwinUNETR models
    - Runs inference on new samples
    - Calculates confidence scores (voxel-wise and class-wise)
    - Computes Dice metrics if ground truth is available
    - Generates comprehensive visualizations
    - Saves predictions and metrics
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info("=" * 80)
            self.logger.info("Starting ModelInferencePiece execution")
            self.logger.info("=" * 80)
            
            # Setup device
            if input_data.use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU")
            
            # Create output directories
            os.makedirs(input_data.output_dir, exist_ok=True)
            viz_dir = os.path.join(input_data.output_dir, "visualizations") if input_data.save_visualizations else None
            pred_dir = os.path.join(input_data.output_dir, "predictions") if input_data.save_predictions else None
            
            if viz_dir:
                os.makedirs(viz_dir, exist_ok=True)
            if pred_dir:
                os.makedirs(pred_dir, exist_ok=True)
            
            # Get image and mask paths
            if input_data.subjects:
                self.logger.info(f"Using {len(input_data.subjects)} subjects from upstream piece")
                image_paths = [s.image_path for s in input_data.subjects]
                mask_paths = [s.mask_path for s in input_data.subjects if s.mask_path]
            elif input_data.image_paths:
                self.logger.info(f"Using {len(input_data.image_paths)} image paths")
                image_paths = input_data.image_paths
                mask_paths = input_data.mask_paths or []
            else:
                raise ValueError("Must provide either 'subjects' or 'image_paths'")
            
            # Limit number of samples if specified
            if input_data.num_samples > 0:
                image_paths = image_paths[:input_data.num_samples]
                if mask_paths:
                    mask_paths = mask_paths[:input_data.num_samples]
            
            has_ground_truth = len(mask_paths) > 0
            self.logger.info(f"Processing {len(image_paths)} images")
            self.logger.info(f"Ground truth available: {has_ground_truth}")
            
            # Create dataset and dataloader
            dataset = InferenceDataset(
                image_paths=image_paths,
                mask_paths=mask_paths if has_ground_truth else None,
                patch_size=input_data.patch_size,
                samples_per_volume=input_data.samples_per_volume,
                logger=self.logger
            )
            
            dataloader = DataLoader(
                dataset,
                batch_size=input_data.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=(device.type == 'cuda')
            )
            
            # Load model
            self.logger.info(f"Loading model from: {input_data.model_path}")
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
            
            # Load model weights
            checkpoint = torch.load(input_data.model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Loaded model from checkpoint (epoch: {checkpoint.get('epoch', 'unknown')})")
            else:
                model.load_state_dict(checkpoint)
                self.logger.info("Loaded model weights")
            
            # Run inference
            self.logger.info("=" * 80)
            self.logger.info("Running inference")
            self.logger.info("=" * 80)
            
            inference_metrics = self._run_inference(
                model=model,
                dataloader=dataloader,
                device=device,
                num_classes=input_data.num_classes,
                has_ground_truth=has_ground_truth,
                viz_dir=viz_dir,
                pred_dir=pred_dir
            )
            
            # Calculate summary statistics
            mean_confidence = np.mean([m.mean_confidence for m in inference_metrics])
            mean_dice = None
            if has_ground_truth:
                mean_dice = np.mean([m.dice_score for m in inference_metrics])
            
            # Create summary report
            summary = self._create_summary_report(
                inference_metrics=inference_metrics,
                mean_confidence=mean_confidence,
                mean_dice=mean_dice,
                num_samples=len(image_paths)
            )
            
            # Save summary
            summary_path = os.path.join(input_data.output_dir, "inference_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            # Save metrics to JSON
            metrics_path = os.path.join(input_data.output_dir, "inference_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump([m.model_dump() for m in inference_metrics], f, indent=2)
            
            self.logger.info(summary)
            self.logger.info("=" * 80)
            self.logger.info("Inference completed successfully")
            self.logger.info("=" * 80)
            
            return OutputModel(
                output_dir=input_data.output_dir,
                num_samples_processed=len(inference_metrics),
                mean_dice_score=mean_dice,
                mean_confidence=mean_confidence,
                inference_metrics=inference_metrics,
                visualization_dir=viz_dir,
                predictions_dir=pred_dir,
                summary_report=summary.strip()
            )
            
        except Exception as e:
            self.logger.error(f"Error in ModelInferencePiece: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def _run_inference(self, model, dataloader, device, num_classes, has_ground_truth, viz_dir, pred_dir):
        """Run inference and collect metrics"""
        model.eval()
        
        inference_metrics = []
        dice_metric = DiceMetric(include_background=False, reduction="mean_batch")
        post_pred = AsDiscrete(argmax=True)
        post_label = AsDiscrete(to_onehot=num_classes)
        
        samples_processed = 0
        
        with torch.no_grad():
            for batch_idx, (img, msk, subject_ids) in enumerate(dataloader):
                img = img.to(device)
                msk = msk.to(device)
                
                # Run inference
                logits = model(img)
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                
                # Calculate Dice if ground truth available
                dice_scores = None
                if has_ground_truth:
                    preds_post = post_pred(logits)
                    preds_onehot = post_label(preds_post)
                    labels_onehot = post_label(msk.squeeze(1))
                    dice_scores = dice_metric(preds_onehot, labels_onehot)
                
                # Process each sample in batch
                batch_size = img.shape[0]
                for i in range(batch_size):
                    subject_id = subject_ids[i] if isinstance(subject_ids, (list, tuple)) else f"sample_{samples_processed}"
                    
                    # Get confidence scores
                    confidence_map = torch.max(probs[i], dim=0)[0]
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
                    
                    # Get Dice score
                    dice_score = 0.0
                    if dice_scores is not None:
                        dice_score = dice_scores[i].item() if dice_scores.dim() > 0 else dice_scores.item()
                    
                    # Log results
                    self.logger.info(f"\nInference for {subject_id}:")
                    if has_ground_truth:
                        self.logger.info(f"  Dice Score: {dice_score:.4f}")
                    self.logger.info(f"  Mean Confidence: {mean_confidence:.4f}")
                    self.logger.info(f"  Max Confidence: {max_confidence:.4f}")
                    self.logger.info(f"  Min Confidence: {min_confidence:.4f}")
                    
                    # Create metrics object
                    metrics = InferenceMetrics(
                        subject_id=subject_id,
                        dice_score=dice_score,
                        mean_confidence=mean_confidence,
                        max_confidence=max_confidence,
                        min_confidence=min_confidence,
                        class_confidences=class_confidences
                    )
                    inference_metrics.append(metrics)
                    
                    # Visualize if requested
                    if viz_dir:
                        self._visualize_inference_result(
                            image=img[i, 0].cpu().numpy(),
                            ground_truth=msk[i, 0].cpu().numpy() if has_ground_truth else None,
                            prediction=preds[i].cpu().numpy(),
                            confidence_map=confidence_map.cpu().numpy(),
                            probabilities=probs[i].cpu().numpy(),
                            subject_id=subject_id,
                            dice_score=dice_score if has_ground_truth else None,
                            mean_confidence=mean_confidence,
                            class_confidences=class_confidences,
                            output_dir=viz_dir,
                            num_classes=num_classes
                        )
                    
                    # Save prediction if requested
                    if pred_dir:
                        pred_nifti = nib.Nifti1Image(preds[i].cpu().numpy(), np.eye(4))
                        nib.save(pred_nifti, os.path.join(pred_dir, f"{subject_id}_prediction.nii.gz"))
                    
                    samples_processed += 1
        
        return inference_metrics
    
    def _visualize_inference_result(self, image, ground_truth, prediction, confidence_map,
                                   probabilities, subject_id, dice_score, mean_confidence,
                                   class_confidences, output_dir, num_classes):
        """Create comprehensive visualization of inference results"""
        
        # Select middle slice
        depth = image.shape[0]
        mid_slice = depth // 2
        
        # Create figure
        has_gt = ground_truth is not None
        num_rows = 3 if has_gt else 2
        fig = plt.figure(figsize=(20, 8 * num_rows))
        gs = fig.add_gridspec(num_rows, 4, hspace=0.3, wspace=0.3)
        
        # Row 1: Image, Ground Truth/Prediction, Confidence Map, Histogram
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(image[mid_slice], cmap='gray')
        ax1.set_title('Input Image', fontsize=12)
        ax1.axis('off')
        
        ax2 = fig.add_subplot(gs[0, 1])
        if has_gt:
            ax2.imshow(ground_truth[mid_slice], cmap='tab10', vmin=0, vmax=num_classes-1)
            ax2.set_title('Ground Truth', fontsize=12)
        else:
            ax2.imshow(prediction[mid_slice], cmap='tab10', vmin=0, vmax=num_classes-1)
            ax2.set_title('Prediction', fontsize=12)
        ax2.axis('off')
        
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(confidence_map[mid_slice], cmap='viridis', vmin=0, vmax=1)
        ax3.set_title(f'Confidence Map (μ={mean_confidence:.3f})', fontsize=12)
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        ax4 = fig.add_subplot(gs[0, 3])
        ax4.hist(confidence_map.flatten(), bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax4.axvline(mean_confidence, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_confidence:.3f}')
        ax4.set_xlabel('Confidence', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.set_title('Confidence Distribution', fontsize=12)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Row 2: Overlays and class probabilities
        if has_gt:
            ax5 = fig.add_subplot(gs[1, 0])
            ax5.imshow(prediction[mid_slice], cmap='tab10', vmin=0, vmax=num_classes-1)
            title = f'Prediction (Dice: {dice_score:.3f})' if dice_score is not None else 'Prediction'
            ax5.set_title(title, fontsize=12)
            ax5.axis('off')
            
            ax6 = fig.add_subplot(gs[1, 1])
            ax6.imshow(image[mid_slice], cmap='gray')
            pred_overlay = np.ma.masked_where(prediction[mid_slice] == 0, prediction[mid_slice])
            ax6.imshow(pred_overlay, cmap='tab10', alpha=0.5, vmin=0, vmax=num_classes-1)
            ax6.set_title('Prediction Overlay', fontsize=12)
            ax6.axis('off')
            
            ax7 = fig.add_subplot(gs[1, 2])
            error_map = (ground_truth[mid_slice] != prediction[mid_slice]).astype(float)
            im7 = ax7.imshow(error_map, cmap='Reds', vmin=0, vmax=1)
            ax7.set_title('Error Map', fontsize=12)
            ax7.axis('off')
            plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
        
        # Class probability maps
        start_row = 2 if has_gt else 1
        for class_idx in range(min(4, num_classes)):
            ax = fig.add_subplot(gs[start_row, class_idx])
            class_prob_map = probabilities[class_idx, mid_slice]
            im = ax.imshow(class_prob_map, cmap='hot', vmin=0, vmax=1)
            conf_val = class_confidences.get(f"class_{class_idx}", 0.0)
            ax.set_title(f'Class {class_idx} Prob\n(conf={conf_val:.3f})', fontsize=10)
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Title
        title_text = f'Inference Results: {subject_id}\nMean Confidence: {mean_confidence:.4f}'
        if dice_score is not None:
            title_text = f'Inference Results: {subject_id}\nDice Score: {dice_score:.4f} | Mean Confidence: {mean_confidence:.4f}'
        fig.suptitle(title_text, fontsize=14, fontweight='bold')
        
        # Save
        output_path = os.path.join(output_dir, f"{subject_id}_inference.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"  Visualization saved: {output_path}")
    
    def _create_summary_report(self, inference_metrics, mean_confidence, mean_dice, num_samples):
        """Create summary report text"""
        summary = f"""
Inference Summary
=================
Total Samples Processed: {num_samples}
Mean Confidence: {mean_confidence:.4f}
"""
        if mean_dice is not None:
            summary += f"Mean Dice Score: {mean_dice:.4f}\n"
        
        summary += f"\nPer-Sample Results:\n"
        for m in inference_metrics:
            summary += f"\n{m.subject_id}:\n"
            if m.dice_score > 0:
                summary += f"  Dice: {m.dice_score:.4f}\n"
            summary += f"  Confidence: {m.mean_confidence:.4f} (min: {m.min_confidence:.4f}, max: {m.max_confidence:.4f})\n"
        
        return summary
