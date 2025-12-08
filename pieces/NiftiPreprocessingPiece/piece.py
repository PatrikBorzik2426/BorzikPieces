from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, NormalizationMethod, PreprocessedSubject
import os
import json
import base64
import traceback
import numpy as np


class NiftiPreprocessingPiece(BasePiece):
    """
    A piece that preprocesses NIfTI medical imaging data.
    
    Performs normalization, optional resizing, and converts data to 
    NumPy format for efficient loading during training. Supports
    z-score, min-max, and percentile-based normalization.
    """

    def normalize_volume(self, volume: np.ndarray, method: NormalizationMethod, 
                         lower_pct: float = 1.0, upper_pct: float = 99.0) -> np.ndarray:
        """Apply normalization to volume"""
        
        if method == NormalizationMethod.ZSCORE:
            mean = volume.mean()
            std = volume.std() + 1e-8
            return (volume - mean) / std
            
        elif method == NormalizationMethod.MINMAX:
            vmin, vmax = volume.min(), volume.max()
            if vmax - vmin > 1e-8:
                return (volume - vmin) / (vmax - vmin)
            return volume - vmin
            
        elif method == NormalizationMethod.PERCENTILE:
            lower = np.percentile(volume, lower_pct)
            upper = np.percentile(volume, upper_pct)
            volume = np.clip(volume, lower, upper)
            if upper - lower > 1e-8:
                return (volume - lower) / (upper - lower)
            return volume - lower
            
        else:  # NONE
            return volume

    def resize_volume(self, volume: np.ndarray, target_shape: list) -> np.ndarray:
        """Resize volume to target shape using scipy zoom"""
        try:
            from scipy.ndimage import zoom
            
            current_shape = volume.shape
            zoom_factors = [t / c for t, c in zip(target_shape, current_shape)]
            return zoom(volume, zoom_factors, order=1)
        except ImportError:
            self.logger.warning("scipy not available, skipping resize")
            return volume

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            import nibabel as nib
            
            print("=" * 60)
            print("Starting NiftiPreprocessingPiece execution")
            print("=" * 60)
            self.logger.info("=" * 60)
            self.logger.info("Starting NiftiPreprocessingPiece execution")
            self.logger.info("=" * 60)
            
            subjects = input_data.subjects
            output_dir = input_data.output_dir
            normalization = input_data.normalization
            save_numpy = input_data.save_as_numpy
            target_shape = input_data.target_shape
            
            print(f"Input configuration:")
            print(f"  - Number of subjects: {len(subjects)}")
            print(f"  - Normalization method: {normalization.value}")
            print(f"  - Output directory: {output_dir}")
            print(f"  - Save as NumPy: {save_numpy}")
            print(f"  - Target shape: {target_shape}")
            self.logger.info(f"Input configuration:")
            self.logger.info(f"  - Number of subjects: {len(subjects)}")
            self.logger.info(f"  - Normalization method: {normalization.value}")
            self.logger.info(f"  - Output directory: {output_dir}")
            self.logger.info(f"  - Save as NumPy: {save_numpy}")
            self.logger.info(f"  - Target shape: {target_shape}")
            self.logger.info(f"  - Lower percentile: {input_data.lower_percentile}")
            self.logger.info(f"  - Upper percentile: {input_data.upper_percentile}")
            
            # Create output directories
            images_out = os.path.join(output_dir, "images")
            masks_out = os.path.join(output_dir, "masks")
            os.makedirs(images_out, exist_ok=True)
            os.makedirs(masks_out, exist_ok=True)
            
            preprocessed_subjects = []
            num_failed = 0
            
            for idx, subject in enumerate(subjects):
                try:
                    self.logger.info(f"Processing subject {idx+1}/{len(subjects)}: {subject.subject_id}")
                    
                    # Load image
                    self.logger.debug(f"  Loading image from: {subject.image_path}")
                    img_nii = nib.load(subject.image_path)
                    img_data = img_nii.get_fdata().astype(np.float32)
                    original_shape = list(img_data.shape)
                    self.logger.debug(f"  Original shape: {original_shape}")
                    
                    # Compute original stats
                    original_stats = {
                        "min": float(img_data.min()),
                        "max": float(img_data.max()),
                        "mean": float(img_data.mean()),
                        "std": float(img_data.std())
                    }
                    
                    # Normalize
                    self.logger.debug(f"  Applying {normalization.value} normalization")
                    img_normalized = self.normalize_volume(
                        img_data, 
                        normalization,
                        input_data.lower_percentile,
                        input_data.upper_percentile
                    )
                    
                    # Resize if specified
                    if target_shape is not None:
                        self.logger.debug(f"  Resizing to target shape: {target_shape}")
                        img_normalized = self.resize_volume(img_normalized, target_shape)
                    
                    preprocessed_shape = list(img_normalized.shape)
                    self.logger.debug(f"  Preprocessed shape: {preprocessed_shape}")
                    
                    # Compute normalized stats
                    normalized_stats = {
                        "min": float(img_normalized.min()),
                        "max": float(img_normalized.max()),
                        "mean": float(img_normalized.mean()),
                        "std": float(img_normalized.std())
                    }
                    
                    # Save preprocessed image
                    if save_numpy:
                        img_out_path = os.path.join(images_out, f"{subject.subject_id}.npy")
                        np.save(img_out_path, img_normalized)
                        self.logger.debug(f"  Saved image as NumPy: {img_out_path}")
                    else:
                        img_out_path = os.path.join(images_out, f"{subject.subject_id}.nii.gz")
                        out_nii = nib.Nifti1Image(img_normalized, img_nii.affine, img_nii.header)
                        nib.save(out_nii, img_out_path)
                        self.logger.debug(f"  Saved image as NIfTI: {img_out_path}")
                    
                    # Process mask if available
                    mask_out_path = None
                    if subject.mask_path and os.path.exists(subject.mask_path):
                        self.logger.debug(f"  Processing mask from: {subject.mask_path}")
                        mask_nii = nib.load(subject.mask_path)
                        mask_data = mask_nii.get_fdata().astype(np.int64)
                        
                        # Resize mask if needed (using nearest neighbor)
                        if target_shape is not None:
                            from scipy.ndimage import zoom
                            zoom_factors = [t / c for t, c in zip(target_shape, mask_data.shape)]
                            mask_data = zoom(mask_data, zoom_factors, order=0)  # order=0 for nearest
                        
                        if save_numpy:
                            mask_out_path = os.path.join(masks_out, f"{subject.subject_id}.npy")
                            np.save(mask_out_path, mask_data)
                        else:
                            mask_out_path = os.path.join(masks_out, f"{subject.subject_id}.nii.gz")
                            out_mask_nii = nib.Nifti1Image(mask_data, mask_nii.affine, mask_nii.header)
                            nib.save(out_mask_nii, mask_out_path)
                    
                    preprocessed_subjects.append(PreprocessedSubject(
                        subject_id=subject.subject_id,
                        original_image_path=subject.image_path,
                        preprocessed_image_path=img_out_path,
                        preprocessed_mask_path=mask_out_path,
                        original_shape=original_shape,
                        preprocessed_shape=preprocessed_shape,
                        image_stats={
                            "original": original_stats,
                            "normalized": normalized_stats
                        }
                    ))
                    
                except Exception as e:
                    self.logger.error(f"Failed to process {subject.subject_id}: {e}")
                    num_failed += 1
                    continue
            
            preprocessing_config = {
                "normalization_method": normalization.value,
                "lower_percentile": input_data.lower_percentile,
                "upper_percentile": input_data.upper_percentile,
                "save_as_numpy": save_numpy,
                "target_shape": target_shape
            }
            
            # Save config to output dir
            config_path = os.path.join(output_dir, "preprocessing_config.json")
            with open(config_path, 'w') as f:
                json.dump(preprocessing_config, f, indent=2)
            
            self.logger.info(f"Preprocessing complete:")
            self.logger.info(f"  - Succeeded: {len(preprocessed_subjects)} subjects")
            self.logger.info(f"  - Failed: {num_failed} subjects")
            self.logger.info(f"  - Success rate: {len(preprocessed_subjects)/(len(preprocessed_subjects)+num_failed)*100:.1f}%")
            self.logger.info("=" * 60)
            self.logger.info("NiftiPreprocessingPiece execution completed")
            self.logger.info("=" * 60)
            
            # Set display result
            summary = {
                "processed": len(preprocessed_subjects),
                "failed": num_failed,
                "output_dir": output_dir,
                "normalization": normalization.value
            }
            summary_text = json.dumps(summary, indent=2)
            base64_content = base64.b64encode(summary_text.encode("utf-8")).decode("utf-8")
            self.display_result = {
                "file_type": "json",
                "base64_content": base64_content
            }
            
            return OutputModel(
                preprocessed_subjects=preprocessed_subjects,
                output_dir=output_dir,
                num_processed=len(preprocessed_subjects),
                num_failed=num_failed,
                preprocessing_config=preprocessing_config
            )
            
        except Exception as e:
            self.logger.error(f"Error in NiftiPreprocessingPiece: {e}")
            print("[NiftiPreprocessingPiece] Exception in piece_function:")
            traceback.print_exc()
            raise
