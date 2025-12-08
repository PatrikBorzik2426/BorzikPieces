from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SubjectInfo
import os
import glob
import json
import base64
import traceback


class NiftiDataLoaderPiece(BasePiece):
    """
    A piece that discovers and loads NIfTI medical imaging data.
    
    This piece scans directories for NIfTI files (.nii.gz) and pairs
    images with their corresponding segmentation masks based on filename.
    Useful for preparing medical imaging datasets for training or inference.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info(f"Scanning for NIfTI files in: {input_data.images_path}")
            
            images_path = input_data.images_path
            masks_path = input_data.masks_path
            file_pattern = input_data.file_pattern
            
            # Validate images directory exists
            if not os.path.exists(images_path):
                raise ValueError(f"Images directory not found: {images_path}")
            
            # Find all image files
            image_files = sorted(glob.glob(os.path.join(images_path, file_pattern)))
            
            if len(image_files) == 0:
                raise ValueError(f"No NIfTI files found matching pattern '{file_pattern}' in {images_path}")
            
            self.logger.info(f"Found {len(image_files)} image files")
            
            subjects = []
            
            for img_path in image_files:
                filename = os.path.basename(img_path)
                # Extract subject ID (remove .nii.gz extension)
                subject_id = filename.replace(".nii.gz", "").replace(".nii", "")
                
                mask_path = None
                if masks_path and os.path.exists(masks_path):
                    # Try to find corresponding mask
                    potential_mask = os.path.join(masks_path, filename)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        self.logger.debug(f"Found mask for {subject_id}")
                    else:
                        self.logger.warning(f"No mask found for {subject_id}")
                
                subjects.append(SubjectInfo(
                    subject_id=subject_id,
                    image_path=img_path,
                    mask_path=mask_path
                ))
            
            # Count subjects with masks
            subjects_with_masks = sum(1 for s in subjects if s.mask_path is not None)
            
            summary = {
                "total_subjects": len(subjects),
                "subjects_with_masks": subjects_with_masks,
                "subjects_without_masks": len(subjects) - subjects_with_masks,
                "images_directory": images_path,
                "masks_directory": masks_path,
                "subject_ids": [s.subject_id for s in subjects[:10]]  # First 10 for preview
            }
            
            self.logger.info(f"Successfully loaded {len(subjects)} subjects ({subjects_with_masks} with masks)")
            
            # Set display result for Domino UI
            summary_text = json.dumps(summary, indent=2)
            base64_content = base64.b64encode(summary_text.encode("utf-8")).decode("utf-8")
            self.display_result = {
                "file_type": "json",
                "base64_content": base64_content
            }
            
            return OutputModel(
                subjects=subjects,
                num_subjects=len(subjects),
                images_dir=images_path,
                masks_dir=masks_path
            )
            
        except Exception as e:
            self.logger.error(f"Error in NiftiDataLoaderPiece: {e}")
            print("[NiftiDataLoaderPiece] Exception in piece_function:")
            traceback.print_exc()
            raise
