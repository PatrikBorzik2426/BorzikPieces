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
            print("=" * 60)
            print("Starting NiftiDataLoaderPiece execution")
            print("=" * 60)
            self.logger.info("=" * 60)
            self.logger.info("Starting NiftiDataLoaderPiece execution")
            self.logger.info("=" * 60)
            
            images_path = input_data.images_path
            masks_path = input_data.masks_path
            file_pattern = input_data.file_pattern
            
            print(f"Input configuration:")
            print(f"  - Images path: {images_path}")
            print(f"  - Masks path: {masks_path if masks_path else 'Not provided'}")
            print(f"  - File pattern: {file_pattern}")
            print(f"Scanning for NIfTI files...")
            self.logger.info(f"Input configuration:")
            self.logger.info(f"  - Images path: {images_path}")
            self.logger.info(f"  - Masks path: {masks_path if masks_path else 'Not provided'}")
            self.logger.info(f"  - File pattern: {file_pattern}")
            self.logger.info(f"Scanning for NIfTI files...")
            
            # Validate images directory exists
            if not os.path.exists(images_path):
                self.logger.error(f"Images directory not found: {images_path}")
                raise ValueError(f"Images directory not found: {images_path}")
            
            self.logger.info(f"Images directory validated: {images_path}")
            
            # Find all image files
            search_pattern = os.path.join(images_path, file_pattern)
            self.logger.debug(f"Searching with pattern: {search_pattern}")
            image_files = sorted(glob.glob(search_pattern))
            
            if len(image_files) == 0:
                self.logger.error(f"No NIfTI files found matching pattern '{file_pattern}'")
                raise ValueError(f"No NIfTI files found matching pattern '{file_pattern}' in {images_path}")
            
            self.logger.info(f"Found {len(image_files)} image files")
            if len(image_files) <= 5:
                for f in image_files:
                    self.logger.debug(f"  - {os.path.basename(f)}")
            else:
                for f in image_files[:3]:
                    self.logger.debug(f"  - {os.path.basename(f)}")
                self.logger.debug(f"  ... and {len(image_files) - 3} more")
            
            subjects = []
            masks_found = 0
            masks_missing = 0
            
            self.logger.info("Processing image files and searching for masks...")
            
            for idx, img_path in enumerate(image_files):
                filename = os.path.basename(img_path)
                # Extract subject ID (remove .nii.gz extension)
                subject_id = filename.replace(".nii.gz", "").replace(".nii", "")
                
                if (idx + 1) % 10 == 0 or idx == 0:
                    self.logger.info(f"Processing subject {idx+1}/{len(image_files)}: {subject_id}")
                
                mask_path = None
                if masks_path and os.path.exists(masks_path):
                    # Try to find corresponding mask
                    potential_mask = os.path.join(masks_path, filename)
                    if os.path.exists(potential_mask):
                        mask_path = potential_mask
                        masks_found += 1
                        self.logger.debug(f"  Found mask for {subject_id}")
                    else:
                        masks_missing += 1
                        self.logger.debug(f"  No mask found for {subject_id}")
                
                subjects.append(SubjectInfo(
                    subject_id=subject_id,
                    image_path=img_path,
                    mask_path=mask_path
                ))
            
            # Count subjects with masks
            subjects_with_masks = sum(1 for s in subjects if s.mask_path is not None)
            
            self.logger.info("Data loading summary:")
            self.logger.info(f"  - Total subjects: {len(subjects)}")
            self.logger.info(f"  - With masks: {subjects_with_masks} ({subjects_with_masks/len(subjects)*100:.1f}%)")
            self.logger.info(f"  - Without masks: {len(subjects) - subjects_with_masks} ({(len(subjects) - subjects_with_masks)/len(subjects)*100:.1f}%)")
            
            summary = {
                "total_subjects": len(subjects),
                "subjects_with_masks": subjects_with_masks,
                "subjects_without_masks": len(subjects) - subjects_with_masks,
                "images_directory": images_path,
                "masks_directory": masks_path,
                "subject_ids": [s.subject_id for s in subjects[:10]]  # First 10 for preview
            }
            
            self.logger.info(f"Sample subject IDs (first 5): {[s.subject_id for s in subjects[:5]]}")
            self.logger.info("=" * 60)
            self.logger.info(f"NiftiDataLoaderPiece execution completed successfully")
            self.logger.info("=" * 60)
            
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
