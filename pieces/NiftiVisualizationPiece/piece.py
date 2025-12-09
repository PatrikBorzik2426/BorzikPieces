from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import os
import glob
import base64
import traceback


class NiftiVisualizationPiece(BasePiece):
    """
    Standalone NIfTI visualization piece.
    Loads images directly from directories and creates a grid visualization.
    No upstream connection required - works independently.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        print("[NiftiVisualizationPiece] === STARTING ===")
        
        try:
            self.logger.info("=" * 50)
            self.logger.info("NiftiVisualizationPiece STARTED")
            self.logger.info("=" * 50)
        except:
            pass
        
        try:
            # Import libraries
            print("[NiftiVisualizationPiece] Importing libraries...")
            import numpy as np
            import nibabel as nib
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            print("[NiftiVisualizationPiece] Libraries imported OK")
            self.logger.info("Libraries imported successfully")
            
            # Log configuration
            self.logger.info(f"Images path: {input_data.images_path}")
            self.logger.info(f"Masks path: {input_data.masks_path}")
            self.logger.info(f"File pattern: {input_data.file_pattern}")
            self.logger.info(f"Max subjects: {input_data.max_subjects}")
            self.logger.info(f"View plane: {input_data.view_plane}")
            
            print(f"[NiftiVisualizationPiece] Images: {input_data.images_path}")
            print(f"[NiftiVisualizationPiece] Masks: {input_data.masks_path}")
            
            # Validate images directory
            if not os.path.isdir(input_data.images_path):
                raise ValueError(f"Images directory not found: {input_data.images_path}")
            
            # Find image files
            pattern = os.path.join(input_data.images_path, input_data.file_pattern)
            image_files = sorted(glob.glob(pattern))
            
            self.logger.info(f"Found {len(image_files)} image files")
            print(f"[NiftiVisualizationPiece] Found {len(image_files)} images")
            
            if not image_files:
                raise ValueError(f"No files matching '{input_data.file_pattern}' in {input_data.images_path}")
            
            # Limit to max_subjects
            image_files = image_files[:input_data.max_subjects]
            num_subjects = len(image_files)
            
            self.logger.info(f"Visualizing {num_subjects} subjects in vertical layout")
            print(f"[NiftiVisualizationPiece] Vertical layout: {num_subjects} subjects")
            
            # Create vertical layout - one subject per row, minimum 400x400 per image
            fig_width = 8  # Width in inches (800px at 100 dpi)
            height_per_subject = 4  # Height per subject in inches (400px at 100 dpi)
            fig_height = max(height_per_subject, num_subjects * height_per_subject)  # Minimum 400px height
            fig, axes = plt.subplots(num_subjects, 1, figsize=(fig_width, fig_height))
            
            # Make axes iterable
            if num_subjects == 1:
                axes = [axes]
            else:
                axes = list(axes)
            
            visualized_ids = []
            
            for idx, img_path in enumerate(image_files):
                ax = axes[idx]
                
                # Extract subject ID from filename
                filename = os.path.basename(img_path)
                subj_id = filename.replace('.nii.gz', '').replace('.nii', '')
                visualized_ids.append(subj_id)
                
                print(f"[NiftiVisualizationPiece] Processing {idx+1}/{num_subjects}: {subj_id}")
                self.logger.info(f"Processing {subj_id}")
                
                try:
                    # Load image
                    img_nii = nib.load(img_path)
                    img_data = img_nii.get_fdata()
                    self.logger.info(f"  Image shape: {img_data.shape}")
                    
                    # Get slice based on view plane
                    # Shape is typically (X, Y_slices, Z) where Y is the thin dimension (11-16 slices)
                    if input_data.view_plane == "sagittal":
                        mid = input_data.slice_index if input_data.slice_index else img_data.shape[0] // 2
                        slice_2d = img_data[mid, :, :]
                    elif input_data.view_plane == "coronal":
                        mid = input_data.slice_index if input_data.slice_index else img_data.shape[2] // 2
                        slice_2d = img_data[:, :, mid]
                    else:  # axial - slice through Y axis to get XxZ (512x512)
                        mid = input_data.slice_index if input_data.slice_index else img_data.shape[1] // 2
                        slice_2d = img_data[:, mid, :]
                    
                    # Display image with proper aspect ratio
                    ax.imshow(slice_2d.T, cmap=input_data.color_map, origin='lower', aspect='equal')
                    
                    # Try to find and overlay mask
                    has_mask = False
                    if input_data.masks_path and input_data.show_mask_overlay:
                        # Try to find matching mask file
                        mask_path = os.path.join(input_data.masks_path, filename)
                        if os.path.exists(mask_path):
                            msk_nii = nib.load(mask_path)
                            msk_data = msk_nii.get_fdata()
                            
                            if input_data.view_plane == "sagittal":
                                msk_slice = msk_data[mid, :, :]
                            elif input_data.view_plane == "coronal":
                                msk_slice = msk_data[:, :, mid]
                            else:  # axial - slice through Y axis
                                msk_slice = msk_data[:, mid, :]
                            
                            # Overlay mask with tab10 colormap and proper aspect ratio
                            ax.imshow(msk_slice.T, cmap='tab10', alpha=input_data.mask_alpha,
                                     origin='lower', vmin=0, vmax=5, aspect='equal')
                            has_mask = True
                            self.logger.info(f"  Mask overlaid")
                    
                    title = f"{subj_id}" + (" + mask" if has_mask else "")
                    ax.set_title(title, fontsize=9)
                    ax.axis('off')
                    
                    self.logger.info(f"  ✓ Done")
                    
                except Exception as e:
                    self.logger.error(f"Error with {subj_id}: {e}")
                    ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}",
                           ha='center', va='center', transform=ax.transAxes, color='red')
                    ax.axis('off')
            
            # Save figure
            plt.tight_layout()
            
            results_dir = getattr(self, 'results_path', '/tmp') or '/tmp'
            output_file = os.path.join(results_dir, f'nifti_grid_{num_subjects}subj.png')
            
            print(f"[NiftiVisualizationPiece] Saving to: {output_file}")
            self.logger.info(f"Saving to: {output_file}")
            
            plt.savefig(output_file, dpi=100, bbox_inches='tight', facecolor='white')
            plt.close()
            
            self.logger.info("Figure saved")
            print("[NiftiVisualizationPiece] Saved successfully")
            
            # Read for display
            with open(output_file, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            self.display_result = {
                "file_type": "png",
                "base64_content": img_b64
            }
            
            self.logger.info("=" * 50)
            self.logger.info(f"DONE: Visualized {num_subjects} subjects")
            self.logger.info("=" * 50)
            print("[NiftiVisualizationPiece] === COMPLETED ===")
            
            return OutputModel(
                num_subjects=num_subjects,
                subject_ids=visualized_ids,
                view_plane=input_data.view_plane,
                grid_size=f"{num_subjects}x1",
                visualization_summary=f"Visualized {num_subjects} subjects in vertical layout from {input_data.images_path}"
            )
            
        except Exception as e:
            print(f"[NiftiVisualizationPiece] ERROR: {e}")
            traceback.print_exc()
            self.logger.error(f"Error: {e}")
            raise
