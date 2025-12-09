from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import os
import base64
import traceback


class NiftiVisualizationPiece(BasePiece):
    """
    A piece that visualizes NIfTI medical imaging data.
    Simple implementation inspired by model.ipynb visualization pattern.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        # FIRST: Print immediately to catch any issues
        print("[NiftiVisualizationPiece] === STARTING ===")
        
        try:
            self.logger.info("=" * 50)
            self.logger.info("NiftiVisualizationPiece STARTED")
            self.logger.info("=" * 50)
        except Exception as log_err:
            print(f"[NiftiVisualizationPiece] Logger error: {log_err}")
        
        try:
            # Import inside function to catch import errors
            print("[NiftiVisualizationPiece] Importing libraries...")
            import numpy as np
            import nibabel as nib
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            print("[NiftiVisualizationPiece] Libraries imported OK")
            
            self.logger.info("Libraries imported successfully")
            
            # Log input info
            print(f"[NiftiVisualizationPiece] input_data type: {type(input_data)}")
            print(f"[NiftiVisualizationPiece] subjects type: {type(input_data.subjects)}")
            print(f"[NiftiVisualizationPiece] subjects count: {len(input_data.subjects) if input_data.subjects else 0}")
            
            self.logger.info(f"Input subjects count: {len(input_data.subjects) if input_data.subjects else 0}")
            
            # Get subjects - handle both list of dicts and list of SubjectInfo
            subjects = input_data.subjects
            if not subjects:
                self.logger.error("No subjects provided!")
                raise ValueError("No subjects provided in input")
            
            # Log first subject details
            first_subj = subjects[0]
            print(f"[NiftiVisualizationPiece] First subject type: {type(first_subj)}")
            self.logger.info(f"First subject type: {type(first_subj)}")
            
            # Access subject data - handle both dict and object
            if isinstance(first_subj, dict):
                print("[NiftiVisualizationPiece] Subjects are dicts")
                get_id = lambda s: s.get('subject_id', 'unknown')
                get_img = lambda s: s.get('image_path', '')
                get_msk = lambda s: s.get('mask_path')
            else:
                print("[NiftiVisualizationPiece] Subjects are objects")
                get_id = lambda s: s.subject_id
                get_img = lambda s: s.image_path
                get_msk = lambda s: s.mask_path
            
            # Limit subjects
            max_subj = min(input_data.max_subjects, len(subjects))
            subjects = subjects[:max_subj]
            num_subjects = len(subjects)
            
            self.logger.info(f"Will visualize {num_subjects} subjects")
            print(f"[NiftiVisualizationPiece] Will visualize {num_subjects} subjects")
            
            # Simple grid calculation
            cols = min(input_data.grid_columns, num_subjects)
            rows = (num_subjects + cols - 1) // cols
            
            self.logger.info(f"Grid: {rows}x{cols}")
            print(f"[NiftiVisualizationPiece] Grid: {rows}x{cols}")
            
            # Create figure - SIMPLE like in notebook
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            print(f"[NiftiVisualizationPiece] Figure created")
            
            # Make axes always 2D array
            if num_subjects == 1:
                axes = [[axes]]
            elif rows == 1:
                axes = [axes]
            elif cols == 1:
                axes = [[ax] for ax in axes]
            
            visualized_ids = []
            
            for idx, subj in enumerate(subjects):
                r = idx // cols
                c = idx % cols
                ax = axes[r][c]
                
                subj_id = get_id(subj)
                img_path = get_img(subj)
                msk_path = get_msk(subj)
                
                visualized_ids.append(subj_id)
                
                print(f"[NiftiVisualizationPiece] Processing {idx+1}/{num_subjects}: {subj_id}")
                self.logger.info(f"Processing {subj_id}")
                
                try:
                    # Check if image exists
                    if not os.path.exists(img_path):
                        ax.text(0.5, 0.5, f"Not found:\n{subj_id}", 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.set_title(subj_id)
                        ax.axis('off')
                        self.logger.warning(f"Image not found: {img_path}")
                        continue
                    
                    # Load image
                    img_nii = nib.load(img_path)
                    img_data = img_nii.get_fdata()
                    
                    self.logger.info(f"  Loaded: shape={img_data.shape}")
                    
                    # Get middle slice (axial by default)
                    mid_slice = img_data.shape[2] // 2
                    if input_data.view_plane == "sagittal":
                        mid_slice = img_data.shape[0] // 2
                        slice_2d = img_data[mid_slice, :, :]
                    elif input_data.view_plane == "coronal":
                        mid_slice = img_data.shape[1] // 2
                        slice_2d = img_data[:, mid_slice, :]
                    else:  # axial
                        slice_2d = img_data[:, :, mid_slice]
                    
                    if input_data.slice_index is not None:
                        mid_slice = input_data.slice_index
                    
                    # Simple display like in notebook
                    ax.imshow(slice_2d.T, cmap=input_data.color_map, origin='lower')
                    
                    # Overlay mask if available
                    if msk_path and input_data.show_mask_overlay and os.path.exists(msk_path):
                        msk_nii = nib.load(msk_path)
                        msk_data = msk_nii.get_fdata()
                        
                        if input_data.view_plane == "sagittal":
                            msk_slice = msk_data[mid_slice, :, :]
                        elif input_data.view_plane == "coronal":
                            msk_slice = msk_data[:, mid_slice, :]
                        else:
                            msk_slice = msk_data[:, :, mid_slice]
                        
                        # Simple mask overlay like notebook uses tab10 colormap
                        ax.imshow(msk_slice.T, cmap='tab10', alpha=input_data.mask_alpha, 
                                 origin='lower', vmin=0, vmax=5)
                        ax.set_title(f"{subj_id} + mask")
                    else:
                        ax.set_title(subj_id)
                    
                    ax.axis('off')
                    self.logger.info(f"  ✓ Visualized {subj_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error with {subj_id}: {e}")
                    ax.text(0.5, 0.5, f"Error:\n{str(e)[:30]}", 
                           ha='center', va='center', transform=ax.transAxes, color='red')
                    ax.axis('off')
            
            # Hide unused subplots
            for idx in range(num_subjects, rows * cols):
                r = idx // cols
                c = idx % cols
                axes[r][c].axis('off')
            
            # Save figure - SIMPLE like in notebook
            plt.tight_layout()
            
            # Get results path
            results_dir = getattr(self, 'results_path', '/tmp')
            if not results_dir:
                results_dir = '/tmp'
            
            output_file = os.path.join(results_dir, f'visualization_{num_subjects}subj.png')
            
            print(f"[NiftiVisualizationPiece] Saving to: {output_file}")
            self.logger.info(f"Saving to: {output_file}")
            
            plt.savefig(output_file)
            plt.close()
            
            print(f"[NiftiVisualizationPiece] Saved successfully")
            self.logger.info("Figure saved and closed")
            
            # Read for display
            with open(output_file, 'rb') as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            
            self.display_result = {
                "file_type": "png",
                "base64_content": img_b64
            }
            
            self.logger.info("=" * 50)
            self.logger.info(f"DONE: Visualized {num_subjects} subjects")
            self.logger.info("=" * 50)
            
            print(f"[NiftiVisualizationPiece] === COMPLETED ===")
            
            return OutputModel(
                num_subjects=num_subjects,
                subject_ids=visualized_ids,
                view_plane=input_data.view_plane,
                grid_size=f"{rows}x{cols}",
                visualization_summary=f"Visualized {num_subjects} subjects"
            )
            
        except Exception as e:
            print(f"[NiftiVisualizationPiece] ERROR: {e}")
            traceback.print_exc()
            self.logger.error(f"Error: {e}")
            raise
