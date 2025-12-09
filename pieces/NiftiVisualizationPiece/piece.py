from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import os
import base64
import traceback
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class NiftiVisualizationPiece(BasePiece):
    """
    A piece that visualizes NIfTI medical imaging data in a grid layout.
    
    This piece loads multiple 3D NIfTI volumes and creates a grid of 2D slice 
    visualizations with optional mask overlays. It visualizes up to the first 
    10 subjects (or max_subjects) from the input list.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            import nibabel as nib
            
            self.logger.info("=" * 60)
            self.logger.info("Starting NiftiVisualizationPiece (Grid Mode)")
            self.logger.info("=" * 60)
            
            # Validate that subjects are provided
            if not input_data.subjects or len(input_data.subjects) == 0:
                raise ValueError(
                    "No subjects provided. This piece must be connected to an upstream piece "
                    "(DataLoader or DataSplit) that provides List[SubjectInfo]."
                )
            
            # Limit to max_subjects
            subjects_to_visualize = input_data.subjects[:input_data.max_subjects]
            num_subjects = len(subjects_to_visualize)
            
            self.logger.info(f"Input configuration:")
            self.logger.info(f"  - Total subjects available: {len(input_data.subjects)}")
            self.logger.info(f"  - Subjects to visualize: {num_subjects}")
            self.logger.info(f"  - View plane: {input_data.view_plane}")
            self.logger.info(f"  - Show mask overlay: {input_data.show_mask_overlay}")
            self.logger.info(f"  - Grid columns: {input_data.grid_columns}")
            
            # Calculate grid dimensions
            cols = input_data.grid_columns
            rows = (num_subjects + cols - 1) // cols  # Ceiling division
            
            self.logger.info(f"Creating {rows}x{cols} grid visualization...")
            
            # Create figure with subplots
            fig_width = cols * 4  # 4 inches per column
            fig_height = rows * 3  # 3 inches per row
            fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), dpi=100)
            
            # Handle single subplot case
            if num_subjects == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes.reshape(1, -1)
            elif cols == 1:
                axes = axes.reshape(-1, 1)
            
            visualized_ids = []
            
            # Process each subject
            for idx, subject in enumerate(subjects_to_visualize):
                row = idx // cols
                col = idx % cols
                ax = axes[row, col]
                
                subject_id = subject.subject_id
                image_path = subject.image_path
                mask_path = subject.mask_path
                
                visualized_ids.append(subject_id)
                
                try:
                    # Load image
                    if not os.path.exists(image_path):
                        self.logger.warning(f"Image not found for {subject_id}: {image_path}")
                        ax.text(0.5, 0.5, f"Image not found\\n{subject_id}", 
                               ha='center', va='center', transform=ax.transAxes)
                        ax.axis('off')
                        continue
                    
                    img_nii = nib.load(image_path)
                    img_data = img_nii.get_fdata().astype(np.float32)
                    
                    # Load mask if available
                    mask_data = None
                    has_mask = False
                    if mask_path and input_data.show_mask_overlay and os.path.exists(mask_path):
                        mask_nii = nib.load(mask_path)
                        mask_data = mask_nii.get_fdata().astype(np.int32)
                        has_mask = True
                    
                    # Extract slice based on view plane
                    slice_index = input_data.slice_index
                    view_plane = input_data.view_plane
                    
                    if view_plane == "axial":
                        if slice_index is None:
                            slice_index = img_data.shape[2] // 2
                        img_slice = img_data[:, :, slice_index]
                        mask_slice = mask_data[:, :, slice_index] if has_mask else None
                    elif view_plane == "sagittal":
                        if slice_index is None:
                            slice_index = img_data.shape[0] // 2
                        img_slice = img_data[slice_index, :, :]
                        mask_slice = mask_data[slice_index, :, :] if has_mask else None
                    else:  # coronal
                        if slice_index is None:
                            slice_index = img_data.shape[1] // 2
                        img_slice = img_data[:, slice_index, :]
                        mask_slice = mask_data[:, slice_index, :] if has_mask else None
                    
                    # Normalize image
                    img_slice_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
                    
                    # Display image
                    ax.imshow(img_slice_norm.T, cmap=input_data.color_map, origin='lower', aspect='auto')
                    
                    # Overlay mask
                    if has_mask and mask_slice is not None:
                        mask_colored = np.zeros((*mask_slice.shape, 4))
                        mask_colored[mask_slice > 0] = matplotlib.colors.to_rgba(
                            input_data.mask_color, 
                            alpha=input_data.mask_alpha
                        )
                        ax.imshow(mask_colored.T, origin='lower', aspect='auto')
                    
                    # Add title
                    title = f"{subject_id}"
                    if has_mask:
                        title += " + mask"
                    ax.set_title(title, fontsize=9)
                    ax.axis('off')
                    
                    self.logger.info(f"  [{idx+1}/{num_subjects}] Visualized {subject_id}")
                    
                except Exception as e:
                    self.logger.error(f"Error visualizing {subject_id}: {e}")
                    ax.text(0.5, 0.5, f"Error\\n{subject_id}\\n{str(e)[:30]}", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=8)
                    ax.axis('off')
            
            # Hide unused subplots
            for idx in range(num_subjects, rows * cols):
                row = idx // cols
                col = idx % cols
                axes[row, col].axis('off')
            
            # Add main title
            fig.suptitle(f"NIfTI Visualization - {input_data.view_plane.capitalize()} View", 
                        fontsize=14, fontweight='bold', y=0.995)
            
            plt.tight_layout()
            
            # Save figure
            results_dir = getattr(self, "results_path", "/tmp")
            output_filename = f"grid_visualization_{input_data.view_plane}_{num_subjects}subjects.png"
            output_path = os.path.join(results_dir, output_filename)
            
            plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
            self.logger.info(f"Grid visualization saved to: {output_path}")
            
            # Read image for display_result
            with open(output_path, 'rb') as f:
                image_bytes = f.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Set display result
            self.display_result = {
                "file_type": "png",
                "base64_content": image_base64
            }
            
            plt.close(fig)
            
            self.logger.info("=" * 60)
            self.logger.info(f"Grid visualization completed: {num_subjects} subjects in {rows}x{cols} grid")
            self.logger.info("=" * 60)
            
            return OutputModel(
                num_subjects=num_subjects,
                subject_ids=visualized_ids,
                view_plane=input_data.view_plane,
                grid_size=f"{rows}x{cols}",
                visualization_summary=f"Visualized {num_subjects} subjects in {rows}x{cols} grid"
            )
            
        except Exception as e:
            self.logger.error(f"Error in NiftiVisualizationPiece: {e}")
            print("[NiftiVisualizationPiece] Exception in piece_function:")
            traceback.print_exc()
            raise
