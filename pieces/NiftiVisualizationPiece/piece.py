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
    A piece that visualizes NIfTI medical imaging data with optional mask overlay.
    
    This piece loads a 3D NIfTI volume and creates 2D slice visualizations
    in axial, sagittal, or coronal planes. It can overlay segmentation masks
    with customizable transparency and colors.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            import nibabel as nib
            
            self.logger.info("=" * 60)
            self.logger.info("Starting NiftiVisualizationPiece execution")
            self.logger.info("=" * 60)
            
            subject = input_data.subject
            image_path = subject.image_path
            mask_path = subject.mask_path
            subject_id = subject.subject_id
            slice_index = input_data.slice_index
            view_plane = input_data.view_plane
            show_mask = input_data.show_mask_overlay
            
            self.logger.info(f"Input configuration:")
            self.logger.info(f"  - Subject ID: {subject_id}")
            self.logger.info(f"  - Image path: {image_path}")
            self.logger.info(f"  - Mask path: {mask_path if mask_path else 'None'}")
            self.logger.info(f"  - View plane: {view_plane}")
            self.logger.info(f"  - Slice index: {slice_index if slice_index is not None else 'Auto (middle)'}")
            
            # Load image
            self.logger.info(f"Loading NIfTI image...")
            if not os.path.exists(image_path):
                raise ValueError(f"Image file not found: {image_path}")
            
            img_nii = nib.load(image_path)
            img_data = img_nii.get_fdata().astype(np.float32)
            image_shape = list(img_data.shape)
            self.logger.info(f"Image loaded successfully: shape {image_shape}")
            
            # Load mask if available
            mask_data = None
            has_mask = False
            if mask_path and show_mask:
                if os.path.exists(mask_path):
                    self.logger.info(f"Loading mask...")
                    mask_nii = nib.load(mask_path)
                    mask_data = mask_nii.get_fdata().astype(np.int32)
                    has_mask = True
                    self.logger.info(f"Mask loaded successfully: shape {list(mask_data.shape)}")
                else:
                    self.logger.warning(f"Mask file not found: {mask_path}")
            
            # Select slice based on view plane
            if view_plane == "axial":
                max_slices = img_data.shape[2]
                if slice_index is None:
                    slice_index = max_slices // 2
                img_slice = img_data[:, :, slice_index]
                mask_slice = mask_data[:, :, slice_index] if has_mask else None
                axis_labels = ("Left → Right", "Posterior → Anterior")
            elif view_plane == "sagittal":
                max_slices = img_data.shape[0]
                if slice_index is None:
                    slice_index = max_slices // 2
                img_slice = img_data[slice_index, :, :]
                mask_slice = mask_data[slice_index, :, :] if has_mask else None
                axis_labels = ("Posterior → Anterior", "Inferior → Superior")
            else:  # coronal
                max_slices = img_data.shape[1]
                if slice_index is None:
                    slice_index = max_slices // 2
                img_slice = img_data[:, slice_index, :]
                mask_slice = mask_data[:, slice_index, :] if has_mask else None
                axis_labels = ("Left → Right", "Inferior → Superior")
            
            slice_shape = list(img_slice.shape)
            self.logger.info(f"Extracted {view_plane} slice {slice_index}/{max_slices-1}, shape: {slice_shape}")
            
            # Normalize image for display
            img_slice_norm = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min() + 1e-8)
            
            # Create visualization
            self.logger.info("Creating visualization...")
            dpi = 100
            fig_width = input_data.figure_width / dpi
            fig_height = input_data.figure_height / dpi
            
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            
            # Display image
            ax.imshow(img_slice_norm.T, cmap=input_data.color_map, origin='lower', aspect='auto')
            
            # Overlay mask if available
            if has_mask and mask_slice is not None:
                # Create colored mask overlay
                mask_colored = np.zeros((*mask_slice.shape, 4))
                mask_colored[mask_slice > 0] = matplotlib.colors.to_rgba(
                    input_data.mask_color, 
                    alpha=input_data.mask_alpha
                )
                ax.imshow(mask_colored.T, origin='lower', aspect='auto')
                self.logger.info(f"Mask overlay applied with alpha={input_data.mask_alpha}")
            
            # Add labels and title
            ax.set_xlabel(axis_labels[0], fontsize=10)
            ax.set_ylabel(axis_labels[1], fontsize=10)
            title = f"{subject_id} - {view_plane.capitalize()} view (slice {slice_index})"
            if has_mask:
                title += " with mask overlay"
            ax.set_title(title, fontsize=12, fontweight='bold')
            
            # Add colorbar for intensity
            cbar = plt.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('Normalized Intensity', fontsize=9)
            
            plt.tight_layout()
            
            # Save figure
            results_dir = getattr(self, "results_path", "/tmp")
            output_filename = f"{subject_id}_{view_plane}_slice{slice_index}.png"
            output_path = os.path.join(results_dir, output_filename)
            
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            self.logger.info(f"Visualization saved to: {output_path}")
            
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
            self.logger.info("NiftiVisualizationPiece execution completed successfully")
            self.logger.info("=" * 60)
            
            return OutputModel(
                subject_id=subject_id,
                slice_index=slice_index,
                view_plane=view_plane,
                image_shape=image_shape,
                slice_shape=slice_shape,
                has_mask=has_mask,
                visualization_path=output_path
            )
            
        except Exception as e:
            self.logger.error(f"Error in NiftiVisualizationPiece: {e}")
            print("[NiftiVisualizationPiece] Exception in piece_function:")
            traceback.print_exc()
            raise
