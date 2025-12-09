from domino.testing import piece_dry_run


def test_nifti_visualization_piece():
    """Test NiftiVisualizationPiece with model validation for standalone mode."""
    from pieces.NiftiVisualizationPiece.models import InputModel
    
    # Test model creation with direct directory paths (standalone mode)
    input_data = InputModel(
        images_path="/home/shared_storage/medical_data/images",
        masks_path="/home/shared_storage/medical_data/masks",
        file_pattern="*.nii.gz",
        max_subjects=6,
        view_plane="axial",
        slice_index=16,
        show_mask_overlay=True,
        mask_alpha=0.5,
        color_map="gray",
        grid_columns=3
    )
    
    # Validate input model structure
    assert input_data.images_path == "/home/shared_storage/medical_data/images"
    assert input_data.masks_path == "/home/shared_storage/medical_data/masks"
    assert input_data.file_pattern == "*.nii.gz"
    assert input_data.view_plane in ["axial", "sagittal", "coronal"]
    assert 0.0 <= input_data.mask_alpha <= 1.0
    assert 1 <= input_data.max_subjects <= 20
    assert 1 <= input_data.grid_columns <= 10
    
    print(f"✓ NiftiVisualizationPiece model validation passed")
