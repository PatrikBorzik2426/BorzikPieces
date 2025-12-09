from domino.testing import piece_dry_run


def test_nifti_visualization_piece():
    """Test NiftiVisualizationPiece with model validation for grid mode."""
    from pieces.NiftiVisualizationPiece.models import InputModel, SubjectInfo
    
    # Test model creation and validation with multiple subjects
    test_subjects = [
        SubjectInfo(
            subject_id=f"test_subject_{i}",
            image_path=f"/home/shared_storage/medical_data/images/sub-test{i:03d}.nii.gz",
            mask_path=f"/home/shared_storage/medical_data/masks/sub-test{i:03d}.nii.gz"
        )
        for i in range(1, 6)
    ]
    
    input_data = InputModel(
        subjects=test_subjects,
        max_subjects=10,
        view_plane="axial",
        slice_index=16,
        show_mask_overlay=True,
        mask_alpha=0.5,
        color_map="gray",
        mask_color="red",
        grid_columns=5
    )
    
    # Validate input model structure
    assert len(input_data.subjects) == 5
    assert input_data.subjects[0].subject_id == "test_subject_1"
    assert input_data.subjects[0].image_path.endswith(".nii.gz")
    assert input_data.view_plane in ["axial", "sagittal", "coronal"]
    assert 0.0 <= input_data.mask_alpha <= 1.0
    assert 1 <= input_data.max_subjects <= 20
    assert 1 <= input_data.grid_columns <= 10
    
    print(f"✓ NiftiVisualizationPiece model validation passed")
