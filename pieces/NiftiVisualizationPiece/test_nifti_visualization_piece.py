from domino.testing import piece_dry_run


def test_nifti_visualization_piece():
    """Test NiftiVisualizationPiece with model validation only."""
    from pieces.NiftiVisualizationPiece.models import InputModel, SubjectInfo
    
    # Test model creation and validation
    test_subject = SubjectInfo(
        subject_id="test_subject",
        image_path="/home/shared_storage/medical_data/images/sub-test001.nii.gz",
        mask_path="/home/shared_storage/medical_data/masks/sub-test001.nii.gz"
    )
    input_data = InputModel(
        subject=test_subject,
        view_plane="axial",
        slice_index=16,
        show_mask_overlay=True,
        mask_alpha=0.5,
        color_map="gray",
        mask_color="red"
    )
    
    # Validate input model structure
    assert input_data.subject.subject_id == "test_subject"
    assert input_data.subject.image_path.endswith(".nii.gz")
    assert input_data.view_plane in ["axial", "sagittal", "coronal"]
    assert 0.0 <= input_data.mask_alpha <= 1.0
    assert 400 <= input_data.figure_width <= 2000
    assert 300 <= input_data.figure_height <= 2000
    
    print(f"✓ NiftiVisualizationPiece model validation passed")
