from domino.testing import piece_dry_run


def test_nifti_visualization_piece():
    """Test NiftiVisualizationPiece with minimal inputs."""
    from .models import InputModel, SubjectInfo
    
    input_data = InputModel(
        subject=SubjectInfo(
            subject_id="test_subject",
            image_path="/home/shared_storage/medical_data/images/test_scan.nii.gz",
            mask_path="/home/shared_storage/medical_data/masks/test_scan.nii.gz"
        ),
        view_plane="axial",
        slice_index=None,
        show_mask_overlay=True,
        mask_alpha=0.5
    )
    
    output = piece_dry_run(
        piece_name="NiftiVisualizationPiece",
        input_data=input_data,
    )
    
    assert output is not None
    assert output.subject_id == "test_subject"
    assert output.view_plane == "axial"
    assert output.slice_index >= 0
    assert len(output.image_shape) == 3
    assert len(output.slice_shape) == 2
    print(f"✓ Visualization test passed: {output.subject_id}, slice {output.slice_index}")
