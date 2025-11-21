from domino.testing import piece_dry_run


def test_example_shapes_piece():
    input_data = dict(
        width=300,
        height=180,
        background_color="white",
        shape="circle",
        count=6,
        shape_size=30,
        color="blue",
        cols=3
    )

    output_data = piece_dry_run(
        "ExampleShapesPiece",
        input_data,
    )

    assert output_data["message"] is not None
    assert "image_path" in output_data
