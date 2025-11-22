from domino.testing import piece_dry_run
from pieces.GenerativeShapesPiece.models import InputModel


def test_generate_shapes():
    inp = InputModel()
    # Ensure we pass a JSON-serializable dict to the testing HTTP client
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    output = piece_dry_run("GenerativeShapesPiece", input_data)
    # output should be a dict-like model dump
    assert isinstance(output, dict)
    assert output.get("file_path") == "shapes.png"
    assert output.get("message") == "Shapes generated"
