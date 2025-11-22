from domino.testing import piece_dry_run
from pieces.HelloWorldPiece.models import InputModel


def test_hello_world():
    # Test with default name
    inp = InputModel()
    input_data = inp.model_dump() if hasattr(inp, "model_dump") else inp.dict()
    output = piece_dry_run("HelloWorldPiece", input_data)

    assert isinstance(output, dict)
    assert output.get("message") == "Hello, World!"

    # Test with custom name
    inp2 = InputModel(name="Alice")
    input_data2 = inp2.model_dump() if hasattr(inp2, "model_dump") else inp2.dict()
    output2 = piece_dry_run("HelloWorldPiece", input_data2)

    assert isinstance(output2, dict)
    assert output2.get("message") == "Hello, Alice!"