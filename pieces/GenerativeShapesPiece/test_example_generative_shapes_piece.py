from domino.testing import piece_dry_run
from .models import InputModel


def test_generate_shapes():
    inp = InputModel()
    res = piece_dry_run("GenerativeShapesPiece", inp)
    # piece_dry_run returns a dict-like model dump
    assert isinstance(res, dict)
    assert res.get("file_path") == "shapes.png"
