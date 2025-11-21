from pydantic import BaseModel, Field
from typing import Literal, Optional


class InputModel(BaseModel):
    width: int = Field(256, ge=1)
    height: int = Field(256, ge=1)
    background_color: str = "#FFFFFF"
    shape: Literal["circle", "square", "triangle"] = "circle"
    count: int = Field(5, ge=1, le=200)
    shape_size: int = Field(30, ge=1)
    color: str = "#FF0000"
    seed: Optional[int] = 42


class OutputModel(BaseModel):
    message: str
    file_path: str
