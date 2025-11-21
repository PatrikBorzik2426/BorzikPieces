from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    Input model for the ExampleShapesPiece
    """

    width: int = Field(description="Image width in pixels", default=400)
    height: int = Field(description="Image height in pixels", default=200)
    background_color: str = Field(description="Background color (name or hex)", default="white")
    shape: str = Field(description="Shape to draw: circle, square, triangle or random", default="circle")
    count: int = Field(description="Number of shapes to draw", default=6)
    shape_size: int = Field(description="Size of each shape in pixels", default=40)
    color: str = Field(description="Fill color for shapes (name or hex)", default="blue")
    cols: int = Field(description="Number of columns to arrange shapes in", default=3)


class OutputModel(BaseModel):
    """
    Output model for the ExampleShapesPiece
    """

    message: str = Field(description="Human readable message about the run")
    image_path: str = Field(description="Path to the generated image file")
