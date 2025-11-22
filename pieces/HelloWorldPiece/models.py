from pydantic import BaseModel, Field


class InputModel(BaseModel):
    """
    Hello World Piece Input Model
    """
    name: str = Field(
        description="Name to greet",
        default="World"
    )


class OutputModel(BaseModel):
    """
    Hello World Piece Output Model
    """
    message: str = Field(
        description="Greeting message"
    )