from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import traceback


class HelloWorldPiece(BasePiece):
    """A simple hello world piece that demonstrates basic text output."""

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info(f"Generating greeting for: {input_data.name}")

            # Create a simple greeting message
            message = f"Hello, {input_data.name}!"

            self.logger.info(f"Generated message: {message}")

            # Set display result for Domino UI
            self.format_display_result(message, input_data.name)

            return OutputModel(message=message)

        except Exception as e:
            self.logger.error(f"Error in HelloWorldPiece: {e}")
            print("[HelloWorldPiece] Exception in piece_function:")
            traceback.print_exc()
            raise

    def format_display_result(self, message: str, name: str):
        md_text = f"""
## Hello World Greeting

**Message:** {message}

**Generated for:** {name}
"""
        file_path = f"{self.results_path}/display_result.md"
        with open(file_path, "w") as f:
            f.write(md_text)
        self.display_result = {
            "file_type": "md",
            "file_path": file_path
        }