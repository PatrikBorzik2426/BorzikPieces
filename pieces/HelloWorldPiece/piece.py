from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import traceback
import base64


class HelloWorldPiece(BasePiece):
    """A simple hello world piece that demonstrates basic text output."""

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info(f"Generating greeting for: {input_data.name}")

            # Create a simple greeting message
            message = f"Hello, {input_data.name}!"

            self.logger.info(f"Generated message: {message}")

            # Set display result for Domino UI
            base64_content = base64.b64encode(message.encode("utf-8")).decode("utf-8")
            self.display_result = {
                "file_type": "txt",
                "base64_content": base64_content
            }

            return OutputModel(message=message)

        except Exception as e:
            self.logger.error(f"Error in HelloWorldPiece: {e}")
            print("[HelloWorldPiece] Exception in piece_function:")
            traceback.print_exc()
            raise