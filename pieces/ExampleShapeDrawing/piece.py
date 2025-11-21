from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from PIL import Image, ImageDraw
from pathlib import Path


class ExampleShapesPiece(BasePiece):

    def piece_function(self, input_data: InputModel):

        self.logger.info("Generating shapes image.")

        width = int(input_data.width)
        height = int(input_data.height)
        bg = input_data.background_color

        img = Image.new("RGB", (width, height), bg)
        draw = ImageDraw.Draw(img)

        cols = max(1, int(input_data.cols))
        count = max(0, int(input_data.count))
        size = int(input_data.shape_size)
        color = input_data.color

        padding = 10
        x_spacing = (width - 2 * padding) / cols
        rows = max(1, (count + cols - 1) // cols)
        y_spacing = (height - 2 * padding) / rows

        shapes_cycle = ["circle", "square", "triangle"]

        for i in range(count):
            col = i % cols
            row = i // cols
            cx = padding + col * x_spacing + x_spacing / 2
            cy = padding + row * y_spacing + y_spacing / 2

            left = cx - size / 2
            top = cy - size / 2
            right = cx + size / 2
            bottom = cy + size / 2

            if input_data.shape == "random":
                shape = shapes_cycle[i % len(shapes_cycle)]
            else:
                shape = input_data.shape

            if shape == "circle":
                draw.ellipse([left, top, right, bottom], fill=color)
            elif shape == "square":
                draw.rectangle([left, top, right, bottom], fill=color)
            elif shape == "triangle":
                draw.polygon([(cx, top), (right, bottom), (left, bottom)], fill=color)
            else:
                # unknown shape: draw a small cross
                draw.line([(left, top), (right, bottom)], fill=color)
                draw.line([(left, bottom), (right, top)], fill=color)

        out_path = Path(self.results_path) / "shapes.png"
        img.save(out_path)

        # Provide a UI hint for the platform. Use a path relative to results root
        # and a specific MIME-like type to maximize compatibility with different UIs.
        self.display_result = {
            "file_type": "image/png",
            "file_path": "shapes.png"
        }

        return OutputModel(message="Shapes image generated.", image_path=str(out_path))
