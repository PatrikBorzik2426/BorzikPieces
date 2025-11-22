from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
from PIL import Image, ImageDraw
import random
import os


class GenerativeShapesPiece(BasePiece):
    """A simple piece that generates an image with geometric shapes."""

    NAME = "GenerativeShapesPiece"

    def piece_function(self, input_data: InputModel) -> OutputModel:
        import traceback
        try:
            self.logger.info("Starting shape generation.")
            width = input_data.width
            height = input_data.height
            bg = input_data.background_color
            shape = input_data.shape
            count = input_data.count
            size = input_data.shape_size
            col = input_data.color
            seed = input_data.seed if input_data.seed is not None else 0

            self.logger.info(f"Generating {count} {shape}(s) of size {size} on {width}x{height} canvas with seed {seed}")
            random.seed(seed)
            im = Image.new("RGB", (width, height), bg)
            draw = ImageDraw.Draw(im)

            self.logger.info(f"Created canvas with background color {bg}")
            for i in range(count):
                x = random.randint(0, max(0, width - size))
                y = random.randint(0, max(0, height - size))
                if shape == "circle":
                    draw.ellipse([x, y, x + size, y + size], fill=col)
                elif shape == "square":
                    draw.rectangle([x, y, x + size, y + size], fill=col)
                else:  # triangle
                    draw.polygon([(x, y + size), (x + size / 2, y), (x + size, y + size)], fill=col)
                
                self.logger.debug(f"Drew {shape} {i+1}/{count} at ({x}, {y})")

            self.logger.info("Shape generation completed, saving image")
            
            # results_path may not be set in some runtime/testing environments;
            # fall back to a safe temporary directory to avoid raising an exception
            # which would cause the HTTP server to return 500.
            results_dir = getattr(self, "results_path", None)
            if results_dir:
                self.logger.info(f"Using results_path: {results_dir}")
                os.makedirs(results_dir, exist_ok=True)
                out_path = os.path.join(results_dir, "shapes.png")
            else:
                self.logger.warning("results_path not set, using /tmp")
                results_dir = os.environ.get("RESULTS_PATH") or "/tmp"
                out_path = os.path.join(results_dir, "shapes.png")
            
            im.save(out_path, format="PNG")
            self.logger.info(f"Image saved to {out_path}")

            # Tell Domino how to display the artifact
            self.display_result = {"file_type": "image/png", "file_path": "shapes.png"}
            self.logger.info("Set display_result for PNG image")

            out_path = os.path.join(results_dir, "shapes.png")
            im.save(out_path, format="PNG")

            # Tell Domino how to display the artifact
            self.display_result = {"file_type": "image/png", "file_path": "shapes.png"}
            
            # Image base 64 encoding for OutputModel
            with open(out_path, "rb") as img_file:
                import base64
                image_bytes = img_file.read()
                image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    

            return OutputModel(message="Shapes generated", file_path="shapes.png", image_base64=image_base64)
        
        except Exception as e:
            # Print full traceback to stdout so Domino logs capture it
            print("[GenerativeShapesPiece] Exception in piece_function:")
            traceback.print_exc()
            # Optionally, re-raise to propagate error to Domino
            raise
