from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import traceback


class ExampleSineWavePiece(BasePiece):

    def piece_function(self, input_data: InputModel):
        try:
            self.logger.info("Generating sine wave.")

            # Generate sine wave
            num_samples = int(input_data.duration * input_data.sample_rate)
            time = np.linspace(0, input_data.duration, num_samples)
            sine_wave = np.sin(2 * np.pi * input_data.frequency * time)

            self.logger.info(f"Generated sine wave with {num_samples} samples")

            # Generate wave figure
            time = np.linspace(0, input_data.duration, len(sine_wave))
            fig = go.Figure(data=go.Scatter(x=time, y=sine_wave, mode='lines'))
            fig.update_layout(title='Sine Wave', xaxis_title='Time (s)', yaxis_title='Amplitude')

            # Save figure to piece file system
            results_path = getattr(self, "results_path", None)
            if results_path:
                self.logger.info(f"Using results_path: {results_path}")
                fig_path = str(Path(results_path) / "sine.json")
            else:
                self.logger.warning("results_path not set, using /tmp")
                fig_path = "/tmp/sine.json"
            
            fig.write_json(fig_path)
            self.logger.info(f"Saved sine wave plot to {fig_path}")

            # Set wave image to be displayed in the results on UI
            self.display_result = {
                'file_type': 'plotly_json',
                'file_path': 'sine.json'
            }
            self.logger.info("Set display_result for plotly JSON")

            # Return success message
            return OutputModel(message="Sine wave generated.")
        except Exception as e:
            self.logger.error(f"Error in ExampleSineWavePiece: {e}")
            print("[ExampleSineWavePiece] Exception in piece_function:")
            traceback.print_exc()
            raise
