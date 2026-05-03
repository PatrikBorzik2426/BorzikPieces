from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SplitStrategy
import random
import json
import base64


class HistoDataSplitPiece(BasePiece):
    """
    Splits a SampleInfo list from HistoDataLoaderPiece or HistoPatchExtractorPiece
    into train / val / test subsets.

    Outputs three SampleInfo lists that wire directly into HistoTrainingPiece,
    HistoValidationPiece, and HistoInferencePiece.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        samples = list(input_data.samples)
        n = len(samples)

        if n == 0:
            raise ValueError("No samples provided to HistoDataSplitPiece.")

        # Normalise ratios
        total = input_data.train_ratio + input_data.val_ratio + input_data.test_ratio
        if abs(total - 1.0) > 0.001:
            self.logger.warning(f"Ratios sum to {total:.3f}, normalising...")
            train_r = input_data.train_ratio / total
            val_r = input_data.val_ratio / total
        else:
            train_r = input_data.train_ratio
            val_r = input_data.val_ratio

        # Shuffle or keep order
        if input_data.split_strategy == SplitStrategy.RANDOM:
            random.seed(input_data.random_seed)
            random.shuffle(samples)
            self.logger.info(f"Random shuffle (seed={input_data.random_seed})")
        else:
            self.logger.info("Sequential split (no shuffle)")

        train_end = int(train_r * n)
        val_end = train_end + int(val_r * n)

        train_samples = samples[:train_end]
        val_samples = samples[train_end:val_end]
        test_samples = samples[val_end:]

        # Ensure no split is empty
        if not train_samples:
            train_samples = samples[:1]
        if not val_samples:
            val_samples = samples[:1]
        if not test_samples:
            test_samples = samples[:1]

        split_info = {
            "total": n,
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "train_ratio_actual": round(len(train_samples) / n, 3),
            "val_ratio_actual": round(len(val_samples) / n, 3),
            "test_ratio_actual": round(len(test_samples) / n, 3),
            "seed": input_data.random_seed,
            "strategy": input_data.split_strategy.value,
        }

        self.logger.info(
            f"Split: train={len(train_samples)} ({split_info['train_ratio_actual']:.1%})  "
            f"val={len(val_samples)} ({split_info['val_ratio_actual']:.1%})  "
            f"test={len(test_samples)} ({split_info['test_ratio_actual']:.1%})"
        )

        self.display_result = {
            "file_type": "json",
            "base64_content": base64.b64encode(
                json.dumps(split_info, indent=2).encode()
            ).decode()
        }

        return OutputModel(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            train_count=len(train_samples),
            val_count=len(val_samples),
            test_count=len(test_samples),
            total_count=n,
            split_info=split_info,
        )
