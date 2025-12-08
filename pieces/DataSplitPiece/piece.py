from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SplitStrategy, SubjectInfo
import random
import json
import base64
import traceback


class DataSplitPiece(BasePiece):
    """
    A piece that splits dataset subjects into train/validation/test sets.
    
    Supports random or sequential splitting with configurable ratios.
    Useful for preparing medical imaging datasets for machine learning pipelines.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            subjects = input_data.subjects
            train_ratio = input_data.train_ratio
            val_ratio = input_data.val_ratio
            test_ratio = input_data.test_ratio
            seed = input_data.random_seed
            strategy = input_data.split_strategy
            
            # Validate ratios
            total_ratio = train_ratio + val_ratio + test_ratio
            if abs(total_ratio - 1.0) > 0.001:
                self.logger.warning(f"Ratios sum to {total_ratio}, normalizing...")
                train_ratio = train_ratio / total_ratio
                val_ratio = val_ratio / total_ratio
                test_ratio = test_ratio / total_ratio
            
            n = len(subjects)
            self.logger.info(f"Splitting {n} subjects with ratios: train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f}")
            
            # Create a copy to avoid modifying input
            subjects_list = list(subjects)
            
            # Apply splitting strategy
            if strategy == SplitStrategy.RANDOM:
                if seed is not None:
                    random.seed(seed)
                random.shuffle(subjects_list)
                self.logger.info(f"Applied random shuffle with seed={seed}")
            else:
                self.logger.info("Using sequential split (no shuffle)")
            
            # Calculate split indices
            train_end = int(train_ratio * n)
            val_end = train_end + int(val_ratio * n)
            
            # Split subjects
            train_subjects = subjects_list[:train_end]
            val_subjects = subjects_list[train_end:val_end]
            test_subjects = subjects_list[val_end:]
            
            # Create split summary
            split_info = {
                "total_subjects": n,
                "train_count": len(train_subjects),
                "val_count": len(val_subjects),
                "test_count": len(test_subjects),
                "train_ratio_actual": len(train_subjects) / n if n > 0 else 0,
                "val_ratio_actual": len(val_subjects) / n if n > 0 else 0,
                "test_ratio_actual": len(test_subjects) / n if n > 0 else 0,
                "random_seed": seed,
                "strategy": strategy.value,
                "train_ids": [s.subject_id for s in train_subjects],
                "val_ids": [s.subject_id for s in val_subjects],
                "test_ids": [s.subject_id for s in test_subjects]
            }
            
            self.logger.info(f"Split complete: train={len(train_subjects)}, val={len(val_subjects)}, test={len(test_subjects)}")
            
            # Set display result for Domino UI
            display_summary = {
                "total": n,
                "train": len(train_subjects),
                "val": len(val_subjects),
                "test": len(test_subjects),
                "seed": seed
            }
            summary_text = json.dumps(display_summary, indent=2)
            base64_content = base64.b64encode(summary_text.encode("utf-8")).decode("utf-8")
            self.display_result = {
                "file_type": "json",
                "base64_content": base64_content
            }
            
            return OutputModel(
                train_subjects=train_subjects,
                val_subjects=val_subjects,
                test_subjects=test_subjects,
                train_count=len(train_subjects),
                val_count=len(val_subjects),
                test_count=len(test_subjects),
                total_count=n,
                split_info=split_info
            )
            
        except Exception as e:
            self.logger.error(f"Error in DataSplitPiece: {e}")
            print("[DataSplitPiece] Exception in piece_function:")
            traceback.print_exc()
            raise
