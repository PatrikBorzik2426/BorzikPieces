from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SampleInfo
import os


class HistoDataLoaderPiece(BasePiece):
    """
    Discovers and validates paired histopathology image/mask files.
    Outputs a SampleInfo list for downstream pieces (PatchExtractor, DataSplit, EDA).
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        images_path = input_data.images_path
        masks_path = input_data.masks_path
        ext = input_data.file_extension.lower()

        self.logger.info(f"Images: {images_path}")
        self.logger.info(f"Masks:  {masks_path}")

        if not os.path.isdir(images_path):
            raise ValueError(f"Images directory not found: {images_path}")
        if not os.path.isdir(masks_path):
            raise ValueError(f"Masks directory not found: {masks_path}")

        image_files = {f for f in os.listdir(images_path) if f.lower().endswith(ext)}
        mask_files = {f for f in os.listdir(masks_path) if f.lower().endswith(ext)}

        matched = sorted(image_files & mask_files)
        images_only = sorted(image_files - mask_files)
        masks_only = sorted(mask_files - image_files)

        if not matched:
            raise ValueError(
                f"No matched image-mask pairs found with extension '{ext}'. "
                f"Images without mask: {len(images_only)}, Masks without image: {len(masks_only)}"
            )

        if images_only:
            self.logger.warning(f"{len(images_only)} images have no matching mask (first 5: {images_only[:5]})")
        if masks_only:
            self.logger.warning(f"{len(masks_only)} masks have no matching image (first 5: {masks_only[:5]})")

        samples = [
            SampleInfo(
                name=fname,
                image_path=os.path.join(images_path, fname),
                mask_path=os.path.join(masks_path, fname),
            )
            for fname in matched
        ]

        summary = "\n".join([
            "Histopathology Dataset",
            "=" * 36,
            f"Images dir   : {images_path}",
            f"Masks dir    : {masks_path}",
            f"Extension    : {ext}",
            f"Matched pairs: {len(matched)}",
            f"Unmatched img: {len(images_only)}",
            f"Unmatched msk: {len(masks_only)}",
            f"First 5      : {matched[:5]}",
        ])
        self.logger.info(summary)

        return OutputModel(
            images_path=images_path,
            masks_path=masks_path,
            samples=samples,
            num_samples=len(samples),
            data_summary=summary,
        )
