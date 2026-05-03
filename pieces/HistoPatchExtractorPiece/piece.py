from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, SampleInfo
import os
import numpy as np
from PIL import Image


class HistoPatchExtractorPiece(BasePiece):
    """
    Tiles large histopathology images into fixed-size patches using a sliding window.

    Patches whose masks contain less than `min_foreground_ratio` non-background
    pixels are discarded, preventing the training set from being dominated by
    empty background patches from whole-slide images.

    Output patches are saved as PNG files and a new SampleInfo list is returned
    for HistoDataSplitPiece.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        patch_size = input_data.patch_size
        stride = input_data.stride if input_data.stride > 0 else patch_size
        bg_rgb = tuple(input_data.background_rgb)
        min_fg = input_data.min_foreground_ratio

        out_images = os.path.join(input_data.output_dir, "images")
        out_masks = os.path.join(input_data.output_dir, "masks")
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_masks, exist_ok=True)

        self.logger.info(
            f"Extracting {patch_size}x{patch_size} patches  "
            f"stride={stride}  min_fg={min_fg:.2f}  "
            f"from {len(input_data.samples)} images"
        )

        patch_samples: list[SampleInfo] = []
        total_skipped = 0
        total_considered = 0

        for sample in input_data.samples:
            try:
                img = np.array(Image.open(sample.image_path).convert("RGB"))
                msk = np.array(Image.open(sample.mask_path).convert("RGB")) \
                    if sample.mask_path and os.path.exists(sample.mask_path) \
                    else None
            except Exception as e:
                self.logger.warning(f"Cannot open {sample.name}: {e}")
                continue

            H, W = img.shape[:2]
            stem = os.path.splitext(sample.name)[0]

            row_starts = list(range(0, max(H - patch_size, 0) + 1, stride))
            if not row_starts or row_starts[-1] + patch_size > H:
                row_starts = [max(H - patch_size, 0)]

            col_starts = list(range(0, max(W - patch_size, 0) + 1, stride))
            if not col_starts or col_starts[-1] + patch_size > W:
                col_starts = [max(W - patch_size, 0)]

            for r in row_starts:
                for c in col_starts:
                    total_considered += 1
                    img_patch = img[r:r + patch_size, c:c + patch_size]

                    if img_patch.shape[0] < patch_size or img_patch.shape[1] < patch_size:
                        # Pad if the image is smaller than patch_size
                        ph = patch_size - img_patch.shape[0]
                        pw = patch_size - img_patch.shape[1]
                        img_patch = np.pad(img_patch, [(0, ph), (0, pw), (0, 0)], constant_values=0)

                    msk_patch = None
                    if msk is not None:
                        mp = msk[r:r + patch_size, c:c + patch_size]
                        if mp.shape[0] < patch_size or mp.shape[1] < patch_size:
                            ph = patch_size - mp.shape[0]
                            pw = patch_size - mp.shape[1]
                            mp = np.pad(mp, [(0, ph), (0, pw), (0, 0)],
                                        constant_values=bg_rgb[0])
                        msk_patch = mp

                        # Foreground filter: check fraction of non-background pixels
                        if min_fg > 0:
                            is_bg = np.all(msk_patch == np.array(bg_rgb, dtype=np.uint8), axis=2)
                            fg_ratio = 1.0 - is_bg.mean()
                            if fg_ratio < min_fg:
                                total_skipped += 1
                                continue

                    patch_name = f"{stem}_r{r}_c{c}.png"
                    img_save_path = os.path.join(out_images, patch_name)
                    Image.fromarray(img_patch).save(img_save_path)

                    msk_save_path = None
                    if msk_patch is not None:
                        msk_save_path = os.path.join(out_masks, patch_name)
                        Image.fromarray(msk_patch).save(msk_save_path)

                    patch_samples.append(SampleInfo(
                        name=patch_name,
                        image_path=img_save_path,
                        mask_path=msk_save_path,
                    ))

            if len(patch_samples) % 500 == 0 and len(patch_samples) > 0:
                self.logger.info(f"  {len(patch_samples)} patches saved so far...")

        summary = "\n".join([
            "HistoPatchExtractorPiece Summary",
            "=" * 40,
            f"Input images       : {len(input_data.samples)}",
            f"Patch size         : {patch_size}x{patch_size}",
            f"Stride             : {stride}",
            f"Min foreground     : {min_fg:.2f}",
            f"Candidate patches  : {total_considered}",
            f"Skipped (bg-only)  : {total_skipped}",
            f"Saved patches      : {len(patch_samples)}",
            f"Output dir         : {input_data.output_dir}",
        ])
        self.logger.info(summary)

        return OutputModel(
            samples=patch_samples,
            num_patches=len(patch_samples),
            patches_images_path=out_images,
            patches_masks_path=out_masks,
            extraction_summary=summary,
        )
