"""
Download MoNuSeg (Multi-organ Nucleus Segmentation) dataset from HuggingFace
and convert to the RGB-mask format expected by the histopathology pipeline.

Dataset: RationAI/MoNuSeg  (CC BY-NC-SA 4.0)
  - 51 H&E images at 1000×1000px from 7 organ types
  - Nuclear instance segmentation masks (per-nucleus binary images)

Conversion:
  - Merge all per-instance binary masks → single semantic mask
  - Class 0 = background  → RGB (0, 0, 0)   black
  - Class 1 = cell nucleus → RGB (0, 255, 0) green

Output directories (mounted as /home/shared_storage inside containers):
  domino_data/histo_data/images/<name>.png   ← H&E tile
  domino_data/histo_data/masks/<name>.png    ← RGB semantic mask

Pipeline notes:
  - Images are 1000×1000 → use HistoPatchExtractorPiece (patch_size=256, stride=128)
  - class_mapping_json = {"0": [0,0,0], "1": [0,255,0]}
  - class_names        = ["background", "nucleus"]
  - num_classes        = 2
"""

import sys
import argparse
import numpy as np
from PIL import Image
from pathlib import Path

TISSUE_NAMES = {
    0: "Unknown", 1: "Breast", 2: "Kidney", 3: "Liver",
    4: "Prostate", 5: "Bladder", 6: "Colon", 7: "Stomach",
}

CLASS_COLORS = {
    0: (0,   0,   0),    # background
    1: (0, 255,   0),    # nucleus
}


def convert_instances_to_semantic_rgb(instances: list) -> np.ndarray:
    """Merge per-nucleus binary masks into a single RGB semantic mask."""
    if not instances:
        h, w = 1000, 1000
        return np.zeros((h, w, 3), dtype=np.uint8)

    h, w = np.array(instances[0]).shape
    semantic = np.zeros((h, w), dtype=np.uint8)  # 0 = background

    for inst_img in instances:
        inst_arr = np.array(inst_img)
        semantic[inst_arr > 0] = 1  # nucleus class

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS.items():
        rgb[semantic == cls_id] = color

    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="domino_data/histo_data",
                        help="Base output directory (default: domino_data/histo_data)")
    parser.add_argument("--splits", nargs="+", default=["train", "test"],
                        help="HuggingFace dataset splits to download")
    args = parser.parse_args()

    images_dir = Path(args.out_dir) / "images"
    masks_dir  = Path(args.out_dir) / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output → {args.out_dir}/images  and  {args.out_dir}/masks")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not found. Run: pip install datasets")
        sys.exit(1)

    total = 0
    for split in args.splits:
        print(f"\nDownloading MoNuSeg split='{split}' ...")
        ds = load_dataset("RationAI/MoNuSeg", split=split)

        for i, sample in enumerate(ds):
            tissue_name = TISSUE_NAMES.get(sample["tissue"], "Unknown")
            patient     = sample["patient"]
            name        = f"monuseg_{split}_{i:03d}_{tissue_name}_{patient}"

            img_path  = images_dir / f"{name}.png"
            mask_path = masks_dir  / f"{name}.png"

            sample["image"].save(img_path)

            rgb_mask = convert_instances_to_semantic_rgb(sample["instances"])
            Image.fromarray(rgb_mask).save(mask_path)

            nucleus_pct = (rgb_mask[:,:,1] == 255).mean() * 100
            print(f"  [{split} {i+1:02d}/{len(ds)}] {tissue_name:<10} {patient}  "
                  f"nucleus={nucleus_pct:.1f}%")
            total += 1

    print(f"\nDone. {total} image/mask pairs saved to {args.out_dir}/")
    print("\nPipeline configuration:")
    print('  images_path      = /home/shared_storage/histo_data/images')
    print('  masks_path       = /home/shared_storage/histo_data/masks')
    print('  class_mapping_json = {"0": [0,0,0], "1": [0,255,0]}')
    print('  class_names      = ["background", "nucleus"]')
    print('  num_classes      = 2')
    print('  → Use HistoPatchExtractorPiece: patch_size=256, stride=128, min_foreground_ratio=0.05')


if __name__ == "__main__":
    main()
