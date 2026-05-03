from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import os
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp


# ── helpers ────────────────────────────────────────────────────────────────────

def _parse_class_mapping(json_str):
    raw = json.loads(json_str)
    class_map = {int(k): tuple(v) for k, v in raw.items()}
    rgb_to_class = {v: k for k, v in class_map.items()}
    return class_map, rgb_to_class


def _rgb_to_class(mask_rgb, rgb_to_class):
    out = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    for rgb, cid in rgb_to_class.items():
        out[np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=2)] = cid
    return out


def _class_to_rgb(mask_cls, class_map):
    out = np.zeros((*mask_cls.shape, 3), dtype=np.uint8)
    for cid, rgb in class_map.items():
        out[mask_cls == cid] = rgb
    return out


def _build_model(arch, encoder_name, num_classes):
    kw = dict(encoder_name=encoder_name, encoder_weights=None,
              in_channels=3, classes=num_classes)
    return {
        'unet': smp.Unet, 'unetplusplus': smp.UnetPlusPlus,
        'fpn': smp.FPN, 'deeplabv3plus': smp.DeepLabV3Plus,
    }.get(arch.lower(), smp.Unet)(**kw)


class DiceAccumulator:
    def __init__(self, n):
        self.n = n
        self.inter = np.zeros(n)
        self.ps = np.zeros(n)
        self.ts = np.zeros(n)

    def update(self, pred, tgt):
        p, t = pred.cpu().numpy(), tgt.cpu().numpy()
        for c in range(self.n):
            self.inter[c] += np.sum((p == c) & (t == c))
            self.ps[c] += np.sum(p == c)
            self.ts[c] += np.sum(t == c)

    def compute(self):
        dice = []
        for c in range(self.n):
            u = self.ps[c] + self.ts[c]
            dice.append(1.0 if u == 0 else 2.0 * self.inter[c] / u)
        return dice, float(np.mean(dice))


# ── piece ──────────────────────────────────────────────────────────────────────

class HistoInferencePiece(BasePiece):
    """
    Runs inference on a histopathology test set using a trained HistoTrainingPiece model.

    For each sample:
      - Resizes and normalises the image (same pipeline as training)
      - Saves the predicted RGB mask
      - Saves a side-by-side comparison figure (input | pred | GT if available)
      - Computes Dice scores if mask_path is present in SampleInfo
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info("=" * 60)
            self.logger.info("HistoInferencePiece")
            self.logger.info("=" * 60)

            if input_data.use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                self.logger.info("CPU mode")

            class_map, rgb_to_class = _parse_class_mapping(input_data.class_mapping_json)
            class_names = input_data.class_names

            # ── load model ─────────────────────────────────────────────────────
            if not os.path.exists(input_data.model_path):
                raise FileNotFoundError(f"Model not found: {input_data.model_path}")

            ckpt = torch.load(input_data.model_path, map_location=device)
            arch = ckpt.get('arch', input_data.model_architecture)
            enc = ckpt.get('encoder_name', input_data.encoder_name)
            nc = ckpt.get('num_classes', input_data.num_classes)
            ch = ckpt.get('image_height', input_data.image_height)
            cw = ckpt.get('image_width', input_data.image_width)

            model = _build_model(arch, enc, nc).to(device)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            model.eval()
            self.logger.info(f"Loaded {arch}/{enc}  classes={nc}  size={ch}x{cw}")
            self.logger.info(f"Running inference on {len(input_data.test_samples)} samples")

            # ── output dirs ────────────────────────────────────────────────────
            preds_dir = os.path.join(input_data.output_dir, "predictions")
            viz_dir = os.path.join(input_data.output_dir, "visualizations")
            os.makedirs(preds_dir, exist_ok=True)
            os.makedirs(viz_dir, exist_ok=True)

            transform = A.Compose([
                A.Resize(ch, cw),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

            dice_acc = DiceAccumulator(nc)
            has_any_mask = any(s.mask_path for s in input_data.test_samples)
            processed = 0
            viz_count = 0

            for sample in input_data.test_samples:
                try:
                    orig = np.array(Image.open(sample.image_path).convert("RGB"))
                except Exception as e:
                    self.logger.warning(f"Cannot open {sample.name}: {e}")
                    continue

                aug = transform(image=orig)
                img_t = aug["image"].unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(img_t)
                pred = torch.argmax(logits, dim=1).squeeze(0)

                # save prediction RGB
                pred_rgb = _class_to_rgb(pred.cpu().numpy(), class_map)
                Image.fromarray(pred_rgb).save(os.path.join(preds_dir, sample.name))
                processed += 1

                # optional ground truth Dice
                gt_rgb_vis = None
                if sample.mask_path and os.path.exists(sample.mask_path):
                    gt_rgb = np.array(Image.open(sample.mask_path).convert("RGB"))
                    gt_cls = _rgb_to_class(gt_rgb, rgb_to_class)
                    gt_r = np.array(
                        Image.fromarray(gt_cls.astype(np.uint8)).resize((cw, ch), Image.NEAREST)
                    ).astype(np.int64)
                    gt_t = torch.from_numpy(gt_r).unsqueeze(0)
                    dice_acc.update(pred.unsqueeze(0), gt_t)
                    gt_rgb_vis = _class_to_rgb(gt_r, class_map)

                if input_data.save_comparison_images and viz_count < input_data.max_comparison_images:
                    self._save_comparison(
                        orig, pred_rgb, gt_rgb_vis,
                        sample.name, class_map, class_names,
                        os.path.join(viz_dir, sample.name)
                    )
                    viz_count += 1

            # ── Dice ───────────────────────────────────────────────────────────
            mean_dice, dice_pc = None, None
            if has_any_mask:
                dice_pc, mean_dice = dice_acc.compute()
                self.logger.info(f"Mean Dice: {mean_dice:.4f}")
                for c, d in enumerate(dice_pc):
                    name = class_names[c] if c < len(class_names) else f"c{c}"
                    self.logger.info(f"  {c} {name}: {d:.4f}")

            summary_lines = [
                "HistoInferencePiece Summary",
                "=" * 40,
                f"Model            : {arch}/{enc}",
                f"Images processed : {processed}",
                f"Predictions      : {preds_dir}",
                f"Visualizations   : {viz_dir}",
            ]
            if mean_dice is not None:
                summary_lines.append(f"Mean Dice        : {mean_dice:.4f}")
                for c, d in enumerate(dice_pc):
                    name = class_names[c] if c < len(class_names) else f"c{c}"
                    summary_lines.append(f"  {c} {name}: {d:.4f}")
            summary = "\n".join(summary_lines)
            self.logger.info(summary)

            return OutputModel(
                predictions_dir=preds_dir,
                visualization_dir=viz_dir,
                num_images_processed=processed,
                mean_dice=mean_dice,
                dice_per_class=dice_pc,
                inference_summary=summary,
            )

        except Exception as e:
            import traceback
            self.logger.error(f"HistoInferencePiece error: {e}\n{traceback.format_exc()}")
            raise

    def _save_comparison(self, orig, pred_rgb, gt_rgb, name, class_map, class_names, save_path):
        cols = 3 if gt_rgb is not None else 2
        fig, axes = plt.subplots(1, cols, figsize=(5 * cols, 4))
        axes[0].imshow(orig);     axes[0].set_title("Input", fontsize=9);      axes[0].axis('off')
        axes[1].imshow(pred_rgb); axes[1].set_title("Prediction", fontsize=9); axes[1].axis('off')
        if gt_rgb is not None:
            axes[2].imshow(gt_rgb); axes[2].set_title("Ground Truth", fontsize=9); axes[2].axis('off')
        n = len(class_map)
        patches = []
        for c in range(n):
            rgb_n = tuple(v / 255.0 for v in class_map[c]) if max(class_map[c]) > 0 else (0.2,) * 3
            lbl = class_names[c] if c < len(class_names) else f"c{c}"
            patches.append(mpatches.Patch(color=rgb_n, label=lbl))
        fig.legend(handles=patches, loc='lower center', ncol=n, fontsize=7)
        plt.suptitle(name, fontsize=8)
        plt.tight_layout(rect=[0, 0.06, 1, 1])
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close()
