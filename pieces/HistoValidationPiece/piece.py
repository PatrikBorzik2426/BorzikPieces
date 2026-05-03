from domino.base_piece import BasePiece
from .models import InputModel, OutputModel
import os
import json
import base64
import io
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


class MetricsAccumulator:
    """Tracks Dice, IoU and pixel accuracy per class."""

    def __init__(self, n):
        self.n = n
        self.inter = np.zeros(n)
        self.ps = np.zeros(n)
        self.ts = np.zeros(n)
        self.correct = 0
        self.total = 0
        self.conf_mat = np.zeros((n, n), dtype=np.int64)

    def update(self, pred, tgt):
        p, t = pred.cpu().numpy(), tgt.cpu().numpy()
        self.correct += int(np.sum(p == t))
        self.total += p.size
        for c in range(self.n):
            self.inter[c] += int(np.sum((p == c) & (t == c)))
            self.ps[c] += int(np.sum(p == c))
            self.ts[c] += int(np.sum(t == c))
        # flattened confusion matrix
        for true_c in range(self.n):
            for pred_c in range(self.n):
                self.conf_mat[true_c, pred_c] += int(np.sum((t == true_c) & (p == pred_c)))

    def compute(self):
        dice, iou = [], []
        for c in range(self.n):
            u = self.ps[c] + self.ts[c]
            inter = self.inter[c]
            dice.append(1.0 if u == 0 else 2.0 * inter / u)
            iou_denom = u - inter
            iou.append(1.0 if iou_denom == 0 else inter / iou_denom)
        acc = self.correct / self.total if self.total > 0 else 0.0
        return dice, float(np.mean(dice)), iou, float(np.mean(iou)), acc


# ── piece ──────────────────────────────────────────────────────────────────────

class HistoValidationPiece(BasePiece):
    """
    Runs thorough validation of a trained histopathology model on a labelled
    validation set. Computes per-class Dice, IoU, pixel accuracy, and confusion
    matrix, then saves an HTML report with metric tables and comparison images.

    Connect:
      val_samples     ← HistoDataSplitPiece.val_samples
      model_path      ← HistoTrainingPiece.best_model_path
      (all arch fields) ← HistoTrainingPiece pass-through outputs
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info("=" * 60)
            self.logger.info("HistoValidationPiece")
            self.logger.info("=" * 60)

            if input_data.use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                self.logger.info("CPU mode")

            class_map, rgb_to_class = _parse_class_mapping(input_data.class_mapping_json)
            num_classes = input_data.num_classes
            class_names = input_data.class_names
            h, w = input_data.image_height, input_data.image_width

            # load model
            if not os.path.exists(input_data.model_path):
                raise FileNotFoundError(f"Model not found: {input_data.model_path}")
            ckpt = torch.load(input_data.model_path, map_location=device)
            arch = ckpt.get('arch', input_data.model_architecture)
            enc = ckpt.get('encoder_name', input_data.encoder_name)
            nc = ckpt.get('num_classes', num_classes)
            ch = ckpt.get('image_height', h)
            cw = ckpt.get('image_width', w)

            model = _build_model(arch, enc, nc).to(device)
            model.load_state_dict(ckpt.get('model_state_dict', ckpt))
            model.eval()
            self.logger.info(f"Loaded {arch}/{enc}  classes={nc}  size={ch}x{cw}")

            os.makedirs(input_data.output_dir, exist_ok=True)

            transform = A.Compose([
                A.Resize(ch, cw),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

            acc = MetricsAccumulator(nc)
            viz_data = []   # (orig_img, pred_rgb, gt_rgb, name)

            for sample in input_data.val_samples:
                try:
                    orig = np.array(Image.open(sample.image_path).convert("RGB"))
                    gt_rgb = np.array(Image.open(sample.mask_path).convert("RGB"))
                except Exception as e:
                    self.logger.warning(f"Cannot open {sample.name}: {e}")
                    continue

                aug = transform(image=orig)
                img_t = aug["image"].unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(img_t)
                pred = torch.argmax(logits, dim=1).squeeze(0)    # [H, W]

                gt_cls = _rgb_to_class(gt_rgb, rgb_to_class)
                gt_resized = np.array(
                    Image.fromarray(gt_cls.astype(np.uint8)).resize((cw, ch), Image.NEAREST)
                ).astype(np.int64)
                gt_t = torch.from_numpy(gt_resized).unsqueeze(0)

                acc.update(pred.unsqueeze(0), gt_t)

                if len(viz_data) < input_data.max_viz_images:
                    pred_rgb = _class_to_rgb(pred.cpu().numpy(), class_map)
                    gt_rgb_r = _class_to_rgb(gt_resized, class_map)
                    viz_data.append((orig, pred_rgb, gt_rgb_r, sample.name))

            dice_pc, mean_dice, iou_pc, mean_iou, px_acc = acc.compute()
            self.logger.info(f"Mean Dice={mean_dice:.4f}  Mean IoU={mean_iou:.4f}  Px Acc={px_acc:.4f}")
            for c in range(nc):
                name = class_names[c] if c < len(class_names) else f"c{c}"
                self.logger.info(f"  {c} {name}: Dice={dice_pc[c]:.4f}  IoU={iou_pc[c]:.4f}")

            # ── HTML report ────────────────────────────────────────────────────
            html = self._build_report(
                arch, enc, len(input_data.val_samples),
                class_names, nc, class_map,
                dice_pc, mean_dice, iou_pc, mean_iou, px_acc,
                acc.conf_mat, viz_data
            )

            report_path = os.path.join(input_data.output_dir, "validation_report.html")
            with open(report_path, 'w') as f:
                f.write(html)

            self.display_result = {
                "file_type": "html",
                "base64_content": base64.b64encode(html.encode()).decode()
            }

            summary_lines = [
                "HistoValidationPiece Summary",
                "=" * 44,
                f"Model        : {arch}/{enc}",
                f"Val samples  : {len(input_data.val_samples)}",
                f"Mean Dice    : {mean_dice:.4f}",
                f"Mean IoU     : {mean_iou:.4f}",
                f"Pixel Acc    : {px_acc:.4f}",
                "",
                "Per-class Dice / IoU:",
            ]
            for c in range(nc):
                name = class_names[c] if c < len(class_names) else f"c{c}"
                summary_lines.append(f"  {c} {name:20s}: Dice={dice_pc[c]:.4f}  IoU={iou_pc[c]:.4f}")
            summary = "\n".join(summary_lines)
            self.logger.info(summary)

            return OutputModel(
                mean_dice=mean_dice,
                dice_per_class=dice_pc,
                mean_iou=mean_iou,
                iou_per_class=iou_pc,
                pixel_accuracy=px_acc,
                report_path=report_path,
                validation_summary=summary,
            )

        except Exception as e:
            import traceback
            self.logger.error(f"HistoValidationPiece error: {e}\n{traceback.format_exc()}")
            raise

    # ── report builder ─────────────────────────────────────────────────────────

    def _build_report(self, arch, enc, n_samples,
                      class_names, num_classes, class_map,
                      dice_pc, mean_dice, iou_pc, mean_iou, px_acc,
                      conf_mat, viz_data):
        parts = [
            "<html><head><meta charset='utf-8'>",
            "<style>body{font-family:sans-serif;max-width:1400px;margin:auto;padding:20px}",
            "h2{color:#333}table{border-collapse:collapse;margin-bottom:20px}",
            "td,th{border:1px solid #ccc;padding:6px 10px;text-align:right}",
            "th{background:#f5f5f5;text-align:center}</style></head><body>",
            "<h1>Histopathology Validation Report</h1>",
            f"<p><b>Model:</b> {arch}/{enc} &nbsp;|&nbsp; <b>Val samples:</b> {n_samples}</p>",
        ]

        # summary card
        parts.append(
            f"<table><tr><th>Metric</th><th>Value</th></tr>"
            f"<tr><td>Mean Dice</td><td>{mean_dice:.4f}</td></tr>"
            f"<tr><td>Mean IoU</td><td>{mean_iou:.4f}</td></tr>"
            f"<tr><td>Pixel Accuracy</td><td>{px_acc:.4f}</td></tr>"
            f"</table>"
        )

        # per-class table
        parts.append("<h2>Per-Class Metrics</h2>")
        parts.append(
            "<table><tr><th>ID</th><th>Name</th><th>Dice</th><th>IoU</th><th>Colour</th></tr>"
        )
        for c in range(num_classes):
            name = class_names[c] if c < len(class_names) else f"c{c}"
            r, g, b = class_map[c]
            parts.append(
                f"<tr><td>{c}</td><td style='text-align:left'>{name}</td>"
                f"<td>{dice_pc[c]:.4f}</td><td>{iou_pc[c]:.4f}</td>"
                f"<td style='background:rgb({r},{g},{b});width:40px'>&nbsp;</td></tr>"
            )
        parts.append("</table>")

        # confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(max(5, num_classes), max(4, num_classes - 1)))
        im = ax.imshow(conf_mat, cmap='Blues')
        ax.set_xticks(range(num_classes))
        ax.set_yticks(range(num_classes))
        labels = [class_names[c] if c < len(class_names) else f"c{c}" for c in range(num_classes)]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix (pixel counts)")
        fig.colorbar(im, ax=ax)
        for i in range(num_classes):
            for j in range(num_classes):
                ax.text(j, i, f"{conf_mat[i, j]:,}", ha='center', va='center', fontsize=6)
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plt.close()
        cm_b64 = base64.b64encode(buf.getvalue()).decode()
        parts.append("<h2>Confusion Matrix</h2>")
        parts.append(f'<img src="data:image/png;base64,{cm_b64}" style="max-width:100%"><br>')

        # sample gallery
        if viz_data:
            parts.append("<h2>Validation Samples (Input | Prediction | Ground Truth)</h2>")
            legend_patches = []
            for c in range(num_classes):
                rgb_n = tuple(v / 255.0 for v in class_map[c]) if max(class_map[c]) > 0 else (0.2,) * 3
                lbl = class_names[c] if c < len(class_names) else f"c{c}"
                legend_patches.append(mpatches.Patch(color=rgb_n, label=lbl))

            for orig, pred_rgb, gt_rgb, name in viz_data:
                fig, axes = plt.subplots(1, 3, figsize=(13, 4))
                axes[0].imshow(orig);      axes[0].set_title("Input", fontsize=8);       axes[0].axis('off')
                axes[1].imshow(pred_rgb);  axes[1].set_title("Prediction", fontsize=8);  axes[1].axis('off')
                axes[2].imshow(gt_rgb);    axes[2].set_title("Ground Truth", fontsize=8); axes[2].axis('off')
                fig.legend(handles=legend_patches, loc='lower center', ncol=num_classes, fontsize=7)
                plt.suptitle(name, fontsize=7)
                plt.tight_layout(rect=[0, 0.06, 1, 1])
                buf2 = io.BytesIO()
                plt.savefig(buf2, format='png', dpi=100)
                plt.close()
                img_b64 = base64.b64encode(buf2.getvalue()).decode()
                parts.append(f'<img src="data:image/png;base64,{img_b64}" style="max-width:100%;margin-bottom:8px"><br>')

        parts.append("</body></html>")
        return "\n".join(parts)
