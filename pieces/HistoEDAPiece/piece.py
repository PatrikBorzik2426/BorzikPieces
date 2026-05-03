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


def _parse_class_mapping(json_str: str):
    """Return {int_id: (R,G,B)} and {(R,G,B): int_id}."""
    raw = json.loads(json_str)
    class_map = {int(k): tuple(v) for k, v in raw.items()}
    rgb_to_class = {v: k for k, v in class_map.items()}
    return class_map, rgb_to_class


def _rgb_to_class_indices(mask_rgb: np.ndarray, rgb_to_class: dict) -> np.ndarray:
    mask_class = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    for rgb, cls_id in rgb_to_class.items():
        hit = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=2)
        mask_class[hit] = cls_id
    return mask_class


class HistoEDAPiece(BasePiece):
    """
    Exploratory Data Analysis for histopathology segmentation datasets.

    Computes:
    - Dataset size and image dimension distribution
    - Per-class pixel count and frequency across masks
    - Visual gallery of sample image/mask pairs with a colour legend
    Saves an HTML report to results_path.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        images_path = input_data.images_path
        masks_path = input_data.masks_path
        ext = input_data.file_extension.lower()

        class_map, rgb_to_class = _parse_class_mapping(input_data.class_mapping_json)
        class_names = input_data.class_names
        num_classes = len(class_map)

        self.logger.info("Starting HistoEDA")
        self.logger.info(f"Images: {images_path}  Masks: {masks_path}")

        # ── collect file list ──────────────────────────────────────────────────
        image_files = sorted(
            f for f in os.listdir(images_path) if f.lower().endswith(ext)
        )
        self.logger.info(f"Found {len(image_files)} image files")

        # ── phase 1: image dimension stats ────────────────────────────────────
        widths, heights = [], []
        for fname in image_files:
            try:
                with Image.open(os.path.join(images_path, fname)) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except Exception:
                pass

        dim_summary = (
            f"Width  : min={min(widths)} max={max(widths)} mean={np.mean(widths):.0f}\n"
            f"Height : min={min(heights)} max={max(heights)} mean={np.mean(heights):.0f}"
        ) if widths else "No images readable."

        # ── phase 2: per-class pixel distribution ─────────────────────────────
        sample_files = image_files[: input_data.max_eda_images]
        pixel_counts = np.zeros(num_classes, dtype=np.int64)
        processed = 0
        for fname in sample_files:
            mask_path = os.path.join(masks_path, fname)
            if not os.path.exists(mask_path):
                continue
            try:
                mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
                cls_mask = _rgb_to_class_indices(mask_rgb, rgb_to_class)
                for c in range(num_classes):
                    pixel_counts[c] += int(np.sum(cls_mask == c))
                processed += 1
            except Exception as e:
                self.logger.warning(f"Could not process mask {fname}: {e}")

        total_pixels = pixel_counts.sum()
        class_dist_lines = []
        for c in range(num_classes):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            pct = 100.0 * pixel_counts[c] / total_pixels if total_pixels > 0 else 0.0
            class_dist_lines.append(f"  {c} {name:20s}: {pixel_counts[c]:>12,}  ({pct:5.1f}%)")

        # ── phase 3: generate figures ──────────────────────────────────────────
        html_parts = [
            "<html><head><meta charset='utf-8'>",
            "<style>body{font-family:sans-serif;max-width:1400px;margin:auto;padding:20px}",
            "h2{color:#333}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style>",
            "</head><body>",
            "<h1>Histopathology Dataset EDA</h1>",
        ]

        # — class distribution bar chart —
        fig, ax = plt.subplots(figsize=(9, 4))
        colors = [
            tuple(v / 255.0 for v in class_map[c])
            if max(class_map[c]) > 0 else (0.2, 0.2, 0.2)
            for c in range(num_classes)
        ]
        bars = ax.bar(
            range(num_classes),
            pixel_counts,
            color=colors,
            edgecolor='black',
            linewidth=0.5
        )
        ax.set_xticks(range(num_classes))
        ax.set_xticklabels(
            [class_names[c] if c < len(class_names) else f"c{c}" for c in range(num_classes)],
            rotation=30, ha='right'
        )
        ax.set_ylabel("Pixel count")
        ax.set_title(f"Class pixel distribution (n={processed} masks)")
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120)
        plt.close()
        bar_b64 = base64.b64encode(buf.getvalue()).decode()

        html_parts.append("<h2>Class Pixel Distribution</h2>")
        html_parts.append(f'<img src="data:image/png;base64,{bar_b64}" style="max-width:100%"><br>')

        # — image dimension scatter —
        if widths:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.scatter(widths, heights, alpha=0.3, s=10, color='steelblue')
            ax.set_xlabel("Width (px)")
            ax.set_ylabel("Height (px)")
            ax.set_title("Image dimension scatter")
            plt.tight_layout()
            buf2 = io.BytesIO()
            plt.savefig(buf2, format='png', dpi=100)
            plt.close()
            dim_b64 = base64.b64encode(buf2.getvalue()).decode()
            html_parts.append("<h2>Image Dimensions</h2>")
            html_parts.append(f'<img src="data:image/png;base64,{dim_b64}" style="max-width:100%"><br>')

        # — sample gallery —
        gallery_n = min(input_data.gallery_samples, len(image_files))
        if gallery_n > 0:
            gallery_files = image_files[:gallery_n]
            fig, axes = plt.subplots(gallery_n, 2, figsize=(10, 3 * gallery_n))
            if gallery_n == 1:
                axes = [axes]
            for row, fname in enumerate(gallery_files):
                try:
                    img_arr = np.array(Image.open(os.path.join(images_path, fname)).convert("RGB"))
                    axes[row][0].imshow(img_arr)
                    axes[row][0].set_title(fname, fontsize=7)
                    axes[row][0].axis('off')
                except Exception:
                    axes[row][0].axis('off')

                mask_fpath = os.path.join(masks_path, fname)
                if os.path.exists(mask_fpath):
                    try:
                        mask_rgb = np.array(Image.open(mask_fpath).convert("RGB"))
                        axes[row][1].imshow(mask_rgb)
                        axes[row][1].set_title("mask", fontsize=7)
                        axes[row][1].axis('off')
                    except Exception:
                        axes[row][1].axis('off')
                else:
                    axes[row][1].axis('off')

            # colour legend
            legend_patches = []
            for c in range(num_classes):
                rgb_norm = tuple(v / 255.0 for v in class_map[c]) if max(class_map[c]) > 0 else (0.2, 0.2, 0.2)
                lbl = class_names[c] if c < len(class_names) else f"class_{c}"
                legend_patches.append(mpatches.Patch(color=rgb_norm, label=lbl))
            fig.legend(handles=legend_patches, loc='lower center', ncol=num_classes, fontsize=8)
            plt.suptitle("Sample Image / Mask Gallery", fontsize=12)
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            buf3 = io.BytesIO()
            plt.savefig(buf3, format='png', dpi=120)
            plt.close()
            gallery_b64 = base64.b64encode(buf3.getvalue()).decode()
            html_parts.append("<h2>Sample Gallery (image | mask)</h2>")
            html_parts.append(f'<img src="data:image/png;base64,{gallery_b64}" style="max-width:100%"><br>')

        # — stats table —
        html_parts.append("<h2>Class Statistics</h2>")
        html_parts.append("<table><tr><th>ID</th><th>Name</th><th>Pixels</th><th>%</th><th>Colour</th></tr>")
        for c in range(num_classes):
            name = class_names[c] if c < len(class_names) else f"class_{c}"
            pct = 100.0 * pixel_counts[c] / total_pixels if total_pixels > 0 else 0.0
            r, g, b = class_map[c]
            html_parts.append(
                f"<tr><td>{c}</td><td>{name}</td>"
                f"<td>{pixel_counts[c]:,}</td><td>{pct:.1f}%</td>"
                f"<td style='background-color:rgb({r},{g},{b});width:40px'>&nbsp;</td></tr>"
            )
        html_parts.append("</table>")

        html_parts.append("<h2>Image Dimension Stats</h2>")
        html_parts.append(f"<pre>{dim_summary}</pre>")
        html_parts.append(f"<p>Masks processed for pixel stats: {processed} / {len(image_files)}</p>")
        html_parts.append("</body></html>")

        # ── save HTML report ───────────────────────────────────────────────────
        report_path = os.path.join(self.results_path, "histo_eda_report.html")
        with open(report_path, 'w') as f:
            f.write("\n".join(html_parts))

        self.display_result = {
            "file_type": "html",
            "base64_content": base64.b64encode("\n".join(html_parts).encode()).decode()
        }

        summary = "\n".join([
            "Histopathology EDA Summary",
            "=" * 40,
            f"Total images found : {len(image_files)}",
            f"Masks analyzed     : {processed}",
            "",
            "Image dimension stats:",
            dim_summary,
            "",
            "Class pixel distribution:",
            *class_dist_lines,
        ])
        self.logger.info(summary)

        return OutputModel(
            eda_summary=summary,
            report_path=report_path,
        )
