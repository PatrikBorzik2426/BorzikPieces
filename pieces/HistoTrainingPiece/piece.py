from domino.base_piece import BasePiece
from .models import InputModel, OutputModel, EpochMetrics, ModelArchitecture
import os
import json
import random
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    mask = np.zeros(mask_rgb.shape[:2], dtype=np.int64)
    for rgb, cid in rgb_to_class.items():
        mask[np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=2)] = cid
    return mask


def _class_to_rgb(mask_cls, class_map):
    out = np.zeros((*mask_cls.shape, 3), dtype=np.uint8)
    for cid, rgb in class_map.items():
        out[mask_cls == cid] = rgb
    return out


class HistoDataset(Dataset):
    def __init__(self, samples, rgb_to_class, transform=None):
        self.samples = samples
        self.rgb_to_class = rgb_to_class
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = np.array(Image.open(s.image_path).convert("RGB"))
        mask_rgb = np.array(Image.open(s.mask_path).convert("RGB"))
        mask = _rgb_to_class(mask_rgb, self.rgb_to_class).astype(np.int64)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        return image, mask


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


def _build_model(arch, encoder_name, enc_weights, num_classes):
    kw = dict(encoder_name=encoder_name, encoder_weights=enc_weights,
              in_channels=3, classes=num_classes)
    mapping = {
        ModelArchitecture.UNET: smp.Unet,
        ModelArchitecture.UNET_PLUS_PLUS: smp.UnetPlusPlus,
        ModelArchitecture.FPN: smp.FPN,
        ModelArchitecture.DEEPLABV3_PLUS: smp.DeepLabV3Plus,
    }
    return mapping.get(arch, smp.Unet)(**kw)


# ── piece ──────────────────────────────────────────────────────────────────────

class HistoTrainingPiece(BasePiece):
    """
    Trains a 2D segmentation model on pre-split histopathology data.
    Receives train_samples and val_samples from HistoDataSplitPiece.
    Supports dry_run for fast end-to-end pipeline validation.
    """

    def piece_function(self, input_data: InputModel) -> OutputModel:
        try:
            self.logger.info("=" * 70)
            self.logger.info(f"HistoTrainingPiece  dry_run={input_data.dry_run}")
            self.logger.info("=" * 70)

            random.seed(input_data.random_seed)
            np.random.seed(input_data.random_seed)
            torch.manual_seed(input_data.random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(input_data.random_seed)

            # dry-run overrides
            num_epochs = input_data.num_epochs
            batch_size = input_data.batch_size
            train_samples = list(input_data.train_samples)
            val_samples = list(input_data.val_samples)
            if input_data.dry_run:
                self.logger.info("DRY RUN: epochs=1, batch_size=4, first 8 samples only")
                num_epochs = 1
                batch_size = 4
                train_samples = train_samples[:min(8, len(train_samples))]
                val_samples = val_samples[:min(4, len(val_samples))]

            # device
            if input_data.use_gpu and torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                self.logger.info("CPU mode")

            class_map, rgb_to_class = _parse_class_mapping(input_data.class_mapping_json)
            num_classes = len(class_map)
            h, w = input_data.image_height, input_data.image_width
            self.logger.info(
                f"Train={len(train_samples)}  Val={len(val_samples)}  "
                f"classes={num_classes}  size={h}x{w}"
            )

            # output dirs
            ckpt_dir = os.path.join(input_data.output_dir, "checkpoints")
            preds_dir = os.path.join(input_data.output_dir, "predictions")
            plots_dir = os.path.join(input_data.output_dir, "plots")
            for d in [input_data.output_dir, ckpt_dir, preds_dir, plots_dir]:
                os.makedirs(d, exist_ok=True)

            # transforms
            train_tf = A.Compose([
                A.Resize(h, w),
                *(
                    [A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
                     A.RandomBrightnessContrast(p=0.3), A.HueSaturationValue(p=0.3)]
                    if input_data.use_augmentation else []
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
            val_tf = A.Compose([
                A.Resize(h, w),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

            train_loader = DataLoader(
                HistoDataset(train_samples, rgb_to_class, train_tf),
                batch_size=batch_size, shuffle=True,
                num_workers=input_data.num_workers
            )
            val_loader = DataLoader(
                HistoDataset(val_samples, rgb_to_class, val_tf),
                batch_size=batch_size, shuffle=False,
                num_workers=input_data.num_workers
            )

            # model
            enc_w = None if input_data.encoder_weights.lower() == "none" \
                else input_data.encoder_weights
            model = _build_model(
                input_data.model_architecture,
                input_data.encoder_name, enc_w, num_classes
            ).to(device)

            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.AdamW(model.parameters(), lr=input_data.learning_rate)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=0.5,
                patience=input_data.lr_scheduler_patience
            )

            history = []
            best_val_dice = 0.0
            best_epoch = 0
            best_model_path = os.path.join(ckpt_dir, "best_model.pth")
            no_improve = 0

            for epoch in range(num_epochs):
                # ── train ──────────────────────────────────────────────────────
                model.train()
                tl, nt = 0.0, 0
                for images, masks in train_loader:
                    images, masks = images.to(device), masks.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), masks)
                    loss.backward()
                    optimizer.step()
                    tl += loss.item(); nt += 1
                avg_train_loss = tl / nt if nt else 0.0

                # ── validate ───────────────────────────────────────────────────
                model.eval()
                vl, nv = 0.0, 0
                dice_acc = DiceAccumulator(num_classes)
                saved = 0
                with torch.no_grad():
                    for b_idx, (images, masks) in enumerate(val_loader):
                        images, masks = images.to(device), masks.to(device)
                        logits = model(images)
                        vl += criterion(logits, masks).item(); nv += 1
                        preds = torch.argmax(logits, dim=1)
                        dice_acc.update(preds, masks)
                        if saved < input_data.max_saved_val_images:
                            n_save = min(input_data.max_saved_val_images - saved, preds.shape[0])
                            for i in range(n_save):
                                rgb = _class_to_rgb(preds[i].cpu().numpy(), class_map)
                                Image.fromarray(rgb).save(
                                    os.path.join(preds_dir, f"ep{epoch+1}_b{b_idx}_s{i}.png")
                                )
                            saved += n_save

                avg_val_loss = vl / nv if nv else 0.0
                dice_per_class, mean_dice = dice_acc.compute()
                lr = optimizer.param_groups[0]['lr']
                scheduler.step(mean_dice)

                history.append(EpochMetrics(
                    epoch=epoch + 1, train_loss=avg_train_loss,
                    val_loss=avg_val_loss, mean_dice=mean_dice,
                    dice_per_class=dice_per_class, learning_rate=lr
                ))
                self.logger.info(
                    f"Ep {epoch+1}/{num_epochs}  "
                    f"train={avg_train_loss:.4f}  val={avg_val_loss:.4f}  "
                    f"dice={mean_dice:.4f}  lr={lr:.2e}"
                )

                if mean_dice > best_val_dice:
                    best_val_dice = mean_dice
                    best_epoch = epoch + 1
                    no_improve = 0
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'mean_dice': mean_dice,
                        'arch': input_data.model_architecture.value,
                        'encoder_name': input_data.encoder_name,
                        'num_classes': num_classes,
                        'image_height': h,
                        'image_width': w,
                        'class_mapping': input_data.class_mapping_json,
                    }, best_model_path)
                    self.logger.info(f"  -> Best model (dice={mean_dice:.4f})")
                else:
                    no_improve += 1

                if (epoch + 1) % input_data.save_checkpoint_interval == 0:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state_dict': model.state_dict(),
                        'mean_dice': mean_dice,
                    }, os.path.join(ckpt_dir, f"ckpt_ep{epoch+1}.pth"))

                if (input_data.early_stopping_patience > 0
                        and no_improve >= input_data.early_stopping_patience):
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # final model
            final_path = os.path.join(input_data.output_dir, "final_model.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'arch': input_data.model_architecture.value,
                'encoder_name': input_data.encoder_name,
                'num_classes': num_classes,
                'image_height': h,
                'image_width': w,
                'class_mapping': input_data.class_mapping_json,
            }, final_path)

            with open(os.path.join(input_data.output_dir, "training_history.json"), 'w') as f:
                json.dump([m.model_dump() for m in history], f, indent=2)

            self._save_plots(history, plots_dir)

            summary = (
                f"HistoTrainingPiece\n{'='*50}\n"
                f"Arch         : {input_data.model_architecture.value}/{input_data.encoder_name}\n"
                f"Train/Val    : {len(train_samples)}/{len(val_samples)}\n"
                f"Epochs done  : {len(history)}\n"
                f"Best epoch   : {best_epoch}  (dice={best_val_dice:.4f})\n"
                f"Final loss   : {history[-1].train_loss:.4f}\n"
            )
            with open(os.path.join(input_data.output_dir, "training_summary.txt"), 'w') as f:
                f.write(summary)
            self.logger.info(summary)

            return OutputModel(
                model_path=final_path,
                checkpoint_dir=ckpt_dir,
                best_model_path=best_model_path,
                best_val_dice=best_val_dice,
                best_epoch=best_epoch,
                final_train_loss=history[-1].train_loss,
                total_epochs_trained=len(history),
                training_summary=summary.strip(),
                num_classes=num_classes,
                class_mapping_json=input_data.class_mapping_json,
                model_architecture=input_data.model_architecture.value,
                encoder_name=input_data.encoder_name,
                image_height=h,
                image_width=w,
            )

        except Exception as e:
            import traceback
            self.logger.error(f"HistoTrainingPiece error: {e}\n{traceback.format_exc()}")
            raise

    def _save_plots(self, history, plots_dir):
        epochs = [m.epoch for m in history]
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(epochs, [m.train_loss for m in history], label="Train Loss", marker='o', ms=3)
        axes[0].plot(epochs, [m.val_loss for m in history], label="Val Loss", marker='s', ms=3)
        axes[0].set(xlabel="Epoch", ylabel="CE Loss", title="Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)
        dices = [m.mean_dice for m in history]
        axes[1].plot(epochs, dices, label="Mean Dice", marker='o', ms=3, color='green')
        best = int(np.argmax(dices))
        axes[1].axvline(epochs[best], color='red', linestyle='--', label=f"Best: {dices[best]:.4f}")
        axes[1].set(xlabel="Epoch", ylabel="Mean Dice", title="Validation Dice"); axes[1].legend(); axes[1].grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, "training_curves.png"), dpi=150)
        plt.close()
