import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

RANDOM_SEED = 42
CLASS_NAMES = ("foreground", "background", "boundary")


class DoubleConv(nn.Module):
    """Two convolution blocks that preserve spatial resolution."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """UNet architecture for image segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 3):
        super().__init__()
        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)
        self.encoder4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = DoubleConv(512, 1024)

        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(128, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    @staticmethod
    def _concat_with_skip(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([skip, x], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool(x1))
        x3 = self.encoder3(self.pool(x2))
        x4 = self.encoder4(self.pool(x3))

        x = self.bottleneck(self.pool(x4))

        x = self.up4(x)
        x = self._concat_with_skip(x, x4)
        x = self.decoder4(x)

        x = self.up3(x)
        x = self._concat_with_skip(x, x3)
        x = self.decoder3(x)

        x = self.up2(x)
        x = self._concat_with_skip(x, x2)
        x = self.decoder2(x)

        x = self.up1(x)
        x = self._concat_with_skip(x, x1)
        x = self.decoder1(x)

        return self.final_conv(x)


class SegmentationLoss(nn.Module):
    """Cross entropy loss for multi-class semantic segmentation."""

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


def compute_iou_stats(preds: torch.Tensor, targets: torch.Tensor, num_classes: int):
    intersections = torch.zeros(num_classes, dtype=torch.float64, device=preds.device)
    unions = torch.zeros(num_classes, dtype=torch.float64, device=preds.device)

    # Accumulate pixel-level intersection and union per class for dataset-level mIoU.
    for class_idx in range(num_classes):
        pred_mask = preds == class_idx
        target_mask = targets == class_idx
        intersections[class_idx] = torch.logical_and(pred_mask, target_mask).sum()
        unions[class_idx] = torch.logical_or(pred_mask, target_mask).sum()

    return intersections, unions


class Trainer:
    """Training / validation loop wrapper."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        num_classes: int,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes

    def train_one_epoch(self, loader, max_batches=None) -> float:
        self.model.train()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (images, masks) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, masks)
            loss.backward()
            self.optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples if total_samples > 0 else 0.0

    @torch.no_grad()
    def validate(self, loader, max_batches=None):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        total_intersections = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)
        total_unions = torch.zeros(self.num_classes, dtype=torch.float64, device=self.device)

        for batch_idx, (images, masks) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            preds = torch.argmax(logits, dim=1)
            intersections, unions = compute_iou_stats(preds, masks, self.num_classes)
            total_intersections += intersections
            total_unions += unions

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        class_ious = []
        valid_ious = []

        for class_idx in range(self.num_classes):
            union = total_unions[class_idx].item()
            if union > 0:
                iou = total_intersections[class_idx].item() / union
                valid_ious.append(iou)
            else:
                iou = float("nan")
            class_ious.append(iou)

        mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else float("nan")
        return avg_loss, mean_iou, class_ious


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_metric(value: float) -> str:
    return "nan" if value != value else f"{value:.4f}"


def display_path(path: str) -> str:
    return path.replace("\\", "/")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a basic U-Net for Oxford-IIIT Pet segmentation.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


def build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: optim.Optimizer,
    best_miou: float,
    val_miou: float,
    val_loss: float,
    class_ious,
    num_classes: int,
    learning_rate: float,
    num_epochs: int,
    model_name: str,
    image_size: int,
):
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_miou": best_miou,
        "val_miou": val_miou,
        "val_loss": val_loss,
        "class_ious": class_ious,
        "num_classes": num_classes,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "model_name": model_name,
        "image_size": image_size,
    }


def append_train_log(
    log_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_miou: float,
    class_ious,
    learning_rate: float,
    is_best: bool,
) -> None:
    fieldnames = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_miou",
        "iou_foreground",
        "iou_background",
        "iou_boundary",
        "learning_rate",
        "is_best",
    ]
    needs_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    with open(log_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_miou": val_miou,
                "iou_foreground": class_ious[0],
                "iou_background": class_ious[1],
                "iou_boundary": class_ious[2],
                "learning_rate": learning_rate,
                "is_best": is_best,
            }
        )


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    num_classes = 3
    image_size = 224
    num_epochs = args.epochs
    learning_rate = args.lr
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    model_name = "basic_unet"
    max_train_batches = 5 if args.quick else None
    max_val_batches = 2 if args.quick else None
    mode = "quick debug" if args.quick else "full training"
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_unet.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_unet.pth")
    log_path = os.path.join(log_dir, "train_log.csv")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = SegmentationLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, criterion, optimizer, device, num_classes=num_classes)
    best_miou = -1.0

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"device: {device}")
    print(f"mode: {mode}")
    print(f"total trainable parameters: {trainable_params:,}")
    print(f"num_epochs: {num_epochs}")
    print(f"learning_rate: {learning_rate}")
    print(f"checkpoint_dir: {display_path(checkpoint_dir)}")
    print(f"log_dir: {display_path(log_dir)}")

    from data import train_loader, val_loader

    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch(train_loader, max_batches=max_train_batches)
        val_loss, val_miou, class_ious = trainer.validate(val_loader, max_batches=max_val_batches)
        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou

        checkpoint = build_checkpoint(
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            best_miou=best_miou,
            val_miou=val_miou,
            val_loss=val_loss,
            class_ious=class_ious,
            num_classes=num_classes,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            model_name=model_name,
            image_size=image_size,
        )

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_mIoU={format_metric(val_miou)}"
        )
        class_iou_text = " ".join(
            f"{class_name}={format_metric(class_iou)}"
            for class_name, class_iou in zip(CLASS_NAMES, class_ious)
        )
        print(f"class IoU: {class_iou_text}")

        if is_best:
            torch.save(checkpoint, best_checkpoint_path)
            print(
                f"New best model saved: {display_path(best_checkpoint_path)} "
                f"(val_mIoU={format_metric(val_miou)})"
            )

        torch.save(checkpoint, last_checkpoint_path)
        print(f"Last checkpoint saved: {display_path(last_checkpoint_path)}")

        append_train_log(
            log_path=log_path,
            epoch=epoch + 1,
            train_loss=train_loss,
            val_loss=val_loss,
            val_miou=val_miou,
            class_ious=class_ious,
            learning_rate=learning_rate,
            is_best=is_best,
        )


if __name__ == "__main__":
    main()
