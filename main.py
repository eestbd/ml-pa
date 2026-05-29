import argparse
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

RANDOM_SEED = 42
CLASS_NAMES = ("foreground", "background", "boundary")
TRAIN_LOG_FIELDNAMES = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_miou",
    "iou_foreground",
    "iou_background",
    "iou_boundary",
    "current_lr",
    "loss_type",
    "ce_weight",
    "dice_weight",
    "scheduler",
    "is_best",
]


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


class DiceLoss(nn.Module):
    """Multi-class Dice loss computed from raw segmentation logits."""

    def __init__(self, num_classes: int, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets.long(), num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersections = torch.sum(probs * targets_one_hot, dim=dims)
        pred_sums = torch.sum(probs, dim=dims)
        target_sums = torch.sum(targets_one_hot, dim=dims)
        dice_scores = (2.0 * intersections + self.smooth) / (pred_sums + target_sums + self.smooth)

        return 1.0 - dice_scores.mean()


class SegmentationLoss(nn.Module):
    """Cross entropy or combined cross entropy and Dice loss."""

    def __init__(
        self,
        loss_type: str = "ce_dice",
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        num_classes: int = 3,
    ):
        super().__init__()
        if loss_type not in {"ce", "ce_dice"}:
            raise ValueError(f"Unsupported loss_type: {loss_type}")
        self.loss_type = loss_type
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss(num_classes=num_classes)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce_loss(logits, targets)
        if self.loss_type == "ce":
            return ce

        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


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
    parser.add_argument("--loss", type=str, default="ce_dice", choices=["ce", "ce_dice"])
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--scheduler", type=str, default="plateau", choices=["none", "plateau"])
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--plateau-patience", type=int, default=2)
    parser.add_argument("--plateau-threshold", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    return parser.parse_args()


def resolve_log_path(log_path: str) -> str:
    if not os.path.exists(log_path) or os.path.getsize(log_path) == 0:
        return log_path

    with open(log_path, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        existing_header = next(reader, [])

    if existing_header == TRAIN_LOG_FIELDNAMES:
        return log_path

    base, ext = os.path.splitext(log_path)
    version = 2
    while True:
        candidate = f"{base}_v{version}{ext}"
        if not os.path.exists(candidate) or os.path.getsize(candidate) == 0:
            return candidate
        with open(candidate, mode="r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            existing_header = next(reader, [])
        if existing_header == TRAIN_LOG_FIELDNAMES:
            return candidate
        version += 1


def get_current_lr(optimizer: optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def build_scheduler(optimizer: optim.Optimizer, args):
    if args.scheduler == "none":
        return None

    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.plateau_factor,
        patience=args.plateau_patience,
        threshold=args.plateau_threshold,
        min_lr=args.min_lr,
    )


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
    loss_type: str,
    ce_weight: float,
    dice_weight: float,
    scheduler_type: str,
    scheduler_state_dict,
    current_lr: float,
    plateau_factor: float,
    plateau_patience: int,
    plateau_threshold: float,
    min_lr: float,
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
        "loss_type": loss_type,
        "ce_weight": ce_weight,
        "dice_weight": dice_weight,
        "scheduler_type": scheduler_type,
        "scheduler_state_dict": scheduler_state_dict,
        "current_lr": current_lr,
        "plateau_factor": plateau_factor,
        "plateau_patience": plateau_patience,
        "plateau_threshold": plateau_threshold,
        "min_lr": min_lr,
    }


def append_train_log(
    log_path: str,
    epoch: int,
    train_loss: float,
    val_loss: float,
    val_miou: float,
    class_ious,
    current_lr: float,
    loss_type: str,
    ce_weight: float,
    dice_weight: float,
    scheduler_type: str,
    is_best: bool,
) -> None:
    needs_header = not os.path.exists(log_path) or os.path.getsize(log_path) == 0

    with open(log_path, mode="a", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=TRAIN_LOG_FIELDNAMES)
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
                "current_lr": current_lr,
                "loss_type": loss_type,
                "ce_weight": ce_weight,
                "dice_weight": dice_weight,
                "scheduler": scheduler_type,
                "is_best": is_best,
            }
        )


def main():
    args = parse_args()
    set_seed(RANDOM_SEED)

    num_classes = 3
    num_epochs = args.epochs
    learning_rate = args.lr
    checkpoint_dir = args.checkpoint_dir
    log_dir = args.log_dir
    loss_type = args.loss
    ce_weight = args.ce_weight
    dice_weight = args.dice_weight
    scheduler_type = args.scheduler
    model_name = "basic_unet"
    max_train_batches = 5 if args.quick else None
    max_val_batches = 2 if args.quick else None
    mode = "quick debug" if args.quick else "full training"
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_unet.pth")
    last_checkpoint_path = os.path.join(checkpoint_dir, "last_unet.pth")
    log_path = os.path.join(log_dir, f"train_log_{loss_type}_{scheduler_type}.csv")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    log_path = resolve_log_path(log_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = SegmentationLoss(
        loss_type=loss_type,
        ce_weight=ce_weight,
        dice_weight=dice_weight,
        num_classes=num_classes,
    )
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = build_scheduler(optimizer, args)
    trainer = Trainer(model, criterion, optimizer, device, num_classes=num_classes)
    best_miou = -1.0

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"device: {device}")
    print(f"mode: {mode}")
    print(f"total trainable parameters: {trainable_params:,}")
    print(f"num_epochs: {num_epochs}")
    print(f"learning_rate: {learning_rate}")
    print(f"loss_type: {loss_type}")
    print(f"ce_weight: {ce_weight}")
    print(f"dice_weight: {dice_weight}")
    print(f"scheduler: {scheduler_type}")
    print(f"plateau_factor: {args.plateau_factor}")
    print(f"plateau_patience: {args.plateau_patience}")
    print(f"plateau_threshold: {args.plateau_threshold}")
    print(f"min_lr: {args.min_lr}")
    print(f"checkpoint_dir: {display_path(checkpoint_dir)}")
    print(f"log_dir: {display_path(log_dir)}")
    print(f"log_file: {display_path(log_path)}")

    from data import IMAGE_SIZE as DATA_IMAGE_SIZE, train_loader, val_loader

    image_size = DATA_IMAGE_SIZE[0] if isinstance(DATA_IMAGE_SIZE, (tuple, list)) else DATA_IMAGE_SIZE

    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch(train_loader, max_batches=max_train_batches)
        val_loss, val_miou, class_ious = trainer.validate(val_loader, max_batches=max_val_batches)
        is_best = val_miou > best_miou
        if is_best:
            best_miou = val_miou

        old_lr = get_current_lr(optimizer)
        if scheduler is not None:
            scheduler.step(val_miou)
        current_lr = get_current_lr(optimizer)
        lr_reduced = old_lr > current_lr

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
            loss_type=loss_type,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            scheduler_type=scheduler_type,
            scheduler_state_dict=scheduler.state_dict() if scheduler is not None else None,
            current_lr=current_lr,
            plateau_factor=args.plateau_factor,
            plateau_patience=args.plateau_patience,
            plateau_threshold=args.plateau_threshold,
            min_lr=args.min_lr,
        )

        print(
            f"[Epoch {epoch + 1}/{num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_mIoU={format_metric(val_miou)} "
            f"lr={current_lr:.6f}"
        )
        class_iou_text = " ".join(
            f"{class_name}={format_metric(class_iou)}"
            for class_name, class_iou in zip(CLASS_NAMES, class_ious)
        )
        print(f"class IoU: {class_iou_text}")
        if lr_reduced:
            print(f"Learning rate reduced: {old_lr:.6f} -> {current_lr:.6f}")

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
            current_lr=current_lr,
            loss_type=loss_type,
            ce_weight=ce_weight,
            dice_weight=dice_weight,
            scheduler_type=scheduler_type,
            is_best=is_best,
        )


if __name__ == "__main__":
    main()
