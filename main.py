import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

RANDOM_SEED = 42


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


class Trainer:
    """Training / validation loop wrapper."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
    ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

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
    def validate(self, loader, max_batches=None) -> float:
        self.model.eval()
        total_loss = 0.0
        total_samples = 0

        for batch_idx, (images, masks) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            images = images.to(self.device, non_blocking=True)
            masks = masks.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, masks)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples if total_samples > 0 else 0.0


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    set_seed(RANDOM_SEED)

    from data import train_loader, val_loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    num_epochs = 1
    learning_rate = 1e-4
    max_train_batches = 5
    max_val_batches = 2

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = SegmentationLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, criterion, optimizer, device)

    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    print(f"device: {device}")
    print(f"total trainable parameters: {trainable_params:,}")
    print(f"num_epochs: {num_epochs}")
    print(f"learning_rate: {learning_rate}")

    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch(train_loader, max_batches=max_train_batches)
        val_loss = trainer.validate(val_loader, max_batches=max_val_batches)
        print(f"[Epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")


if __name__ == "__main__":
    main()
