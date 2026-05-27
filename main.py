import torch
import torch.nn as nn
import torch.optim as optim

from data import train_loader, val_loader


class UNet(nn.Module):
    """UNet architecture for image segmentation."""

    def __init__(self, in_channels: int = 3, num_classes: int = 1):
        super().__init__()
        # TODO: define encoder (contracting path)

        # TODO: define bottleneck (e.g., 512 -> 1024)

        # TODO: define decoder (expanding path)

        # TODO: define final 1x1 conv (out_channels = num_classes)
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: encoder forward, store feature maps for skip connections
        # TODO: bottleneck forward
        # TODO: decoder forward, concat with skip connections
        # TODO: return segmentation logits via final 1x1 conv
        raise NotImplementedError


class SegmentationLoss(nn.Module):
    """Loss function for segmentation (e.g., BCE / CrossEntropy / Dice / combined)."""

    def __init__(self):
        super().__init__()
        # TODO: define the loss to use
        #   - BCEWithLogitsLoss for binary segmentation
        #   - CrossEntropyLoss for multi-class segmentation
        #   - optionally combine with Dice loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # TODO: compute and return loss from logits and targets
        raise NotImplementedError


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

    def train_one_epoch(self, loader) -> float:
        self.model.train()
        # TODO: iterate over (images, masks) batches from loader
        #   1) move tensors to device
        #   2) optimizer.zero_grad()
        #   3) forward -> compute loss
        #   4) loss.backward() -> optimizer.step()
        #   5) accumulate and return average loss
        raise NotImplementedError

    @torch.no_grad()
    def validate(self, loader) -> float:
        self.model.eval()
        # TODO: run forward only and compute loss / metrics (IoU, Dice, etc.)
        raise NotImplementedError


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: set hyperparameters (lr, num_epochs, num_classes, etc.)
    num_epochs = 10
    learning_rate = 1e-4
    num_classes = 1

    model = UNet(in_channels=3, num_classes=num_classes).to(device)
    criterion = SegmentationLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    trainer = Trainer(model, criterion, optimizer, device)

    for epoch in range(num_epochs):
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss = trainer.validate(val_loader)
        print(f"[Epoch {epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    # TODO: save best model checkpoint / visualize predictions / report metrics


if __name__ == "__main__":
    main()
