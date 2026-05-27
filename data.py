import torch
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import ConcatDataset, DataLoader, random_split

BATCH_SIZE = 16
NUM_WORKERS = 4
PIN_MEMORY = True
IMAGE_SIZE = (224, 224)
DATA_DIR = "./oxford_pet_data"
RANDOM_SEED = 42


def remap_mask(mask):
    mask = mask.squeeze(0).long()
    mask = mask - 1
    return torch.clamp(mask, min=0, max=2)


image_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

mask_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.NEAREST),
    transforms.PILToTensor(),
    transforms.Lambda(remap_mask),
])


trainval_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="trainval",
    target_types="segmentation",
    download=True,
    transform=image_transform,
    target_transform=mask_transform,
)

test_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="test",
    target_types="segmentation",
    download=True,
    transform=image_transform,
    target_transform=mask_transform,
)

full_dataset = ConcatDataset([trainval_dataset, test_dataset])

total_size = len(full_dataset)
train_size = int(0.9 * total_size)
val_size = total_size - train_size

generator = torch.Generator().manual_seed(RANDOM_SEED)
train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(
    train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)

val_loader = DataLoader(
    val_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


if __name__ == "__main__":
    images, masks = next(iter(train_loader))

    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Train dataset size: {len(train_set)}")
    print(f"Validation dataset size: {len(val_set)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Mask dtype: {masks.dtype}")
    print(f"Mask unique values: {torch.unique(masks)}")
