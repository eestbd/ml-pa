import torch
from pathlib import Path
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as TF

BATCH_SIZE = 4
NUM_WORKERS = 4
PIN_MEMORY = True
DATA_DIR = "./oxford_pet_data"
RANDOM_SEED = 42
USE_TRAIN_AUGMENTATION = True
VALIDATION_SPLIT_MODE = "random"  # "random" or "breed_holdout"
HOLDOUT_BREEDS = ()  # Example: ("Abyssinian", "american_bulldog")

IMAGE_SIZE = (384, 384)
TRAIN_RESIZE_SIZE = (448, 448)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]


def remap_mask(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.squeeze(0).long()
    # Oxford raw trimap labels are {1, 2, 3}; this task expects {0, 1, 2}.
    return torch.clamp(mask - 1, min=0, max=2)


class JointTrainTransform:
    def __init__(self):
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.05,
        )
        self.image_to_tensor = transforms.ToTensor()
        self.mask_to_tensor = transforms.PILToTensor()
        self.normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    def __call__(self, image, mask):
        image = TF.resize(image, TRAIN_RESIZE_SIZE, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, TRAIN_RESIZE_SIZE, interpolation=InterpolationMode.NEAREST)

        # Use one sampled crop box for both image and mask to keep labels aligned.
        top, left, height, width = transforms.RandomCrop.get_params(image, output_size=IMAGE_SIZE)
        image = TF.crop(image, top, left, height, width)
        mask = TF.crop(mask, top, left, height, width)

        # Use one sampled flip decision for both image and mask.
        if torch.rand(1).item() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Use one sampled rotation angle for both image and mask.
        angle = transforms.RandomRotation.get_params(degrees=(-10.0, 10.0))
        image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=2)

        image = self.color_jitter(image)
        image = self.normalize(self.image_to_tensor(image))
        mask = remap_mask(self.mask_to_tensor(mask))
        return image, mask


class JointValTransform:
    def __init__(self):
        self.image_to_tensor = transforms.ToTensor()
        self.mask_to_tensor = transforms.PILToTensor()
        self.normalize = transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD)

    def __call__(self, image, mask):
        image = TF.resize(image, IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, IMAGE_SIZE, interpolation=InterpolationMode.NEAREST)

        image = self.normalize(self.image_to_tensor(image))
        mask = remap_mask(self.mask_to_tensor(mask))
        return image, mask


class TransformDataset(Dataset):
    def __init__(self, subset, joint_transform):
        self.subset = subset
        self.joint_transform = joint_transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        image, mask = self.subset[idx]
        image, mask = self.joint_transform(image, mask)
        return image, mask


def normalize_breed_name(breed_name: str) -> str:
    return breed_name.strip().lower().replace(" ", "_").replace("-", "_")


def parse_breed_from_stem(image_stem: str) -> str:
    return normalize_breed_name(image_stem.rsplit("_", 1)[0])


def get_concat_source(concat_dataset: ConcatDataset, index: int):
    previous_size = 0
    for dataset, cumulative_size in zip(concat_dataset.datasets, concat_dataset.cumulative_sizes):
        if index < cumulative_size:
            return dataset, index - previous_size
        previous_size = cumulative_size
    raise IndexError(f"Index {index} out of range for dataset of size {len(concat_dataset)}")


def get_image_stem_from_concat(concat_dataset: ConcatDataset, index: int) -> str:
    dataset, local_index = get_concat_source(concat_dataset, index)
    image_paths = getattr(dataset, "_images", None)
    if image_paths is None:
        raise AttributeError("OxfordIIITPet dataset does not expose the expected '_images' attribute.")
    return Path(image_paths[local_index]).stem


def split_full_dataset(full_dataset: ConcatDataset):
    if VALIDATION_SPLIT_MODE == "random":
        total_size = len(full_dataset)
        train_size = int(0.9 * total_size)
        val_size = total_size - train_size
        generator = torch.Generator().manual_seed(RANDOM_SEED)
        return random_split(full_dataset, [train_size, val_size], generator=generator)

    if VALIDATION_SPLIT_MODE == "breed_holdout":
        holdout_breeds = {normalize_breed_name(breed) for breed in HOLDOUT_BREEDS}
        if not holdout_breeds:
            raise ValueError("HOLDOUT_BREEDS must be non-empty when using breed_holdout split.")

        train_indices = []
        val_indices = []
        for index in range(len(full_dataset)):
            breed = parse_breed_from_stem(get_image_stem_from_concat(full_dataset, index))
            if breed in holdout_breeds:
                val_indices.append(index)
            else:
                train_indices.append(index)

        if not train_indices or not val_indices:
            raise ValueError(
                "breed_holdout split produced an empty train or validation set. "
                f"Check HOLDOUT_BREEDS={HOLDOUT_BREEDS}."
            )
        return Subset(full_dataset, train_indices), Subset(full_dataset, val_indices)

    raise ValueError(f"Unsupported VALIDATION_SPLIT_MODE: {VALIDATION_SPLIT_MODE}")


base_trainval_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="trainval",
    target_types="segmentation",
    download=True,
    transform=None,
    target_transform=None,
)

base_test_dataset = datasets.OxfordIIITPet(
    root=DATA_DIR,
    split="test",
    target_types="segmentation",
    download=True,
    transform=None,
    target_transform=None,
)

full_dataset = ConcatDataset([base_trainval_dataset, base_test_dataset])

train_subset, val_subset = split_full_dataset(full_dataset)

train_joint_transform = JointTrainTransform() if USE_TRAIN_AUGMENTATION else JointValTransform()
val_joint_transform = JointValTransform()

train_set = TransformDataset(train_subset, train_joint_transform)
val_set = TransformDataset(val_subset, val_joint_transform)
final_train_set = TransformDataset(full_dataset, train_joint_transform)

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

final_train_loader = DataLoader(
    final_train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
)


if __name__ == "__main__":
    images, masks = next(iter(train_loader))

    print(f"USE_TRAIN_AUGMENTATION: {USE_TRAIN_AUGMENTATION}")
    print(f"Total dataset size: {len(full_dataset)}")
    print(f"Train dataset size: {len(train_set)}")
    print(f"Validation dataset size: {len(val_set)}")
    print(f"Final train dataset size: {len(final_train_set)}")
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    print(f"Image dtype: {images.dtype}")
    print(f"Mask dtype: {masks.dtype}")
    print(f"Mask unique values: {torch.unique(masks)}")
    print(f"Image min/max: {images.min().item():.4f} / {images.max().item():.4f}")
    print(f"Mask min/max: {masks.min().item()} / {masks.max().item()}")
