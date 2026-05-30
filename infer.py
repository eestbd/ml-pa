import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from main import build_model

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
NUM_CLASSES = 3
EXPECTED_TEST_COUNT = 244


def parse_args():
    parser = argparse.ArgumentParser(description="Run U-Net inference and save per-image .npy masks.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/aug_ce_dice_plateau_30/best_unet.pth",
    )
    parser.add_argument("--test-dir", type=str, default="test_images")
    parser.add_argument("--sample", type=str, default="sample_submission.csv")
    parser.add_argument("--pred-dir", "--out-dir", dest="pred_dir", type=str, default="predictions")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--model", type=str, default="auto", choices=["auto", "unet", "resnet34_unet"])
    parser.add_argument("--encoder-pretrained", action="store_true")
    parser.add_argument("--tta-flip", action="store_true")
    return parser.parse_args()


def load_sample_rows(sample_path: Path):
    if not sample_path.is_file():
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_path}")

    rows = []
    with open(sample_path, "r", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"id", "height", "width"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Sample CSV missing columns: {sorted(missing)}")

        for row in reader:
            rows.append(
                {
                    "id": row["id"].strip(),
                    "height": int(row["height"]),
                    "width": int(row["width"]),
                }
            )
    return rows


def build_image_index(test_dir: Path):
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test image directory not found: {test_dir}")

    image_paths = sorted(
        path for path in test_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg"}
    )
    index = {}
    duplicates = []
    for path in image_paths:
        key = path.stem
        if key in index:
            duplicates.append(key)
        index[key] = path

    if duplicates:
        raise ValueError(f"Duplicate test image ids found: {duplicates[:5]}")
    return image_paths, index


def find_image_path(image_index, sample_id: str):
    candidates = [sample_id, Path(sample_id).stem]
    for candidate in candidates:
        if candidate in image_index:
            return image_index[candidate]
    raise FileNotFoundError(f"No matching jpg found for sample id: {sample_id}")


def build_preprocess(image_size: int):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_checkpoint(path: Path, device: torch.device):
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint must be a dict containing 'model_state_dict'.")
    return checkpoint


def normalize_model_type(model_type):
    if model_type in (None, "", "basic_unet"):
        return "unet"
    if model_type in {"unet", "resnet34_unet"}:
        return model_type
    raise ValueError(f"Unsupported checkpoint model type: {model_type}")


def resolve_model_type(args, checkpoint):
    if args.model != "auto":
        return args.model
    return normalize_model_type(checkpoint.get("model_type") or checkpoint.get("model_name"))


def resolve_image_size(args, checkpoint):
    if args.image_size is not None:
        return args.image_size

    checkpoint_image_size = checkpoint.get("image_size", 224)
    if isinstance(checkpoint_image_size, (tuple, list)):
        return int(checkpoint_image_size[0])
    return int(checkpoint_image_size)


def predict_logits(model, image_tensor: torch.Tensor, use_tta_flip: bool):
    logits = model(image_tensor)
    if not use_tta_flip:
        return logits

    flipped_tensor = torch.flip(image_tensor, dims=[3])
    flipped_logits = model(flipped_tensor)
    flipped_logits = torch.flip(flipped_logits, dims=[3])
    return (logits + flipped_logits) / 2.0


def validate_prediction(pred: np.ndarray, expected_shape, sample_id: str):
    if pred.shape != expected_shape:
        raise ValueError(f"{sample_id}: prediction shape {pred.shape} != expected {expected_shape}")

    unique_values = np.unique(pred)
    if not np.all((unique_values >= 0) & (unique_values < NUM_CLASSES)):
        raise ValueError(f"{sample_id}: invalid labels {unique_values.tolist()}")


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint)
    test_dir = Path(args.test_dir)
    sample_path = Path(args.sample)
    pred_dir = Path(args.pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    rows = load_sample_rows(sample_path)
    image_paths, image_index = build_image_index(test_dir)

    print(f"Found test images: {len(image_paths)}")
    print(f"Sample rows: {len(rows)}")
    if len(rows) != EXPECTED_TEST_COUNT:
        raise ValueError(f"Expected {EXPECTED_TEST_COUNT} sample rows, got {len(rows)}")

    missing_ids = []
    for row in rows:
        try:
            find_image_path(image_index, row["id"])
        except FileNotFoundError:
            missing_ids.append(row["id"])
    if missing_ids:
        raise FileNotFoundError(f"Missing jpg files for ids: {missing_ids[:5]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_type = resolve_model_type(args, checkpoint)
    image_size = resolve_image_size(args, checkpoint)

    # The checkpoint supplies trained weights, so inference does not need to download ImageNet weights.
    model = build_model(model_type, NUM_CLASSES, encoder_pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocess = build_preprocess(image_size)
    print(f"Device: {device}")
    print(f"Model: {model_type}")
    print(f"Checkpoint encoder_pretrained: {checkpoint.get('encoder_pretrained')}")
    print(f"CLI encoder_pretrained: {args.encoder_pretrained}")
    print(f"Image size: {image_size}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Prediction dir: {pred_dir}")
    print(f"TTA horizontal flip: {args.tta_flip}")

    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            sample_id = row["id"]
            height = row["height"]
            width = row["width"]
            image_path = find_image_path(image_index, sample_id)

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = preprocess(image).unsqueeze(0).to(device)

            logits = predict_logits(model, image_tensor, args.tta_flip)
            logits = F.interpolate(
                logits,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
            pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            validate_prediction(pred, (height, width), sample_id)
            np.save(pred_dir / f"{sample_id}.npy", pred)

            if index == 1:
                print(
                    f"First prediction: id={sample_id}, "
                    f"shape={pred.shape}, labels={np.unique(pred).tolist()}"
                )
            if index % 50 == 0 or index == len(rows):
                print(f"Processed {index}/{len(rows)}")

    npy_files = sorted(pred_dir.glob("*.npy"))
    print(f"Saved npy files: {len(npy_files)}")
    if len(npy_files) != EXPECTED_TEST_COUNT:
        raise ValueError(f"Expected {EXPECTED_TEST_COUNT} .npy files, got {len(npy_files)}")
    print("Inference completed successfully.")


if __name__ == "__main__":
    main()
