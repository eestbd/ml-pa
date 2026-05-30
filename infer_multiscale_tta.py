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
    parser = argparse.ArgumentParser(description="Run multi-scale hflip TTA inference for Kaggle submission.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/resnet34_unet320_ce_dice_plateau_40/best_unet.pth",
    )
    parser.add_argument("--test-dir", type=str, default="test_images")
    parser.add_argument("--sample", type=str, default="sample_submission.csv")
    parser.add_argument("--pred-dir", "--out-dir", dest="pred_dir", type=str, default="predictions_multiscale_tta")
    parser.add_argument("--model", type=str, default="resnet34_unet", choices=["unet", "resnet34_unet"])
    parser.add_argument("--encoder-pretrained", action="store_true")
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[288, 320, 352])
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

    image_paths = sorted(path for path in test_dir.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg"})
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
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if value is None:
                continue
            if hasattr(value, "state_dict"):
                return value.state_dict()
            if isinstance(value, dict):
                return value
    return checkpoint


def strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        raise ValueError("Loaded checkpoint does not contain a state_dict-like object.")
    if not any(key.startswith("module.") for key in state_dict.keys()):
        return state_dict
    return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}


def predict_multiscale_probs(model, image: Image.Image, image_sizes, height: int, width: int, device: torch.device):
    prob_sum = None

    for image_size in image_sizes:
        preprocess = build_preprocess(image_size)
        image_tensor = preprocess(image).unsqueeze(0).to(device)

        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)

        flipped_tensor = torch.flip(image_tensor, dims=[3])
        flipped_logits = model(flipped_tensor)
        flipped_probs = torch.softmax(flipped_logits, dim=1)
        flipped_probs = torch.flip(flipped_probs, dims=[3])

        probs = (probs + flipped_probs) / 2.0
        probs = F.interpolate(probs, size=(height, width), mode="bilinear", align_corners=False)
        prob_sum = probs if prob_sum is None else prob_sum + probs

    return prob_sum / len(image_sizes)


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
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))

    # The checkpoint already contains trained weights, so avoid downloading ImageNet weights at inference time.
    model = build_model(args.model, NUM_CLASSES, encoder_pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Checkpoint encoder_pretrained: {checkpoint.get('encoder_pretrained') if isinstance(checkpoint, dict) else None}")
    print(f"CLI encoder_pretrained: {args.encoder_pretrained}")
    print(f"Image sizes: {args.image_sizes}")
    print("TTA horizontal flip: True")
    print(f"Prediction dir: {pred_dir}")

    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            sample_id = row["id"]
            height = row["height"]
            width = row["width"]
            image_path = find_image_path(image_index, sample_id)

            with Image.open(image_path) as image:
                image = image.convert("RGB")
                avg_probs = predict_multiscale_probs(model, image, args.image_sizes, height, width, device)

            pred = torch.argmax(avg_probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
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
    print("Multi-scale TTA inference completed successfully.")


if __name__ == "__main__":
    main()
