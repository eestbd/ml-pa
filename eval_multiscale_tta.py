import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from main import CLASS_NAMES, build_model, compute_iou_stats

NUM_CLASSES = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-scale hflip TTA on the validation split.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/resnet34_unet320_ce_dice_plateau_40/best_unet.pth",
    )
    parser.add_argument("--model", type=str, default="resnet34_unet", choices=["unet", "resnet34_unet"])
    parser.add_argument("--image-sizes", type=int, nargs="+", default=[288, 320, 352])
    return parser.parse_args()


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


def load_checkpoint(path: Path, device):
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def load_model(checkpoint_path: Path, model_type: str, device):
    checkpoint = load_checkpoint(checkpoint_path, device)
    state_dict = strip_module_prefix(extract_state_dict(checkpoint))

    # Checkpoint weights are loaded immediately, so no ImageNet download is needed here.
    model = build_model(model_type, NUM_CLASSES, encoder_pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_multiscale_probs(model, images, image_sizes, target_size):
    prob_sum = None

    for image_size in image_sizes:
        scaled_images = F.interpolate(
            images,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )

        logits = model(scaled_images)
        probs = torch.softmax(logits, dim=1)

        flipped_images = torch.flip(scaled_images, dims=[3])
        flipped_logits = model(flipped_images)
        flipped_probs = torch.softmax(flipped_logits, dim=1)
        flipped_probs = torch.flip(flipped_probs, dims=[3])

        probs = (probs + flipped_probs) / 2.0
        probs = F.interpolate(probs, size=target_size, mode="bilinear", align_corners=False)
        prob_sum = probs if prob_sum is None else prob_sum + probs

    return prob_sum / len(image_sizes)


def compute_iou_from_totals(total_intersections, total_unions):
    class_ious = []
    valid_ious = []

    for class_idx in range(NUM_CLASSES):
        union = total_unions[class_idx].item()
        if union > 0:
            iou = total_intersections[class_idx].item() / union
            valid_ious.append(iou)
        else:
            iou = float("nan")
        class_ious.append(iou)

    mean_iou = sum(valid_ious) / len(valid_ious) if valid_ious else float("nan")
    return mean_iou, class_ious


def format_metric(value):
    return "nan" if value != value else f"{value:.4f}"


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"image_sizes: {args.image_sizes}")

    from data import val_loader

    model = load_model(Path(args.checkpoint), args.model, device)
    total_intersections = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    total_unions = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            probs = predict_multiscale_probs(
                model=model,
                images=images,
                image_sizes=args.image_sizes,
                target_size=masks.shape[-2:],
            )
            preds = torch.argmax(probs, dim=1)
            intersections, unions = compute_iou_stats(preds, masks, NUM_CLASSES)
            total_intersections += intersections
            total_unions += unions

            if batch_idx % 20 == 0:
                print(f"Evaluated {batch_idx} validation batches")

    mean_iou, class_ious = compute_iou_from_totals(total_intersections, total_unions)

    print(f"image_sizes: {args.image_sizes}")
    print(f"val_mIoU: {format_metric(mean_iou)}")
    print(f"foreground IoU: {format_metric(class_ious[0])}")
    print(f"background IoU: {format_metric(class_ious[1])}")
    print(f"boundary IoU: {format_metric(class_ious[2])}")
    print(
        "class IoU: "
        + " ".join(
            f"{class_name}={format_metric(class_iou)}"
            for class_name, class_iou in zip(CLASS_NAMES, class_ious)
        )
    )


if __name__ == "__main__":
    main()
