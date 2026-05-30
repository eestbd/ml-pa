import argparse
from pathlib import Path

import torch

from main import CLASS_NAMES, build_model, compute_iou_stats

NUM_CLASSES = 3


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate an ensemble on the validation split.")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--model", type=str, default="resnet34_unet", choices=["unet", "resnet34_unet"])
    parser.add_argument("--encoder-pretrained", action="store_true")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--tta-flip", action="store_true")
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


def load_models(checkpoint_paths, model_type, device):
    models = []
    for checkpoint_path in checkpoint_paths:
        checkpoint = load_checkpoint(Path(checkpoint_path), device)
        state_dict = strip_module_prefix(extract_state_dict(checkpoint))

        # Checkpoint weights are loaded immediately, so no ImageNet download is needed here.
        model = build_model(model_type, NUM_CLASSES, encoder_pretrained=False).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        models.append(model)
        print(f"Loaded checkpoint: {checkpoint_path}")
    return models


def predict_probs(model, images, use_tta_flip):
    logits = model(images)
    probs = torch.softmax(logits, dim=1)
    if not use_tta_flip:
        return probs

    flipped_images = torch.flip(images, dims=[3])
    flipped_logits = model(flipped_images)
    flipped_probs = torch.softmax(flipped_logits, dim=1)
    flipped_probs = torch.flip(flipped_probs, dims=[3])
    return (probs + flipped_probs) / 2.0


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
    print(f"encoder_pretrained_arg: {args.encoder_pretrained}")
    print(f"image_size: {args.image_size}")
    print(f"tta_flip: {args.tta_flip}")
    print(f"num_checkpoints: {len(args.checkpoints)}")

    from data import val_loader

    models = load_models(args.checkpoints, args.model, device)
    total_intersections = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)
    total_unions = torch.zeros(NUM_CLASSES, dtype=torch.float64, device=device)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(val_loader, start=1):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            prob_sum = None
            for model in models:
                probs = predict_probs(model, images, args.tta_flip)
                prob_sum = probs if prob_sum is None else prob_sum + probs

            ensemble_probs = prob_sum / len(models)
            preds = torch.argmax(ensemble_probs, dim=1)
            intersections, unions = compute_iou_stats(preds, masks, NUM_CLASSES)
            total_intersections += intersections
            total_unions += unions

            if batch_idx % 20 == 0:
                print(f"Evaluated {batch_idx} validation batches")

    mean_iou, class_ious = compute_iou_from_totals(total_intersections, total_unions)
    print(f"val_mIoU={format_metric(mean_iou)}")
    print(
        "class IoU: "
        + " ".join(
            f"{class_name}={format_metric(class_iou)}"
            for class_name, class_iou in zip(CLASS_NAMES, class_ious)
        )
    )


if __name__ == "__main__":
    main()
