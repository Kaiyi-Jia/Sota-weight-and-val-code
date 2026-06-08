"""
Emotion Classification Evaluation Script
=========================================
This script evaluates custom-trained CLIP-based models for 7-class or 8-class
emotion recognition. The model architecture is a modified version of CLIP
(ViT-L/14 as backbone, with changes to internal configurations). Weights are
obtained by training on that modified architecture.

Supports evaluating multiple weight files in a single run and exporting results
to CSV.

Usage:
    python evaluate.py --weights weight1.pth weight2.pth ... [options]

Example:
    python evaluate.py --weights affect7.pth affect8.pth --test_root D:/val --num_classes 7 --output results.csv
"""

import os
import argparse
import torch
import clip
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import csv
from datetime import datetime


# ==================== Configuration Defaults ====================
# Backbone model identifier (custom weights contain modified internal configurations)
DEFAULT_MODEL_NAME = "ViT-L/14"
DEFAULT_BATCH_SIZE = 64
DEFAULT_NUM_WORKERS = 0


# ==================== Emotion Definitions ====================
def get_emotions(num_classes: int):
    """
    Returns emotion labels based on number of classes.
    """
    if num_classes == 7:
        return ['Neutral', 'Happy', 'Sad', 'Surprised', 'Scared', 'Disgusted', 'Angry']
    elif num_classes == 8:
        return ['Neutral', 'Happy', 'Sad', 'Surprised', 'Scared', 'Disgusted', 'Angry', 'Contemptuous']
    else:
        raise ValueError(f"Unsupported number of classes: {num_classes}. Only 7 or 8 are allowed.")


# ==================== Core Functions ====================
def build_zeroshot_classifier(model, emotions, templates, device):
    """
    Build zero-shot classification weight matrix [embed_dim, num_classes]
    by encoding text prompts and averaging over templates.
    """
    with torch.no_grad():
        weights = []
        for emotion in emotions:
            texts = [t.format(emotion=emotion) for t in templates]
            text_tokens = clip.tokenize(texts).to(device)
            text_features = model.encode_text(text_tokens).float()

            # Handle 3D output (e.g. ViT with token dimension)
            if text_features.dim() == 3:
                text_features = text_features[:, -1, :]  # take last token

            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.mean(dim=0, keepdim=True)  # average across templates
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            weights.append(text_features)
        weights = torch.cat(weights, dim=0).T  # [embed_dim, num_classes]
    return weights


@torch.no_grad()
def evaluate(model, dataloader, zeroshot_weights, device):
    """
    Run evaluation and return overall accuracy, predictions, and labels.
    """
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_labels = []

    for images, labels in tqdm(dataloader, desc="Testing", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        with torch.cuda.amp.autocast(enabled=True):
            image_features = model.encode_image(images).float()
            if image_features.dim() == 3:
                image_features = image_features[:, 0, :]  # CLS token
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            logits = 100.0 * image_features @ zeroshot_weights
            predictions = logits.argmax(dim=-1)

        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        all_preds.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    accuracy = total_correct / total_samples
    return accuracy, np.array(all_preds), np.array(all_labels)


def print_per_class_results(predictions, labels, emotions):
    """
    Print per-class accuracy.
    """
    print("\n" + "=" * 50)
    print("Per-Class Accuracy:")
    for i, emotion in enumerate(emotions):
        mask = labels == i
        if mask.any():
            acc = np.mean(predictions[mask] == i)
            print(f"  [{i}] {emotion:<12}: {acc:.4%} ({np.sum(predictions[mask] == i)}/{np.sum(mask)})")
        else:
            print(f"  [{i}] {emotion:<12}: no samples")
    print("=" * 50 + "\n")


def test_single_weight(weight_path, test_root, num_classes, batch_size, num_workers, device):
    """
    Test a single weight file and return results dict.
    """
    print(f"\n{'='*60}")
    print(f"Testing weight: {weight_path}")
    print(f"{'='*60}")

    # Load model architecture
    model, preprocess = clip.load(DEFAULT_MODEL_NAME, device=device, jit=False)
    model.float()

    # Load custom weights
    state_dict = torch.load(weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("✅ Weights loaded successfully.")

    # Emotion labels and templates
    emotions = get_emotions(num_classes)
    templates = ["a {emotion} expression."]

    # Build classifier (text features are computed with current weight)
    zeroshot_weights = build_zeroshot_classifier(model, emotions, templates, device)

    # Load dataset
    full_dataset = ImageFolder(test_root, transform=preprocess)
    valid_indices = [i for i, (_, label) in enumerate(full_dataset.samples) if label in range(num_classes)]
    if len(valid_indices) == 0:
        raise RuntimeError(f"No valid samples found for classes 0~{num_classes-1} in {test_root}")

    dataset = Subset(full_dataset, valid_indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    print(f"Valid test samples: {len(dataset)} (classes 0~{num_classes-1})")

    # Evaluate
    print("🔥 Starting evaluation...")
    accuracy, predictions, labels = evaluate(model, dataloader, zeroshot_weights, device)

    # Print results
    print(f"\n🎯 Overall Accuracy ({num_classes}-class): {accuracy:.4%} ({np.sum(predictions == labels)}/{len(labels)})")
    print_per_class_results(predictions, labels, emotions)

    # Clean up
    del model
    torch.cuda.empty_cache()

    return {
        "weight": weight_path,
        "num_classes": num_classes,
        "accuracy": accuracy,
        "predictions": predictions,
        "labels": labels,
        "emotions": emotions
    }


def main():
    parser = argparse.ArgumentParser(description="CLIP Emotion Recognition Evaluation")
    parser.add_argument("--weights", type=str, nargs="+", required=True,
                        help="Path(s) to weight file(s). Multiple files can be provided.")
    parser.add_argument("--test_root", type=str, required=True,
                        help="Root directory of test dataset (with subfolders 0,1,...).")
    parser.add_argument("--num_classes", type=int, default=8, choices=[7, 8],
                        help="Number of emotion classes (7 or 8). Default: 8.")
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Batch size for evaluation.")
    parser.add_argument("--num_workers", type=int, default=DEFAULT_NUM_WORKERS,
                        help="Number of data loading workers.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (cuda/cpu).")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional CSV file to export summary results.")

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isdir(args.test_root):
        raise FileNotFoundError(f"Test root directory not found: {args.test_root}")

    for w in args.weights:
        if not os.path.isfile(w):
            raise FileNotFoundError(f"Weight file not found: {w}")

    # Run evaluations for each weight
    all_results = []
    for weight_path in args.weights:
        try:
            res = test_single_weight(
                weight_path=weight_path,
                test_root=args.test_root,
                num_classes=args.num_classes,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=args.device
            )
            all_results.append(res)
        except Exception as e:
            print(f"❌ Error testing {weight_path}: {e}")

    # Print summary
    if len(all_results) > 1:
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for r in all_results:
            print(f"{os.path.basename(r['weight']):30s} : {r['accuracy']:.4%}")

    # Export to CSV if requested
    if args.output and all_results:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["weight_file", "num_classes", "accuracy", "timestamp"])
            for r in all_results:
                writer.writerow([
                    os.path.basename(r["weight"]),
                    r["num_classes"],
                    f"{r['accuracy']:.6f}",
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ])
        print(f"\n📄 Results exported to {args.output}")


if __name__ == "__main__":
    main()
