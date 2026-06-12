import os
import torch
from tqdm import tqdm

from config import Config
from model import BoostingFER
from dataset import get_dataloaders

WEIGHT_PATH = r""   # Path to the saved weights


def main():
    config = Config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    num_classes = config.class_num
    class_names = config.class_names

    print(f"Starting model evaluation. Device: {device}, Number of classes: {num_classes}")

    # Initialize the model and load weights
    dummy_freqs = [100] * num_classes
    model = BoostingFER(config, dummy_freqs, class_names).to(device)

    checkpoint = torch.load(WEIGHT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()

    # Compute semantic anchors
    model.compute_semantic_anchors(device)

    # Prepare the test dataset
    _, val_loader, _ = get_dataloaders(config, num_classes=config.class_num)
    dataset_size = len(val_loader.dataset)
    print(f"📸 Total test samples loaded: {dataset_size}\n")

    total_correct = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)

            with torch.cuda.amp.autocast(enabled=config.use_amp):
                logits = model(images, return_loss=False)
                preds = logits.argmax(dim=1)

            total_correct += (preds == labels).sum().item()

    # Output final accuracy
    overall_acc = total_correct / dataset_size
    print("\n" + "=" * 55)
    print(f"Final test set accuracy: {overall_acc:.4%} ({total_correct}/{dataset_size})")
    print("=" * 55 + "\n")


if __name__ == '__main__':
    main()
