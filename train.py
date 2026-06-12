import os
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import Config
from dataset import get_dataloaders
from model import BoostingFER
from utils import set_seed, save_checkpoint


def train_one_epoch(model, loader, optimizer, scaler, writer, epoch, device, use_amp=False):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # Gradient accumulation steps to simulate a larger batch size on limited VRAM
    accumulation_steps = 8

    optimizer.zero_grad()  # Initialize zero gradients outside the loop

    pbar = tqdm(loader, desc=f"Train Epoch [{epoch}]")
    for i, (images, targets) in enumerate(pbar):
        images, targets = images.to(device), targets.to(device)

        # Automatic Mixed Precision (AMP) training
        if use_amp:
            with autocast():
                logits, loss, loss_dict = model(images, return_loss=True, targets=targets)
                # Divide loss by accumulation steps to maintain correct gradient scaling
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()
            # if i == 0:
            #     # Check if the fine_classifier in EHSA has gradients
            #     grad_sum = model.ehsa.fine_classifier.weight.grad.abs().sum().item()
            #     print(f"\n[DEBUG] 🚀 Total gradients for EHSA classification head: {grad_sum:.4f}")

            # Parameter update
            if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(loader)):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()  # Clear gradients after update
        else:
            logits, loss, loss_dict = model(images, return_loss=True, targets=targets)
            loss = loss / accumulation_steps
            loss.backward()

            if ((i + 1) % accumulation_steps == 0) or (i + 1 == len(loader)):
                optimizer.step()
                optimizer.zero_grad()

        actual_loss = loss.item() * accumulation_steps
        total_loss += actual_loss

        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

        pbar.set_postfix({
            'Loss': f"{actual_loss:.4f}",
            'Acc': f"{total_correct / total_samples:.4f}"
        })

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total_samples

    if writer:
        writer.add_scalar('Train/Loss_Total', avg_loss, epoch)
        writer.add_scalar('Train/Accuracy', accuracy, epoch)
        for k, v in loss_dict.items():
            writer.add_scalar(f'Train/Loss_{k.upper()}', v.item(), epoch)

    return avg_loss, accuracy


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_correct = 0
    total_samples = 0

    for images, targets in tqdm(loader, desc="Validation"):
        images, targets = images.to(device), targets.to(device)

        # Testing phase: return_loss=False and targets are not needed
        logits = model(images, return_loss=False)
        preds = logits.argmax(dim=1)

        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

    return total_correct / total_samples


def main():
    config = Config()

    # Initialize directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)

    set_seed(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} | AMP Enabled: {config.use_amp}")

    # Get dataloaders and class frequency statistics
    train_loader, val_loader, class_freqs = get_dataloaders(config, num_classes=config.class_num)
    class_names = config.class_names  # Assuming this is defined in Config

    # Initialize the model
    model = BoostingFER(config, class_freqs, class_names).to(device)

    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),  # Only optimize parameters with requires_grad=True
        lr=config.lr,
        weight_decay=config.weight_decay,
        eps=config.epsilon
    )

    # Pre-compute semantic anchors
    print("Computing Semantic Anchors (t_global, t_local)...")
    model.compute_semantic_anchors(device)
    print("Semantic Anchors Cached!")

    scaler = GradScaler(enabled=config.use_amp)
    writer = SummaryWriter(log_dir=config.log_dir)

    best_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, scaler, writer, epoch, device, config.use_amp
        )
        print(f"-> Epoch [{epoch}/{config.epochs}]: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")

        if epoch % config.eval_epoch_freq == 0:
            val_acc = validate(model, val_loader, device)
            print(f"-> Epoch [{epoch}/{config.epochs}]: Val Acc = {val_acc:.4f}")
            writer.add_scalar('Val/Accuracy', val_acc, epoch)

            # Save the best model
            if val_acc > best_acc:
                best_acc = val_acc
                save_checkpoint({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                }, os.path.join(config.save_dir, 'best_model.pth'))
                print(f"   [*] New Best Accuracy: {best_acc:.4f} - Checkpoint Saved!")

        # Save periodic checkpoints
        if epoch % 10 == 0:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
            }, os.path.join(config.save_dir, f'checkpoint_epoch{epoch}.pth'))

    writer.close()
    print(f"Training Finished. Overall Best Val Acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
