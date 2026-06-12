import random
import numpy as np
import torch
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic results for convolutional operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_dar_weights(class_freqs, total_samples, num_classes, eps=1e-6, w_max=10.0):
    """
    Calculate class-balanced weights.
    """
    weights = []
    for n_c in class_freqs:
        w = total_samples / (num_classes * (n_c + eps))
        w = min(w, w_max)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(state, filename):
    """
    Save checkpoint.
    """
    model_state_dict = state.get('model_state_dict', {})

    lean_state_dict = {
        k: v for k, v in model_state_dict.items()
        if not k.startswith('clip.text_model.')
    }

    # Update model weights in the state dictionary
    state['model_state_dict'] = lean_state_dict

    torch.save(state, filename)
    print(f"Checkpoint (Lean Version) saved to {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """
    Load checkpoint.
    """
    checkpoint = torch.load(filename, map_location=device)

    # Since the text_model was excluded during saving, use strict=False during loading.
    # This ensures the frozen CLIP text_model weights are not overwritten, 
    # allowing the visual backbone to load normally.
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    best_acc = checkpoint.get('best_acc', 0)
    return epoch, best_acc
