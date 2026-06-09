import random
import numpy as np
import torch
import os


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保卷积操作的结果一致性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_dar_weights(class_freqs, total_samples, num_classes, eps=1e-6, w_max=10.0):
    """
    计算类别平衡权重。
    """
    weights = []
    for n_c in class_freqs:
        w = total_samples / (num_classes * (n_c + eps))
        w = min(w, w_max)
        weights.append(w)
    return torch.tensor(weights, dtype=torch.float32)


def save_checkpoint(state, filename):
    """
    保存 Checkpoint
    """
    model_state_dict = state.get('model_state_dict', {})

    lean_state_dict = {
        k: v for k, v in model_state_dict.items()
        if not k.startswith('clip.text_model.')
    }

    # 更新 state 里的模型权重
    state['model_state_dict'] = lean_state_dict

    torch.save(state, filename)
    print(f"Checkpoint (Lean Version) saved to {filename}")


def load_checkpoint(model, optimizer, filename, device):
    """
    加载Checkpoint
    """
    checkpoint = torch.load(filename, map_location=device)

    # 因为我们保存时剔除了 text_model，加载时使用 strict=False，
    # 这样模型中自带的冻结 CLIP text_model 权重不会被覆盖，视觉部分正常加载。
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    best_acc = checkpoint.get('best_acc', 0)
    return epoch, best_acc