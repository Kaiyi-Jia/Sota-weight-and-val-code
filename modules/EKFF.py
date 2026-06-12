import torch
import torch.nn as nn
import torch.nn.functional as F


class STEHardMask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, F_afs, mask, attn_prob, beta, F_img):
        ctx.save_for_backward(mask, attn_prob, beta, F_img)
        output = F_afs * mask
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask, attn_prob, beta, F_img = ctx.saved_tensors
        B, C, H, W = grad_output.shape

        beta_exp = beta.view(1, C, 1, 1)

        # Sum along the channel dimension (dim=1)
        grad_P_att = (grad_output * F_img * beta_exp).sum(dim=1)  # [B, H, W]

        # Expand dimensions for broadcasting
        attn_prob_exp = attn_prob.unsqueeze(1)  # [B, 1, H, W]
        grad_beta = (grad_output * F_img * attn_prob_exp * mask).sum(dim=(0, 2, 3))  # [C]

        # 3. Gradient with respect to F_afs
        grad_F_afs = grad_output * mask

        # F_afs, mask (non-differentiable), attn_prob, beta, F_img
        return grad_F_afs, None, grad_P_att, grad_beta, None


class EKFF(nn.Module):
    """
    Expression Key-core Features Focusing (EKFF) Module
    """

    def __init__(self, config, feature_dim=768):
        super().__init__()
        self.retention_ratio = getattr(config, 'retention_ratio', 0.5)  # Default is 0.5 in the paper

        # ANO
        self.W_n = nn.Parameter(torch.tensor(1.0))
        self.b_n = nn.Parameter(torch.tensor(0.0))

        # AFS
        self.beta = nn.Parameter(torch.ones(feature_dim))

    def forward(self, visual_features, raw_attn_map):
        B, C, H, W = visual_features.shape

        # ANO
        attn_ano = F.relu(self.W_n * raw_attn_map + self.b_n)  # [B, H, W]

        # AFS
        attn_ano_flat = attn_ano.view(B, -1)
        attn_prob_flat = F.softmax(attn_ano_flat, dim=-1)
        attn_prob = attn_prob_flat.view(B, H, W)

        beta_exp = self.beta.view(1, C, 1, 1)
        F_afs = visual_features * attn_prob.unsqueeze(1) * beta_exp  # [B, C, H, W]

        # KFS
        k = max(1, int(H * W * self.retention_ratio))

        # Use kthvalue to find the threshold
        threshold = torch.kthvalue(attn_prob_flat, H * W - k + 1, dim=-1, keepdim=True).values
        mask_flat = (attn_prob_flat >= threshold).float()
        mask = mask_flat.view(B, 1, H, W)

        # STE
        V_final = STEHardMask.apply(F_afs, mask, attn_prob, self.beta, visual_features)

        return V_final, mask, attn_prob
