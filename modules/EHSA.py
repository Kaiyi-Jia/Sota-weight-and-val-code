import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_dar_weights


class EHSA(nn.Module):
    """
    Expression High-Level Semantic Aligning (EHSA) Module
    """

    def __init__(self, config, class_freqs, class_names):
        super().__init__()
        self.config = config
        self.class_names = class_names
        self.num_classes = len(class_names)

        # DAR
        total_samples = sum(class_freqs)
        self.register_buffer('class_weights', compute_dar_weights(
            class_freqs, total_samples, self.num_classes, config.eps_dar, config.w_max_dar
        ))

        # HSG
        # (Neutral: 0, Positive: 1, Negative: 2)
        self.coarse_map = self._build_coarse_polarity_map()
        self.num_coarse = 3  # Strictly corresponds to 3 emotion polarities

        self.fine_classifier = nn.Linear(768, self.num_classes)
        self.coarse_classifier = nn.Linear(768, self.num_coarse)
        self.alpha = config.alpha_hsg

        # DGFA
        # tau initialized to 0.07
        self.tau = nn.Parameter(torch.tensor(config.tau_infonce))

    def _build_coarse_polarity_map(self):
        """
        Based on human cognitive priors, map the 8 fine-grained expressions to 3 coarse-grained emotion polarities.
        - Positive (1): Happy
        - Neutral (0): Neutral, Surprise(Surprised)
        - Negative (2): Anger, Disgust, Fear, Sadness, Contempt
        """
        mapping = {}
        for idx, name in enumerate(self.class_names):
            if name in ["Neutral", "Surprise", "Surprised"]:
                mapping[idx] = 0  # Neutral
            elif name in ["Happy", "Happiness"]:
                mapping[idx] = 1  # Positive
            else:
                mapping[idx] = 2  # Negative

        coarse_labels = [mapping[i] for i in range(self.num_classes)]
        return torch.tensor(coarse_labels, dtype=torch.long)

    def compute_hierarchical_loss(self, visual_global, targets):

        device = visual_global.device

        # 1. DAR
        fine_logits = self.fine_classifier(visual_global)  # [B, 768] -> [B, 8]
        wce_loss = F.cross_entropy(fine_logits, targets, weight=self.class_weights.to(device))

        # 2. Coarse-grained polarity loss
        coarse_targets = self.coarse_map[targets].to(device)
        coarse_logits = self.coarse_classifier(visual_global)  # [B, 768] -> [B, 3]
        coarse_loss = F.cross_entropy(coarse_logits, coarse_targets)

        # 3. Fusion
        hce_loss = self.alpha * coarse_loss + (1 - self.alpha) * wce_loss
        return hce_loss, wce_loss, coarse_loss, fine_logits

    def compute_dual_itc_loss(self, visual_global, visual_local, t_global, t_local, targets):

        B = visual_global.shape[0]
        device = visual_global.device

        v_glob = F.normalize(visual_global, dim=-1)
        v_loc = F.normalize(visual_local, dim=-1)
        t_glob = F.normalize(t_global[targets], dim=-1)
        t_loc = F.normalize(t_local[targets], dim=-1)

        labels = torch.arange(B, device=device)

        # Global ITC (I2T + T2I)
        logits_glob = (v_glob @ t_glob.t()) / self.tau
        loss_glob = (F.cross_entropy(logits_glob, labels) + F.cross_entropy(logits_glob.t(), labels)) / 2

        # Local ITC (I2T + T2I)
        logits_loc = (v_loc @ t_loc.t()) / self.tau
        loss_loc = (F.cross_entropy(logits_loc, labels) + F.cross_entropy(logits_loc.t(), labels)) / 2

        # Dual-granularity weighted fusion
        loss_dual = self.config.gamma_dgfa * loss_glob + (1 - self.config.gamma_dgfa) * loss_loc
        return loss_dual

    def forward(self, visual_global, visual_local, t_global, t_local, targets):

        hce_loss, wce_loss, coarse_loss, fine_logits = self.compute_hierarchical_loss(visual_global, targets)
        dual_loss = self.compute_dual_itc_loss(visual_global, visual_local, t_global, t_local, targets)

        # lambda1 and lambda2 default to 1.0
        total_loss = self.config.lambda1 * hce_loss + self.config.lambda2 * dual_loss

        loss_dict = {
            'hce': hce_loss,
            'wce': wce_loss,
            'coarse': coarse_loss,
            'dual': dual_loss
        }

        return total_loss, loss_dict, fine_logits
