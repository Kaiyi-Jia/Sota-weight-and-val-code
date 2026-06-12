import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from modules.EHSE import EHSE
from modules.EKFF import EKFF
from modules.EHSA import EHSA


class BoostingFER(nn.Module):
    def __init__(self, config, class_freqs, class_names):
        super().__init__()
        self.config = config
        self.num_classes = config.class_num

        print(f"Loading native OpenAI CLIP model: {config.clip_model_name}...")
        self.clip_model, _ = clip.load(config.clip_model_name, device="cpu", jit=False)

        # Freeze the text encoder, keep the visual encoder trainable
        # Completely freeze the text encoder
        for param in self.clip_model.transformer.parameters():
            param.requires_grad = False
        self.clip_model.token_embedding.requires_grad = False
        self.clip_model.positional_embedding.requires_grad = False
        self.clip_model.ln_final.requires_grad = False
        self.clip_model.text_projection.requires_grad = False

        for param in self.clip_model.visual.parameters():
            param.requires_grad = True

# for param in self.clip_model.visual.parameters():  #freeze if need
#             param.requires_grad = False
#         for i in range(12, 24):
#             for param in self.clip_model.visual.transformer.resblocks[i].parameters():
#                 param.requires_grad = True

        self.clip_model.visual.ln_post.requires_grad = True
        self.clip_model.visual.proj.requires_grad = True

        # Ensure the model is trained in float32
        self.clip_model.float()

        # Initialize core modules
        self.ehse = EHSE(num_classes=self.num_classes, class_names=class_names)
        self.ekff = EKFF(config, feature_dim=768)
        self.ehsa = EHSA(config, class_freqs, class_names)

        self.cross_attn = nn.MultiheadAttention(embed_dim=768, num_heads=8, batch_first=True)

        self.register_buffer('t_global', torch.zeros(self.num_classes, 768))
        self.register_buffer('t_local', torch.zeros(self.num_classes, 768))

    def compute_semantic_anchors(self, device):
        """Call EHSE to obtain semantic anchors"""
        t_global, t_local = self.ehse.get_semantic_embeddings(self.clip_model, device)
        self.t_global.data = t_global
        self.t_local.data = t_local

    def encode_image_patches(self, images):

        visual = self.clip_model.visual
        x = images.type(visual.conv1.weight.dtype)

        x = visual.conv1(x)  # [B, 1024, 16, 16]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 1024, 256]
        x = x.permute(0, 2, 1)  # [B, 256, 1024]

        cls_emb = visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype,
                                                                   device=x.device)
        x = torch.cat([cls_emb, x], dim=1)  # [B, 257, 1024]

        x = x + visual.positional_embedding.to(x.dtype)
        x = visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # [257, B, 1024]
        x = visual.transformer(x)
        x = x.permute(1, 0, 2)  # [B, 257, 1024]

        x_proj = x @ visual.proj

        patch_tokens = x_proj[:, 1:, :]  # [B, 256, 768]
        return patch_tokens

    def forward(self, images, return_loss=True, targets=None):
        B = images.shape[0]

        # Extract visual patch features
        patch_tokens = self.encode_image_patches(images)
        L = patch_tokens.shape[1]
        H = W = int(L ** 0.5)  # 16x16
        patch_grid = patch_tokens.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, 768, 16, 16]

        # Generate raw attention map using MHCA
        mean_text_query = self.t_global.mean(dim=0).unsqueeze(0).unsqueeze(0).expand(B, -1, -1)
        attn_out, attn_weights = self.cross_attn(query=mean_text_query, key=patch_tokens, value=patch_tokens)
        raw_attn_map = attn_weights.view(B, H, W)

        # EKFF
        V_final, mask, attn_prob = self.ekff(patch_grid, raw_attn_map)

        # Dual-granularity feature pooling
        visual_global = F.adaptive_avg_pool2d(patch_grid, (1, 1)).view(B, -1)
        visual_local = F.adaptive_avg_pool2d(V_final, (1, 1)).view(B, -1)

        # Compute loss or output predictions
        if return_loss and targets is not None:
            total_loss, loss_dict, fine_logits = self.ehsa(
                visual_global, visual_local, self.t_global, self.t_local, targets
            )
            return fine_logits, total_loss, loss_dict
        else:
            fine_logits = self.ehsa.fine_classifier(visual_global)
            return fine_logits
