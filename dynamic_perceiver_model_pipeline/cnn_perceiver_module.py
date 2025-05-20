
import torch
import torch.nn as nn
from torchvision import models
from einops import rearrange

class CNNPerceiverClassifier(nn.Module):
    def __init__(self, cnn_weights='mobilenet_sar.pt', latent_dim=256, num_latents=8, output_dim=2, freeze_cnn=True):
        super().__init__()

        # Load pretrained MobileNetV3 features
        base = models.mobilenet_v3_small(pretrained=False)
        self.cnn_features = base.features
        self.cnn_features.load_state_dict(torch.load(cnn_weights, map_location='cpu'))

        if freeze_cnn:
            for param in self.cnn_features.parameters():
                param.requires_grad = False

        cnn_output_dim = 576  # MobileNetV3-small output for 64x64 input

        # Perceiver modules
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_proj = nn.Linear(cnn_output_dim, latent_dim)
        self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads=4)
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim)
        )

    def forward(self, x):
        feats = self.cnn_features(x)  # Shape: (B, C, H, W)
        b, c, h, w = feats.shape
        x = rearrange(feats, 'b c h w -> b (h w) c')  # → (B, HW, C)
        x = self.input_proj(x)                        # → (B, HW, latent_dim)
        x = x.permute(1, 0, 2)                        # → (HW, B, latent_dim)

        latents = self.latents.unsqueeze(1).repeat(1, b, 1)  # (num_latents, B, latent_dim)
        attended, _ = self.cross_attention(latents, x, x)
        attended = self.norm(attended)
        pooled = attended.mean(dim=0)  # (B, latent_dim)
        return self.mlp(pooled)        # (B, output_dim)