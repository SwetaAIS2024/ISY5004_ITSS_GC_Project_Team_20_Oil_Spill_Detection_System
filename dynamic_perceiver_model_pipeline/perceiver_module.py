import torch
import torch.nn as nn
from einops import rearrange

class PerceiverClassifier(nn.Module):
    def __init__(self, input_dim=3*64*64, latent_dim=512, num_latents=8, output_dim=2):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
        self.input_proj = nn.Linear(input_dim, latent_dim)
        self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads=4)
        self.norm = nn.LayerNorm(latent_dim)
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, output_dim)
        )

    def forward(self, x):  # x: (B, 3, 64, 64)
        b = x.size(0)
        x = rearrange(x, 'b c h w -> b (h w c)')  # Flatten spatial dimensions
        x = self.input_proj(x)  # (B, latent_dim)

        latents = self.latents.unsqueeze(1).repeat(1, b, 1)  # (num_latents, B, latent_dim)
        x = x.unsqueeze(0)  # (1, B, latent_dim)

        attended, _ = self.cross_attention(latents, x, x)
        attended = self.norm(attended)

        pooled = attended.mean(dim=0)  # (B, latent_dim)
        return self.mlp(pooled)        # (B, output_dim)