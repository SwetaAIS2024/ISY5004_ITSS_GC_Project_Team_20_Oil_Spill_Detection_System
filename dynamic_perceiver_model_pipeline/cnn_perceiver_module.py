# # import torch
# # import torch.nn as nn
# # from einops import rearrange

# # class PerceiverClassifier(nn.Module):
# #     def __init__(self, input_dim=3*64*64, latent_dim=512, num_latents=8, output_dim=2):
# #         super().__init__()
# #         self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
# #         self.input_proj = nn.Linear(input_dim, latent_dim)
# #         self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads=4)
# #         self.norm = nn.LayerNorm(latent_dim)
# #         self.mlp = nn.Sequential(
# #             nn.Linear(latent_dim, latent_dim),
# #             nn.ReLU(),
# #             nn.Linear(latent_dim, output_dim)
# #         )

# #     def forward(self, x):  # x: (B, 3, 64, 64)
# #         b = x.size(0)
# #         x = rearrange(x, 'b c h w -> b (h w c)')  # Flatten spatial dimensions
# #         x = self.input_proj(x)  # (B, latent_dim)

# #         latents = self.latents.unsqueeze(1).repeat(1, b, 1)  # (num_latents, B, latent_dim)
# #         x = x.unsqueeze(0)  # (1, B, latent_dim)

# #         attended, _ = self.cross_attention(latents, x, x)
# #         attended = self.norm(attended)

# #         pooled = attended.mean(dim=0)  # (B, latent_dim)
# #         return self.mlp(pooled)        # (B, output_dim)


# import torch
# import torch.nn as nn
# from torchvision import models
# from einops import rearrange
# from collections import OrderedDict

# class CNNPerceiverClassifier(nn.Module):
#     def __init__(self, cnn_weights='mobilenet_sar.pt', latent_dim=256, num_latents=8, output_dim=2, freeze_cnn=True):
#         super().__init__()

#         # # Load MobileNetV3-Small pretrained on SAR patches
#         # base = models.mobilenet_v3_small(pretrained=False)
#         # base.load_state_dict(torch.load(cnn_weights, map_location='cpu'))
#         # self.cnn_features = nn.Sequential(*list(base.features.children()))
#         # cnn_output_dim = 576  # MobileNetV3-small output for 64x64 input

#         base = models.mobilenet_v3_small(pretrained=False)
#         full_weights = torch.load(cnn_weights, map_location='cpu')
#         filtered_weights = OrderedDict()

#         for k, v in full_weights.items():
#             if k.startswith("base.features."):
#                 new_k = k.replace("base.features.", "")
#                 filtered_weights[new_k] = v

#         self.cnn_features = nn.Sequential(*list(base.features.children()))
#         self.cnn_features.load_state_dict(filtered_weights)
#         cnn_output_dim = 576  # MobileNetV3-small output for 64x64 input

#         if freeze_cnn:
#             for param in self.cnn_features.parameters():
#                 param.requires_grad = False

#         # Perceiver module
#         self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))
#         self.input_proj = nn.Linear(cnn_output_dim, latent_dim)
#         self.cross_attention = nn.MultiheadAttention(latent_dim, num_heads=4)
#         self.norm = nn.LayerNorm(latent_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(latent_dim, latent_dim),
#             nn.ReLU(),
#             nn.Linear(latent_dim, output_dim)
#         )

#     def forward(self, x):
#         feats = self.cnn_features(x)  # Shape: (B, C, H, W)
#         b, c, h, w = feats.shape
#         x = rearrange(feats, 'b c h w -> b (h w) c')  # Shape: (B, HW, C)
#         x = self.input_proj(x)                        # Shape: (B, HW, latent_dim)
#         x = x.permute(1, 0, 2)                        # Shape: (HW, B, latent_dim)

#         latents = self.latents.unsqueeze(1).repeat(1, b, 1)  # (num_latents, B, latent_dim)
#         attended, _ = self.cross_attention(latents, x, x)
#         attended = self.norm(attended)
#         pooled = attended.mean(dim=0)  # (B, latent_dim)
#         return self.mlp(pooled)        # (B, output_dim)



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