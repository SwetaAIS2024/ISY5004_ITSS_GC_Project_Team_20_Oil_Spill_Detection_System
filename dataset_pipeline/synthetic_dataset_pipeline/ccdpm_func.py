import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision.utils import save_image
from torchsummary import summary
import contextlib

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SARLabelDataset(Dataset):
    def __init__(self, root_dir, image_size=400):
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.image_paths = []
        self.labels = []
        for label in ['0', '1']:
            folder = os.path.join(root_dir, label)
            for img in os.listdir(folder):
                self.image_paths.append(os.path.join(folder, img))
                self.labels.append(int(label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        return self.transform(img), self.labels[idx]

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels)
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + self.shortcut(x))


class ConditionalUNet(nn.Module):
    def __init__(self, num_classes, noise_steps=200):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 128)  # Larger embedding
        self.time_emb = nn.Embedding(noise_steps, 128)  # Time embedding

        # Encoder
        self.enc1 = ResBlock(1 + 128, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)

        # Downsampling
        self.pool1 = nn.MaxPool2d(2)  # H/2
        self.pool2 = nn.MaxPool2d(2)  # H/4

        # Bottleneck
        self.bottleneck = ResBlock(256, 512)

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec1 = ResBlock(512, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ResBlock(256, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec3 = ResBlock(128, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, y, t):
        #y_emb = self.label_emb(y).unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        y_emb = self.label_emb(y)
        t_emb = self.time_emb(t)
        cond = y_emb + t_emb
        cond = cond.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, cond], dim=1)  # Shape: [B, 1+128+128, H, W]
        #x = torch.cat([x, y_emb], dim=1)  # [B, 1+128, H, W]

        # Encoder
        x1 = self.enc1(x)                   # [B, 64, H, W]
        x2 = self.enc2(self.pool1(x1))     # [B, 128, H/2, W/2]
        x3 = self.enc3(self.pool2(x2))     # [B, 256, H/4, W/4]

        # Bottleneck
        x4 = self.bottleneck(x3)           # [B, 512, H/4, W/4]

        # Decoder
        x = self.up1(x4)                   # [B, 256, H/2, W/2]
        if x.shape[-2:] != x3.shape[-2:]:  # Ensure match
            x = torch.nn.functional.interpolate(x, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec1(torch.cat([x, x3], dim=1))  # [B, 256, H/2, W/2]

        x = self.up2(x)                    # [B, 128, H, W]
        if x.shape[-2:] != x2.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec2(torch.cat([x, x2], dim=1))  # [B, 128, H, W]

        x = self.up3(x)                    # [B, 64, 2H, 2W]
        if x.shape[-2:] != x1.shape[-2:]:
            x = torch.nn.functional.interpolate(x, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x = self.dec3(torch.cat([x, x1], dim=1))  # [B, 64, 2H, 2W]

        return self.final(x)               # [B, 1, 2H, 2W]

class DiffusionScheduler:
    def __init__(self, noise_steps=200):
        self.noise_steps = noise_steps
        self.betas = torch.linspace(1e-4, 0.02, noise_steps)
        self.alphas = 1. - self.betas
        self.alpha_hat = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, noise, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None].to(device)
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None].to(device)
        return sqrt_alpha_hat * x0 + sqrt_one_minus_alpha_hat * noise


class cDDPMTrainer:
    def __init__(self, data_path=None, image_size=400, batch_size=1, noise_steps=200, epochs=2, lr=1e-4, model=None, device=None):
        # self.image_size = image_size
        # self.batch_size = batch_size
        # self.noise_steps = noise_steps
        # self.epochs = epochs
        # self.lr = lr

        # self.dataset = SARLabelDataset(data_path, image_size)
        # self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        # self.model = ConditionalUNet(num_classes=2, noise_steps=self.noise_steps).to(device)
        # self.scheduler = DiffusionScheduler(noise_steps)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        # self.loss_fn = nn.MSELoss()

        self.image_size = image_size
        self.batch_size = batch_size
        self.noise_steps = noise_steps
        self.epochs = epochs
        self.lr = lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load dataset only if path is provided
        if data_path:
            self.dataset = SARLabelDataset(data_path, image_size)
            self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)

        # Model + scheduler
        self.model = model.to(self.device) if model else ConditionalUNet(num_classes=2, noise_steps=noise_steps).to(self.device)
        self.scheduler = DiffusionScheduler(noise_steps)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for imgs, labels in tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                imgs, labels = imgs.to(device), torch.tensor(labels).to(device)
                noise = torch.randn_like(imgs)
                t = torch.randint(0, self.noise_steps, (imgs.size(0),), device=device).long()
                noisy_imgs = self.scheduler.add_noise(imgs, noise, t)
                predicted_noise = self.model(noisy_imgs, labels, t)
                loss = self.loss_fn(predicted_noise, noise)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1} Loss: {total_loss / len(self.dataloader):.4f}")

    def save_model(self, path="cddpm_sar_model.pt"):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def model_summary(self):    
        summary(self.model, input_size=(1, 400, 400))  # (channels, height, width)
        # Also save to file
        with open("model_architecture.txt", "w") as f:
            with contextlib.redirect_stdout(f):
                summary(self.model, input_size=(1, self.image_size, self.image_size))
        # Save raw structure
        with open("model_structure_layers.txt", "w") as f:
            print(self.model, file=f)

    def generate_samples(self, num_samples=5, label=1, output_dir="synthetic_dataset"):
        print(f"Generating {num_samples} samples for label {label}...")
        out_dir_label = os.path.join(output_dir, str(label))
        os.makedirs(out_dir_label, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((num_samples, 1, self.image_size, self.image_size)).to(device)
            y = torch.tensor([label] * num_samples).to(device)

            for t in reversed(range(self.noise_steps)):
                t_tensor = torch.tensor([t] * num_samples).to(device)
                predicted_noise = self.model(x, y, t_tensor)
                alpha = self.scheduler.alphas[t]
                alpha_hat = self.scheduler.alpha_hat[t]
                beta = self.scheduler.betas[t]

                if t > 0:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)

                x = (1 / torch.sqrt(alpha)) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * predicted_noise) + torch.sqrt(beta) * noise

            for i in range(num_samples):
                save_image(x[i], os.path.join(out_dir_label, f"sample_class{label}_{i}.png"))

        print(f"Generated {num_samples} synthetic images for class {label} in {out_dir_label}")