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


class ConditionalUNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.label_emb = nn.Embedding(num_classes, 32)

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1 + 32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # Downsample
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU()
        )

        # Decoder
        self.up1 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Upsample

        self.up2 = nn.Sequential(
            nn.Conv2d(128 + 64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU()
        )
        self.final = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x, y):
        # Embed class label and broadcast to match image shape
        y_emb = self.label_emb(y).unsqueeze(2).unsqueeze(3).expand(-1, -1, x.size(2), x.size(3))
        x = torch.cat([x, y_emb], dim=1)  # Shape: [B, 1+32, H, W]

        # Encoder
        x1 = self.enc1(x)         # [B, 64, H, W]
        x2 = self.enc2(x1)        # [B, 128, H/2, W/2]

        # Bottleneck
        x3 = self.bottleneck(x2)  # [B, 256, H/2, W/2]

        # Decoder
        x4 = self.up1(x3)         # [B, 128, H/2, W/2]
        x4 = self.upconv1(x4)     # [B, 128, H, W]

        # Skip connection
        x = torch.cat([x4, x1], dim=1)  # [B, 192, H, W]
        x = self.up2(x)                # [B, 64, H, W]

        return self.final(x)           # [B, 1, H, W]


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
    def __init__(self, data_path, image_size=400, batch_size=1, noise_steps=200, epochs=2, lr=1e-4):
        self.image_size = image_size
        self.batch_size = batch_size
        self.noise_steps = noise_steps
        self.epochs = epochs
        self.lr = lr

        self.dataset = SARLabelDataset(data_path, image_size)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        self.model = ConditionalUNet(num_classes=2).to(device)
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
                predicted_noise = self.model(noisy_imgs, labels)
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
        out_dir_label = os.path.join(output_dir, str(label))
        os.makedirs(out_dir_label, exist_ok=True)
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((num_samples, 1, self.image_size, self.image_size)).to(device)
            y = torch.tensor([label] * num_samples).to(device)

            for t in reversed(range(self.noise_steps)):
                t_tensor = torch.tensor([t] * num_samples).to(device)
                predicted_noise = self.model(x, y)
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