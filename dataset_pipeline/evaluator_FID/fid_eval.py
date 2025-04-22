import os
import torch
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image


class SAREvaluatorFIDOnly:
    def __init__(self, real_dir, generated_dir, image_size=(128, 128), max_images=200):
        self.real_dir = real_dir
        self.generated_dir = generated_dir
        self.image_size = image_size
        self.max_images = max_images
        self.fid_metric = FrechetInceptionDistance(feature=64, normalize=True)
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
        ])

    def load_and_update(self, path, real_label):
        count = 0
        for img_name in sorted(os.listdir(path)):
            if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                continue
            img_path = os.path.join(path, img_name)
            img = Image.open(img_path).convert("L")
            img_tensor = self.transform(img).repeat(3, 1, 1).unsqueeze(0)  # replicate grayscale to RGB-like
            self.fid_metric.update(img_tensor, real=real_label)
            count += 1
            if count >= self.max_images:
                break
        print(f"{'Real' if real_label else 'Generated'}: Loaded {count} images.")

    def compute_fid(self):
        print("Computing FID (efficient mode)...")
        with torch.no_grad():
            self.load_and_update(self.real_dir, real_label=True)
            self.load_and_update(self.generated_dir, real_label=False)
            return self.fid_metric.compute().item()

    def evaluate(self):
        fid = self.compute_fid()
        print(f"FID Score: {fid:.4f}")
        return fid