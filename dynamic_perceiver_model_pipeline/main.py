
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from cnn_pretrain_module import pretrain_cnn
from training_module import train_model
from eval import evaluate_multiple_models, display_results


# ---------------------------
# Dataset Loader
# ---------------------------
class SARImagePatchDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])

        for label in os.listdir(data_dir):
            label_path = os.path.join(data_dir, label)
            if not os.path.isdir(label_path):
                continue
            for file in os.listdir(label_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.data.append(os.path.join(label_path, file))
                    self.labels.append(int(label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ---------------------------
# Run Full Pipeline
# ---------------------------
if __name__ == '__main__':
    # --- Load datasets ---
    print("Loading datasets...")
    original_dataset = SARImagePatchDataset('data/original/')
    synthetic_dataset = SARImagePatchDataset('data/synthetic/')
    combined_dataset = SARImagePatchDataset('data/combined/')

#    --- Pretrain CNN on combined SAR patches ---
#    already done
    print("\nPretraining CNN backbone on combined dataset...")
    pretrain_cnn(combined_dataset, epochs=10, lr=1e-4, save_path="mobilenet_sar.pt")

#this part done 
#     # --- Train CNN + Perceiver ---
    print("\nTraining CNN+Perceiver on original dataset...")
    train_model(original_dataset, epochs=10, lr=1e-4, save_path='perceiver_original.pt')

    print("\nTraining CNN+Perceiver on synthetic dataset...")
    train_model(synthetic_dataset, epochs=10, lr=1e-4, save_path='perceiver_synthetic.pt')

    print("\nTraining CNN+Perceiver on combined dataset...")
    train_model(combined_dataset, epochs=10, lr=1e-4, save_path='perceiver_combined.pt')


model_paths = {
    "Original": "perceiver_original.pt",
    "Synthetic": "perceiver_synthetic.pt",
    "Combined": "perceiver_combined.pt"
}

results = evaluate_multiple_models(test_dataset=combined_dataset, model_paths=model_paths)
display_results(results)