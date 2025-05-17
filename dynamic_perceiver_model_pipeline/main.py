import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from perceiver_module import PerceiverClassifier
from backbone_module_yolo import YOLOBackbone
from training_module import train_model
from eval import evaluate_model

# ---------------------------
# Dummy Dataset Loader Class
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
            for file in os.listdir(label_path):
                self.data.append(os.path.join(label_path, file))
                self.labels.append(int(label))  # assumes folder name = class index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label

# ---------------------------
# Run Pipeline
# ---------------------------
if __name__ == '__main__':
    # --- Step 1: Detection and Cropping ---
#     image_path = 'sample_sar_image.jpg'
#     yolo = YOLOBackbone(model_path='yolov8n.pt')
#     crops, boxes = yolo.detect_and_crop(image_path)
#     print(f"Detected {len(crops)} regions from {image_path}")

    # --- Step 2: Load Datasets ---
    original_dataset = SARImagePatchDataset('/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dynamic_perceiver_model_pipeline/data/original/')
    synthetic_dataset = SARImagePatchDataset('/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dynamic_perceiver_model_pipeline/data/synthetic/')
    combined_dataset = SARImagePatchDataset('/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dynamic_perceiver_model_pipeline/data/combined/')

    # --- Step 3: Train Model ---
    print("Training Perceiver on original dataset...")
    train_model(original_dataset, epochs=100, lr=1e-4, save_path='perceiver_original.pt')
    print("Training Perceiver on synthetic dataset...")
    train_model(synthetic_dataset, epochs=100, lr=1e-4, save_path='perceiver_synthetic.pt')
    print("Training Perceiver on combined dataset...")
    train_model(combined_dataset, epochs=100, lr=1e-4, save_path='perceiver_combined.pt')

    # --- Step 4: Evaluate Model ---
    print("Evaluating Perceiver on original dataset...")
    evaluate_model(original_dataset, model_path='perceiver_original.pt', dataset_type='original')
    print("Evaluating Perceiver on synthetic dataset...")
    evaluate_model(synthetic_dataset, model_path='perceiver_synthetic.pt', dataset_type='synthetic')
    print("Evaluating Perceiver on combined dataset...")
    evaluate_model(combined_dataset, model_path='perceiver_combined.pt', dataset_type='combined')

#     # --- Step 5: Inference on New Crops ---
#     model = PerceiverClassifier()
#     model.load_state_dict(torch.load('perceiver.pt'))
#     model.eval()

#     for idx, crop in enumerate(crops):
#         crop = crop.unsqueeze(0)  # Add batch dim
#         with torch.no_grad():
#             pred = model(crop)
#             class_id = torch.argmax(pred).item()
#             print(f"Crop {idx+1}: Predicted class {class_id}")