
import torch
from ccdpm_func import cDDPMTrainer, ConditionalUNet  # Assuming both are defined in ccdpm_func.py

# Paths
synthetic_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/synthetic_dataset_pipeline/synthetic_dataset/'
model_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/synthetic_dataset_pipeline/cddpm_sar_model.pt'

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize model architecture and load weights
model = ConditionalUNet(num_classes=2, noise_steps=200)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# Initialize the trainer for inference (no dataset path needed)
generator_sar_img = cDDPMTrainer(model=model, device=device)

generator_sar_img.noise_steps = 50  # Reduce from 200
generator_sar_img.image_size = 128  # Reduce from 400

# Generate synthetic samples
generator_sar_img.generate_samples(num_samples=200, label=1, output_dir=synthetic_dataset_path)  # Oil spill
generator_sar_img.generate_samples(num_samples=200, label=0, output_dir=synthetic_dataset_path)  # No-oil