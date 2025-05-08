# Generate synthetic images (e.g., 10 oil spill, 10 no-oil) using the trained model

import torch
from ccdpm_func import cDDPMTrainer  # Assuming CCDPMTrainer is defined in ccdpm_trainer.py

# Path to the synthetic dataset and the saved model
synthetic_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/synthetic_dataset_pipeline/synthetic_dataset/'
model_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/synthetic_dataset_pipeline/cddpm_sar_model.pt'

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)

# Initialize the trainer with the loaded model
generator_sar_img = cDDPMTrainer(model=model, device=device)

# Generate synthetic samples
generator_sar_img.generate_samples(num_samples=100, label=1, output_dir=synthetic_dataset_path)  # For oil spill
generator_sar_img.generate_samples(num_samples=100, label=0, output_dir=synthetic_dataset_path)  # For no-oil