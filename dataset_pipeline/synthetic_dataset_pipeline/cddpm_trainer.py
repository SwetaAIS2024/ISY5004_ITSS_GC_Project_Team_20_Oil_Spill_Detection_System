from ccdpm_func import cDDPMTrainer

original_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/original_dataset_CSIRO/'

import os
import shutil
import random

# Set paths
original_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/original_dataset_CSIRO/'
subset_path = 'subset_dataset'

# Seed for reproducibility
random.seed(42)

# Process each label
for label in ['0', '1']:
    src_dir = os.path.join(original_dataset_path, label)
    dst_dir = os.path.join(subset_path, label)
    os.makedirs(dst_dir, exist_ok=True)

    # List image files
    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif'))]
    print(f"Found {len(image_files)} images in label {label}")

    # Select up to 500 randomly
    selected_files = random.sample(image_files, min(500, len(image_files)))

    # Copy selected files
    for file_name in selected_files:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.copy(src_path, dst_path)

    print(f"Copied {len(selected_files)} images to {dst_dir}")

print("Done creating subset dataset.")


#trainer = cDDPMTrainer(original_dataset_path)
trainer = cDDPMTrainer(subset_path)
trainer.train()
trainer.save_model()
trainer.model_summary()
