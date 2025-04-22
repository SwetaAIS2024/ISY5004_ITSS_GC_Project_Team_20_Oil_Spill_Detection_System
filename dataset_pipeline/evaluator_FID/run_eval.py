from fid_eval import SAREvaluatorFIDOnly

original_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/original_dataset_CSIRO/'
synthetic_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/synthetic_dataset_pipeline/synthetic_dataset/'


# Evaluate FID for class 0 (no-oil)
evaluator0 = SAREvaluatorFIDOnly(
    real_dir=original_dataset_path + "/0",
    generated_dir=synthetic_dataset_path + "/0",
    image_size=(128, 128),  # Adjust as needed
    max_images=100  # Adjust as needed
)
fid_0 = evaluator0.evaluate()

# Evaluate FID for class 1 (oil spill)
evaluator1 = SAREvaluatorFIDOnly(
    real_dir=original_dataset_path + "/1",
    generated_dir=synthetic_dataset_path + "/1",
    image_size=(128, 128),  # Adjust as needed
    max_images=100  # Adjust as needed
)
fid_1 = evaluator1.evaluate()

print(f"FID for class 0 (no-oil): {fid_0:.4f}")
print(f"FID for class 1 (oil): {fid_1:.4f}")