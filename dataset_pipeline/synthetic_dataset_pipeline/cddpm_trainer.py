from ccdpm_func import cDDPMTrainer

original_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/original_dataset_CSIRO/'

trainer = cDDPMTrainer(original_dataset_path)
trainer.train()
trainer.save_model()
trainer.model_summary()
