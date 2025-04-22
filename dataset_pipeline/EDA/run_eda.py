from dataset_eda import EDAdataset

original_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/original_dataset_CSIRO/'
synthetic_dataset_path = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/synthetic_dataset_pipeline/synthetic_dataset/'

original_eda = EDAdataset(original_dataset_path, eda_output_dir='dataset_eda/original_dataset')
synthetic_eda = EDAdataset(synthetic_dataset_path, eda_output_dir='dataset_eda/synthetic_dataset')
# original_eda.analyse_dataset()
# synthetic_eda.analyse_dataset()
# original_eda.plot_stats()
# synthetic_eda.plot_stats()
# original_eda.show_sample_images()
# synthetic_eda.show_sample_images()