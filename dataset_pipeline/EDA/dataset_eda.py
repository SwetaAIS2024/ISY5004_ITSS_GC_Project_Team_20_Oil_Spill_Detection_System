import os
import sys
import numpy as np
from PIL import Image, UnidentifiedImageError
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from datetime import datetime

class Logger:
    def __init__(self, log_path):
        self.log_file = open(log_path, "w", encoding="utf-8")
        self.terminal = sys.stdout
    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
    def flush(self):
        pass

class EDAdataset:
    
    def __init__(self, dataset_path, eda_output_dir='dataset_eda'):
        self.dataset_path = dataset_path
        self.eda_output_dir = eda_output_dir
        self.dataset_classes = ['0','1']
        os.makedirs(self.eda_output_dir, exist_ok=True)
        self.eda_stats = { 
            stats_per_cls: {
                'dataset_count':0,
                'dataset_shape': set(),
                'dataset_mean':[],
                'dataset_stds':[],
                'dataset_min':[],
                'dataset_max':[],
                'dataset_corrupted':0  
         } for stats_per_cls in self.dataset_classes}
        
        #logging
        log_file = os.path.join(self.eda_output_dir, f"dataset_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        sys.stdout = Logger(log_file)
        print("Logging to file: ", log_file)

    def analyse_dataset(self):
        print("Analyzing dataset at : ", self.dataset_path)
        for cls in self.dataset_classes:
            class_path = os.path.join(self.dataset_path, cls)
            if not os.path.exists(class_path):
                print(f"Warning: Folder '{cls}' not found in dataset path.")
                continue

            for img_name in tqdm(os.listdir(class_path), desc=f"Analyzing class {cls}"):
                try:
                    img_path = os.path.join(class_path, img_name)
                    img = Image.open(img_path).convert("L")
                    img = np.array(img).astype(np.float32) / 255.0
                    self.eda_stats[cls]['dataset_count'] += 1
                    self.eda_stats[cls]['dataset_shape'].add(img.shape)
                    self.eda_stats[cls]['dataset_mean'].append(np.mean(img))
                    self.eda_stats[cls]['dataset_stds'].append(np.std(img))
                    self.eda_stats[cls]['dataset_min'].append(np.min(img))
                    self.eda_stats[cls]['dataset_max'].append(np.max(img))
                except UnidentifiedImageError:
                    self.eda_stats[cls]['dataset_corrupted'] += 1
                    print(f"Corrupted image: {img_path}")
        print(f"\nAnalysis complete. Results stored in `self.eda_stats`.")
    
    def plot_stats(self):
        print("Plotting stats for dataset at : ", self.dataset_path)
        for cls in self.dataset_classes:
            stats = self.eda_stats[cls]
            print(f"\nClass: {cls}")
            print(f"Images: {stats['dataset_count']}")
            print(f"Corrupted: {stats['dataset_corrupted']}")
            print(f"Shapes: {stats['dataset_shape']}")
            print(f"Mean: {np.mean(stats['dataset_mean']):.4f}")
            print(f"Std: {np.mean(stats['dataset_stds']):.4f}")
            print(f"Min: {np.mean(stats['dataset_min']):.4f}")
            print(f"Max: {np.mean(stats['dataset_max']):.4f}")

            for metric in ['dataset_mean', 'dataset_stds', 'dataset_min', 'dataset_max']:
                plt.figure()
                sns.histplot(stats[metric], kde=True)
                plt.title(f"{metric.upper()} distribution - Class {cls}")
                plt.xlabel(metric)
                plt.ylabel("Frequency")
                plot_path = os.path.join(self.eda_output_dir, f"{metric}_distribution_class_{cls}.png")
                plt.savefig(plot_path)
                plt.close

    def show_sample_images(self, n=5):
        print("Showing sample images for dataset at :", self.dataset_path)
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
        for i, cls in enumerate(self.dataset_classes):
            class_path = os.path.join(self.dataset_path, cls)
            images = os.listdir(class_path)[:n]
            for j, img_name in enumerate(images):
                img_path = os.path.join(class_path, img_name)
                img = Image.open(img_path).convert("L")
                axes[i, j].imshow(img, cmap='gray')
                axes[i, j].axis('off')
                axes[i, j].set_title(f"Class {cls}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.eda_output_dir, "sample_images_grid.png"))
        plt.close()
        print(f"Saved sample grid to: {self.eda_output_dir}/sample_images_grid.png")