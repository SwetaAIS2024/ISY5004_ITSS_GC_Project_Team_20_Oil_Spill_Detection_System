import os
import numpy as np 
from PIL import Image, UnidentifiedImageError
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns

DATASET_PATH = '/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dataset_pipeline/original_dataset_CSIRO/'
CLASSES = ['0', '1']

def analze_dataset():
    stats = { cls: 
             {'count': 0,
              'shape_set': set(),
              'mean': [],
              'stds': [],
              'min': [],
              'max': [],
              'corrupted': 0}
              for cls in CLASSES}
    for class_ in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_)
        for img_name in tqdm(os.listdir(class_path), desc= f"Analyzing class {class_}"):
            try:
                img = Image.open(os.path.join(class_path, img_name))
                img = np.array(img)
                stats[class_]['count'] += 1
                stats[class_]['shape_set'].add(img.shape)
                stats[class_]['mean'].append(np.mean(img))
                stats[class_]['stds'].append(np.std(img))
                stats[class_]['min'].append(np.min(img))
                stats[class_]['max'].append(np.max(img))
            except UnidentifiedImageError:
                stats[class_]['corrupted'] += 1
                print(f"Corrupted image: {os.path.join(class_path, img_name)}")
                continue
    return stats

def plot_stats(stats):
    for cls in CLASSES:
        print(f"Class: {cls}")
        print(f"Number of images: {stats[cls]['count']}")
        print(f"Number of corrupted images: {stats[cls]['corrupted']}")
        print(f"Unique shapes: {stats[cls]['shape_set']}")
        print(f"Mean: {np.mean(stats[cls]['mean'])}")
        print(f"Std: {np.mean(stats[cls]['stds'])}")
        print(f"Min: {np.mean(stats[cls]['min'])}")
        print(f"Max: {np.mean(stats[cls]['max'])}")
        print("\n")

        sns.histplot(stats[cls]['mean'], kde=True)
        plt.title(f"Mean distribution for class {cls}")
        plt.show()

        sns.histplot(stats[cls]['stds'], kde=True)
        plt.title(f"Std distribution for class {cls}")
        plt.show()

        sns.histplot(stats[cls]['min'], kde=True)
        plt.title(f"Min distribution for class {cls}")
        plt.show()

        sns.histplot(stats[cls]['max'], kde=True)
        plt.title(f"Max distribution for class {cls}")
        plt.show()

def show_sample_images():
    for class_ in CLASSES:
        class_path = os.path.join(DATASET_PATH, class_)
        sample_images = os.listdir(class_path)[:5]
        for img_name in enumerate(sample_images):
            img = Image.open(os.path.join(class_path, img_name))
            plt.imshow(img)
            plt.title(f"Class: {class_}")
            plt.show()
            break

if __name__ == "__main__":
     stats = analze_dataset()
     plot_stats(stats)
     show_sample_images()
     