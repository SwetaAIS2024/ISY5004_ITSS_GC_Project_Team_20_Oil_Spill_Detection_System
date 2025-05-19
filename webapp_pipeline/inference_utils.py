import sys
import os
import torch
import time
from torchvision import transforms
from PIL import Image
import numpy as np

# Add the project root to Python path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from dynamic_perceiver_model_pipeline.cnn_perceiver_module import CNNPerceiverClassifier

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def load_model(model_path):
    model = CNNPerceiverClassifier(cnn_weights="mobilenet_sar.pt")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def run_inference(image, model):
    """
    image: PIL Image (RGB)
    model: CNNPerceiverClassifier
    Returns: pred class (0 or 1), confidence score, inference time
    """
    x = transform(image).unsqueeze(0)  # (1, 3, 64, 64)
    start_time = time.time()
    with torch.no_grad():
        output = model(x)
        probs = torch.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()
    elapsed = time.time() - start_time
    return pred, confidence, elapsed