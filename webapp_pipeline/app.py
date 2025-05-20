import sys
import os
import streamlit as st
import torch
from PIL import Image
import numpy as np

# Set root path so imports work
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

from inference_utils import load_model, run_inference

# Streamlit setup
st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("Oil Spill Detection using Dynamic Perceiver Model trained on SAR Images")

# Upload SAR image
uploaded_file = st.file_uploader("Upload a SAR image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

# Load models (cached)
@st.cache_resource
def load_all_models():
    models = {}
    for name, path in {
#         "Original": "perceiver_original.pt",
#         "Synthetic": "perceiver_synthetic.pt",
        "Perceiver": "perceiver_combined.pt"
    }.items():
        model = load_model(path)
        models[name] = model
    return models

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    #st.image(image, caption="Uploaded SAR Image", use_container_width=True)
    st.image(image, caption="Uploaded SAR Image", width=300)

    st.subheader("Inference Results")

    models = load_all_models()
    cols = st.columns(len(models))

    for i, (name, model) in enumerate(models.items()):
        with cols[i]:
            st.markdown(f"### {name} Model")
            pred, conf, elapsed = run_inference(image, model)
            label = "Oil Spill" if pred == 1 else "Non-Spill"
            st.write(f"Prediction   : **{label}**")
            st.write(f"Confidence   : **{conf:.2f}**")
            st.write(f"Inference Time: **{elapsed:.3f} sec**")