# app.py
# Add the project root to Python path
import sys
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
import streamlit as st
import cv2
import numpy as np
from inference_utils import (
    load_model, run_inference, draw_boxes
)

st.set_page_config(page_title="Oil Spill Detection Comparator", layout="wide")
st.title("Oil Spill Detection using YOLO + Perceiver Models")

uploaded_file = st.file_uploader("Upload a SAR image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert uploaded image to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    st.image(image, caption="Uploaded SAR Image", use_container_width=True)

    # Load models
    with st.spinner("Loading models..."):
        model_org = load_model("/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dynamic_perceiver_model_pipeline/perceiver_original.pt")
        model_syn = load_model("/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dynamic_perceiver_model_pipeline/perceiver_synthetic.pt")
        model_com = load_model("/Users/swetapattnaik/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/ISY5004_ITSS_GC_Project_Team_20_Oil_Spill_Detection_System/dynamic_perceiver_model_pipeline/perceiver_combined.pt")

    # Inference
    st.subheader("Model Inference Results")

    cols = st.columns(3)
    models = [("Original", model_org, (255, 0, 0)), 
              ("Synthetic", model_syn, (0, 255, 0)), 
              ("Combined", model_com, (0, 0, 255))]

    for i, (name, model, color) in enumerate(models):
        with cols[i]:
            st.markdown(f"### {name} Model")
            preds, confs, boxes, t = run_inference(image, model)
            img_result = draw_boxes(image, boxes, preds, confs, color)
            st.image(img_result, caption=f"{name} Predictions", use_column_width=True)
            st.write(f"**Inference Time:** {t:.2f} sec")
            st.write(f"**Num Patches:** {len(preds)}")
            st.write(f"**Predictions:** {preds}")
            st.write(f"**Confidences:** {[round(c, 2) for c in confs]}")