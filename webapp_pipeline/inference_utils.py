# inference_utils.py
import torch
import cv2
import numpy as np
import time
from torchvision import transforms
from dynamic_perceiver_model_pipeline.perceiver_module import PerceiverClassifier
from dynamic_perceiver_model_pipeline.backbone_module_yolo import YOLOBackbone

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

def load_model(model_path):
    model = PerceiverClassifier()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def run_inference(image, model):
    crops, boxes = detect_patches(image)
    predictions = []
    confidences = []
    start_time = time.time()
    for crop in crops:
        x = transform(crop).unsqueeze(0)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred].item()
            predictions.append(pred)
            confidences.append(confidence)
    elapsed = time.time() - start_time
    return predictions, confidences, boxes, elapsed

def detect_patches(image):
    yolo = YOLOBackbone(model_path='yolov8n.pt')
    results = yolo.model(image)[0]
    crops = []
    boxes = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = image[y1:y2, x1:x2]
        if crop.size > 0:
            crops.append(crop)
            boxes.append((x1, y1, x2, y2))
    return crops, boxes

def draw_boxes(image, boxes, labels, confidences, color):
    img_copy = image.copy()
    for (x1, y1, x2, y2), label, conf in zip(boxes, labels, confidences):
        text = f"{label} ({conf:.2f})"
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img_copy, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_copy