from ultralytics import YOLO
import cv2
import torch
import torchvision.transforms as T

class YOLOBackbone:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
        self.resize = T.Compose([
            T.ToPILImage(),
            T.Resize((64, 64)),
            T.ToTensor()
        ])

    def detect_and_crop(self, image_path):
        image = cv2.imread(image_path)
        results = self.model(image)[0]
        crops = []
        boxes = []
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box.tolist())
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_tensor = self.resize(crop)
            crops.append(crop_tensor)
            boxes.append((x1, y1, x2, y2))
        return crops, boxes