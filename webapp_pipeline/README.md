1. ** Web UI ** 
- Upload SAR image
- Trigger backend API
2. ** Backend API (Flask / FastAPI) **
- Loads yolov8n.pt (for region proposals)
- Loads perceiver.pt (for classification)
- Runs detection and classification
- Returns prediction results with bounding boxes
3. ** Frontend **
- Displays image with detected oil spill areas
- Shows labels (spill / non-spill) and confidence scores