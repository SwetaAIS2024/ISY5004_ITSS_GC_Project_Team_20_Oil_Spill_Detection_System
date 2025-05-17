Workflow
Upload SAR image
Run YOLOv8 detection to extract patches
Load each of the 3 trained Perceiver models
Run inference on the patches
Measure and display:
Inference time
Predicted labels
Model confidence (softmax probs)
Accuracy / recall if ground truth is known (optional)
Overlay predictions on image
Display all results side by side
