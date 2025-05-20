import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from cnn_perceiver_module import CNNPerceiverClassifier

def evaluate_multiple_models(test_dataset, model_paths):
    results = {}
    dataloader = DataLoader(test_dataset, batch_size=8)

    for model_name, path in model_paths.items():
        model = CNNPerceiverClassifier(cnn_weights='mobilenet_sar.pt')
        model.load_state_dict(torch.load(path, map_location='cpu'))
        model.eval()

        y_true = []
        y_pred = []

        with torch.no_grad():
            for x, y in dataloader:
                out = model(x)
                preds = torch.argmax(out, dim=1)
                y_true.extend(y.tolist())
                y_pred.extend(preds.tolist())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='binary')
        rec = recall_score(y_true, y_pred, average='binary')
        f1 = f1_score(y_true, y_pred, average='binary')
        cm = confusion_matrix(y_true, y_pred)

        results[model_name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "conf_matrix": cm
        }

    return results

def display_results(results):
    for name, metrics in results.items():
        print(f"\nModel: {name}")
        print(f"Accuracy : {metrics['accuracy']*100:.2f}%")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall   : {metrics['recall']:.2f}")
        print(f"F1-Score : {metrics['f1']:.2f}")
        
        disp = ConfusionMatrixDisplay(
            confusion_matrix=metrics["conf_matrix"],
            display_labels=["Non-Spill", "Oil Spill"]
        )
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {name}")
        plt.show()