import torch
from torch.utils.data import DataLoader
from perceiver_module import PerceiverClassifier

def evaluate_model(test_dataset, model_path='perceiver.pt', dataset_type='original'):
    model = PerceiverClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    dataloader = DataLoader(test_dataset, batch_size=8)
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in dataloader:
            out = model(x)
            preds = torch.argmax(out, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"[{dataset_type.upper()}] Accuracy: {100 * correct / total:.2f}%")