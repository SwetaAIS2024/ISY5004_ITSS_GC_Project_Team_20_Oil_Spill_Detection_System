import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader

class MobileNetSARSimple(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()
        self.base = models.mobilenet_v3_small(pretrained=True)
        self.base.classifier[3] = nn.Linear(self.base.classifier[3].in_features, output_dim)

    def forward(self, x):
        return self.base(x)

def pretrain_cnn(dataset, epochs=10, lr=1e-4, save_path="mobilenet_sar.pt"):
    model = MobileNetSARSimple()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

    torch.save(model.base.features.state_dict(), save_path)
    print(f"Saved MobileNet feature weights to {save_path}")