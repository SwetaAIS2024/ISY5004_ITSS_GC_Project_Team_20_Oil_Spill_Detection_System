import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from perceiver_module import PerceiverClassifier

def train_model(train_dataset, epochs=10, lr=1e-4, save_path='perceiver.pt'):
    model = PerceiverClassifier()
    dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
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

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")