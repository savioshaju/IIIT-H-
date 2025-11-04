import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class AccentDataset(Dataset):
    def __init__(self, base_dir, accent_map):
        self.data = []
        self.labels = []
        for accent, idx in accent_map.items():
            folder = os.path.join(base_dir, accent)
            for f in os.listdir(folder):
                if f.endswith(".pt"):
                    self.data.append(os.path.join(folder, f))
                    self.labels.append(idx)
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = torch.load(self.data[idx])
        y = torch.tensor(self.labels[idx])
        return x, y

class AccentClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=512, num_classes=6):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x): return self.model(x)

def train_model(train_loader, val_loader, save_path):
    model = AccentClassifier().to(DEVICE)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(25):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(X), y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: {total_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"✅ Model saved: {save_path}")
