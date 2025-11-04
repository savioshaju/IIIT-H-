import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os

# ------------------- CONFIG -------------------
DATA_DIR = "mfcc_features"
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6  # gujarati, hindi, kannada, malayalam, tamil, telugu

# ------------------- LOAD DATA -------------------
print("📦 Loading MFCC feature tensors...")
X_train = torch.load(os.path.join(DATA_DIR, "train_mfcc.pt"))
y_train = torch.load(os.path.join(DATA_DIR, "train_labels.pt"))
X_val = torch.load(os.path.join(DATA_DIR, "val_mfcc.pt"))
y_val = torch.load(os.path.join(DATA_DIR, "val_labels.pt"))

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=BATCH_SIZE)

print(f"✅ Data loaded | Train: {len(X_train)} | Val: {len(X_val)} | Device: {DEVICE}")

# ------------------- MODEL -------------------
class MFCC_CNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),

            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x: [B, 40, T]
        x = x.unsqueeze(1)  # -> [B, 1, 40, T]
        return self.net(x)

# ------------------- TRAINING SETUP -------------------
model = MFCC_CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ------------------- TRAIN LOOP -------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation phase
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    acc = correct / total * 100
    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

# ------------------- SAVE -------------------
torch.save(model.state_dict(), "mfcc_baseline_model.pt")
print("\n✅ Training complete. Model saved as 'mfcc_baseline_model.pt'")
