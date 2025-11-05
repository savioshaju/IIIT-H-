import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------- CONFIG -------------------
DATA_DIR = "mfcc_features"
BATCH_SIZE = 64
EPOCHS = 25
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 6  # classes: 0–5

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
        x = x.unsqueeze(1)  # [B, 1, 40, T]
        return self.net(x)

# ------------------- TRAINING SETUP -------------------
model = MFCC_CNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

train_losses, val_accuracies = [], []

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
    val_preds, val_labels = [], []

    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            preds = model(X).argmax(1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(y.cpu().numpy())

    val_preds = np.array(val_preds)
    val_labels = np.array(val_labels)

    acc = (val_preds == val_labels).mean() * 100
    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    val_accuracies.append(acc)

    print(f"\nEpoch {epoch+1:02d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {acc:.2f}%")
    print("📊 Classification Report:")
    print(classification_report(val_labels, val_preds, digits=4, labels=[0, 1, 2, 3, 4, 5]))

# ------------------- CONFUSION MATRIX -------------------
cm = confusion_matrix(val_labels, val_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(7, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)])
plt.title("Normalized Confusion Matrix (Validation Set)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

# ------------------- TRAINING CURVES -------------------
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label="Training Loss", color='red')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label="Validation Accuracy (%)", color='blue')
plt.title("Validation Accuracy Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()

# ------------------- SAVE MODEL -------------------
torch.save(model.state_dict(), "mfcc_baseline_model.pt")
print("\n✅ Training complete. Model saved as 'mfcc_baseline_model.pt'")
