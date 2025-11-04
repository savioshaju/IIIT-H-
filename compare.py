import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ===============================================================
# 1. Dataset Loader
# ===============================================================
class FeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = []
        self.labels = []
        self.label2idx = {label: i for i, label in enumerate(sorted(os.listdir(root_dir)))}
        for label in self.label2idx:
            accent_dir = os.path.join(root_dir, label)
            for file in os.listdir(accent_dir):
                if file.endswith(".pt"):
                    self.paths.append(os.path.join(accent_dir, file))
                    self.labels.append(self.label2idx[label])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        x = torch.load(self.paths[idx])
        y = self.labels[idx]
        return x, torch.tensor(y)


# ===============================================================
# 2. Model
# ===============================================================
class AccentClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# ===============================================================
# 3. Train / Test Helpers
# ===============================================================
def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3, device="cuda"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        val_acc = accuracy_score(labels, preds)
        print(f"Epoch {epoch+1}: loss={total_loss/len(train_loader):.4f}, val_acc={val_acc:.3f}")

def test_model(model, test_loader, device="cuda"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return accuracy_score(labels, preds)


# ===============================================================
# 4. Main Comparison Pipeline
# ===============================================================
def run_experiment(feature_root, input_dim, device="cuda"):
    print(f"\n=== Running experiment for {feature_root.split('/')[-1].upper()} ===")

    train_ds = FeatureDataset(os.path.join(feature_root, "adult/train"))
    val_ds   = FeatureDataset(os.path.join(feature_root, "adult/val"))
    test_ds  = FeatureDataset(os.path.join(feature_root, "child/test"))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)
    test_loader  = DataLoader(test_ds, batch_size=32)

    num_classes = len(train_ds.label2idx)
    model = AccentClassifier(input_dim=input_dim, num_classes=num_classes)

    train_model(model, train_loader, val_loader, epochs=10, device=device)
    acc = test_model(model, test_loader, device=device)
    print(f"Child Test Accuracy ({feature_root.split('/')[-1]}): {acc:.3f}\n")
    return acc


# ===============================================================
# 5. Run Both MFCC and HuBERT Models
# ===============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Paths to extracted features
    mfcc_root = "features_mfcc"
    hubert_root = "feature_hubert"

    acc_mfcc = run_experiment(mfcc_root, input_dim=40, device=device)
    acc_hubert = run_experiment(hubert_root, input_dim=768, device=device)

    # ===============================================================
    # 6. Compare
    # ===============================================================
    print("=== Final Comparison ===")
    print(f"MFCC   → Child Test Accuracy: {acc_mfcc:.3f}")
    print(f"HuBERT → Child Test Accuracy: {acc_hubert:.3f}")

    plt.bar(["MFCC", "HuBERT"], [acc_mfcc, acc_hubert], color=["gray", "steelblue"])
    plt.ylabel("Child Test Accuracy")
    plt.title("Accent Model Age Generalization Comparison")
    plt.show()
