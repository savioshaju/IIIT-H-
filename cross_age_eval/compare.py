import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ===============================================================
# 1. Dataset Loader
# ===============================================================
class FeatureDataset(Dataset):
    def __init__(self, root_dir):
        self.paths, self.labels = [], []
        self.label2idx = {label: i for i, label in enumerate(sorted(os.listdir(root_dir)))}
        for label in self.label2idx:
            accent_dir = os.path.join(root_dir, label)
            if not os.path.isdir(accent_dir):
                continue
            for file in os.listdir(accent_dir):
                if file.endswith(".pt"):
                    self.paths.append(os.path.join(accent_dir, file))
                    self.labels.append(self.label2idx[label])

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        x = torch.load(self.paths[idx])
        y = torch.tensor(self.labels[idx])
        return x, y


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
# 3. Training & Evaluation
# ===============================================================
def train_model(model, train_loader, val_loader, epochs=20, lr=1e-3, device="cuda"):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    train_losses, val_accs = [], []

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

        # Validation accuracy
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds.extend(out.argmax(1).cpu().numpy())
                labels.extend(y.cpu().numpy())
        val_acc = accuracy_score(labels, preds)

        train_losses.append(total_loss / len(train_loader))
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1:02d}/{epochs} | Loss: {train_losses[-1]:.4f} | Val Acc: {val_acc:.4f}")

    return model, train_losses, val_accs


def evaluate_model(model, test_loader, label_names, split_name="Test", device="cuda"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            preds.extend(out.argmax(1).cpu().numpy())
            labels.extend(y.cpu().numpy())

    acc = accuracy_score(labels, preds)
    print(f"\n=== {split_name} Results ===")
    print(f"Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=label_names))

    # Confusion Matrix
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"{split_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()
    return acc


# ===============================================================
# 4. Main Pipeline
# ===============================================================
def run_experiment(feature_root, input_dim, device="cuda"):
    print(f"\n🚀 Running experiment for {feature_root.upper()}")

    # Load datasets
    train_ds = FeatureDataset(os.path.join(feature_root, "adult/train"))
    val_ds   = FeatureDataset(os.path.join(feature_root, "adult/val"))
    test_adult_ds = FeatureDataset(os.path.join(feature_root, "adult/test"))
    test_child_ds = FeatureDataset(os.path.join(feature_root, "child/test"))

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=32)
    test_adult_loader = DataLoader(test_adult_ds, batch_size=32)
    test_child_loader = DataLoader(test_child_ds, batch_size=32)

    label_names = list(train_ds.label2idx.keys())
    num_classes = len(label_names)
    model = AccentClassifier(input_dim, num_classes)

    # Train
    model, losses, val_accs = train_model(model, train_loader, val_loader, device=device)

    # Save model
    model_path = f"{feature_root}_accent_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Model saved to: {model_path}")

    # Plot loss/accuracy curves
    plt.figure(figsize=(8, 5))
    plt.plot(losses, label="Train Loss")
    plt.plot(val_accs, label="Val Accuracy")
    plt.title(f"{feature_root.upper()} - Training Curve")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Evaluate
    adult_acc = evaluate_model(model, test_adult_loader, label_names, "Adult Test", device)
    child_acc = evaluate_model(model, test_child_loader, label_names, "Child Test", device)

    return adult_acc, child_acc


# ===============================================================
# 5. Compare MFCC vs HuBERT
# ===============================================================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mfcc_root = "features_mfcc"
    hubert_root = "feature_hubert"

    results = {}
    results["MFCC"] = run_experiment(mfcc_root, input_dim=40, device=device)
    results["HuBERT"] = run_experiment(hubert_root, input_dim=768, device=device)

    # Final Comparison
    print("\n=== Final Accuracy Comparison ===")
    print("Feature\tAdult Acc\tChild Acc")
    for k, (a, c) in results.items():
        print(f"{k}\t{a:.3f}\t\t{c:.3f}")

    # Visualization
    plt.figure(figsize=(6, 4))
    x = np.arange(len(results))
    width = 0.35
    adult_accs = [v[0] for v in results.values()]
    child_accs = [v[1] for v in results.values()]

    plt.bar(x - width/2, adult_accs, width, label="Adult")
    plt.bar(x + width/2, child_accs, width, label="Child")

    plt.xticks(x, results.keys())
    plt.ylabel("Accuracy")
    plt.title("MFCC vs HuBERT Accent Classification Performance")
    plt.legend()
    plt.tight_layout()
    plt.show()
