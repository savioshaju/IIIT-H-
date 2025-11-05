

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "hubert_features"
BATCH_SIZE = 64
EPOCHS = 30
LR = 2e-4
INPUT_DIM = 768
HIDDEN_DIM = 512
NUM_CLASSES = 6
N_DOMAINS = 4
DOMAIN_LOSS_WEIGHT = 0.5
SEED = 42
BEST_MODEL_PATH = "accent_dann_model.pt"
os.makedirs(os.path.dirname(BEST_MODEL_PATH) or ".", exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.startswith("cuda"):
    torch.cuda.manual_seed_all(SEED)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def load_split(name):
    feat_path = os.path.join(DATA_DIR, name, f"{name}_hubert.pt")
    lab_path = os.path.join(DATA_DIR, name, f"{name}_labels.pt")
    if not os.path.exists(feat_path) or not os.path.exists(lab_path):
        raise FileNotFoundError(f"Missing files for split '{name}': {feat_path} or {lab_path}")
    feats = torch.load(feat_path)
    labels = torch.load(lab_path)
    labels = labels.long().view(-1)
    return feats, labels

X_train, y_train = load_split("train")
X_val, y_val = load_split("val")
X_test, y_test = load_split("test")
print(f"Loaded shapes | train: {X_train.shape}, val: {X_val.shape}, test: {X_test.shape}")

X_train = X_train.float()
X_val = X_val.float()
X_test = X_test.float()

print("Running KMeans on train features to create pseudo-domain labels...")
kmeans = KMeans(n_clusters=N_DOMAINS, random_state=SEED, n_init=10)
kmeans.fit(X_train.cpu().numpy().astype(np.float32))
domain_train = torch.from_numpy(kmeans.labels_).long()
domain_val = torch.from_numpy(kmeans.predict(X_val.cpu().numpy().astype(np.float32))).long()
domain_test = torch.from_numpy(kmeans.predict(X_test.cpu().numpy().astype(np.float32))).long()

print("Domain distribution (train):", np.bincount(domain_train.numpy()))
print("Domain distribution (val):", np.bincount(domain_val.numpy()))
print("Domain distribution (test):", np.bincount(domain_test.numpy()))

class_counts = torch.bincount(y_train)
print("Class counts (train):", class_counts.tolist())
class_weights = 1.0 / (class_counts.float() + 1e-8)
sample_weights = class_weights[y_train]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_ds = TensorDataset(X_train, y_train, domain_train)
val_ds = TensorDataset(X_val, y_val, domain_val)
test_ds = TensorDataset(X_test, y_test, domain_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, drop_last=False)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

class GRL(nn.Module):
    def forward(self, x, lambd):
        return GradientReversal.apply(x, lambd)

class DANNModel(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES, n_domains=N_DOMAINS):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.domain_disc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, n_domains)
        )
        self.grl = GRL()

    def forward(self, x, grl_lambda=0.0):
        feat = self.encoder(x)
        class_logits = self.classifier(feat)
        rev_feat = self.grl(feat, grl_lambda)
        domain_logits = self.domain_disc(rev_feat)
        return class_logits, domain_logits

model = DANNModel().to(DEVICE)
print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# --------------- OPT / LOSS ----------------
criterion_class = nn.CrossEntropyLoss()
criterion_domain = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# GRL schedule (from DANN paper style)
def grl_lambda_schedule(epoch, max_epoch, alpha=10.0):
    p = float(epoch) / float(max_epoch)
    lam = 2.0 / (1.0 + math.exp(-alpha * p)) - 1.0
    return float(lam)

# --------------- TRAIN LOOP ----------------
best_val_acc = 0.0
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    lam = grl_lambda_schedule(epoch, EPOCHS)
    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")
    for xb, yb, db in loop:
        xb = xb.to(DEVICE).float()
        yb = yb.to(DEVICE).long()
        db = db.to(DEVICE).long()

        optimizer.zero_grad()
        class_logits, domain_logits = model(xb, grl_lambda=lam)
        loss_c = criterion_class(class_logits, yb)
        loss_d = criterion_domain(domain_logits, db)
        loss = loss_c + DOMAIN_LOSS_WEIGHT * loss_d
        loss.backward()
        optimizer.step()

        preds = class_logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
        total_loss += loss.item() * xb.size(0)
        loop.set_postfix(train_acc=100.0 * correct / total, loss=loss.item(), grl_lambda=lam)

    scheduler.step()

    train_acc = correct / total if total > 0 else 0.0

    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for xb, yb, db in val_loader:
            xb = xb.to(DEVICE).float()
            yb = yb.to(DEVICE).long()
            class_logits, _ = model(xb, grl_lambda=0.0)
            preds = class_logits.argmax(dim=1)
            val_correct += (preds == yb).sum().item()
            val_total += xb.size(0)
    val_acc = val_correct / max(1, val_total)
    avg_loss = total_loss / max(1, total)

    print(f"Epoch {epoch} summary -> Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}% | avg_loss: {avg_loss:.4f} | grl_lambda={lam:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc
        }, BEST_MODEL_PATH)
        print("✅ Best model saved.")

# --------------- FINAL EVAL ----------------
checkpoint = torch.load(BEST_MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.eval()

all_preds = []
all_true = []
with torch.no_grad():
    for xb, yb, db in test_loader:
        xb = xb.to(DEVICE).float()
        yb = yb.to(DEVICE).long()
        class_logits, _ = model(xb, grl_lambda=0.0)
        preds = class_logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_true.extend(yb.cpu().numpy())

print("\n=== Final Test Classification Report ===")
print(classification_report(all_true, all_preds, digits=4))
# Compute confusion matrix
cm = confusion_matrix(all_true, all_preds)
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
            xticklabels=[f"Class {i}" for i in range(NUM_CLASSES)],
            yticklabels=[f"Class {i}" for i in range(NUM_CLASSES)])
plt.title("Normalized Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()
