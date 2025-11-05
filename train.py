import os
import torch
import joblib
from torch import nn, optim
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, f1_score

FEATURE_DIR = "IndicAccentDB_16k_features_hubert"
MODEL_SAVE_DIR = "models_accent_hubert"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 64
EPOCHS = 35
LEARNING_RATE = 1e-4
HIDDEN_DIM = 1024
DROPOUT = 0.4

def load_features(level, split):
    path = os.path.join(FEATURE_DIR, level, f"{split}.pkl")
    data = joblib.load(path)
    return torch.tensor(data["features"], dtype=torch.float32), data["labels"]

def make_batches(x, y, batch_size):
    for i in range(0, len(x), batch_size):
        yield x[i:i+batch_size], y[i:i+batch_size]

class AccentClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    def forward(self, x):
        return self.net(x)

def train_model(level):
    x_train, y_train = load_features(level, "train")
    x_val, y_val = load_features(level, "val")
    x_test, y_test = load_features(level, "test")

    le = LabelEncoder()
    y_train_enc = torch.tensor(le.fit_transform(y_train), dtype=torch.long)
    y_val_enc = torch.tensor(le.transform(y_val), dtype=torch.long)
    y_test_enc = torch.tensor(le.transform(y_test), dtype=torch.long)

    input_dim = x_train.shape[1]
    num_classes = len(le.classes_)

    class_counts = torch.tensor([sum(y_train_enc == i) for i in range(num_classes)], dtype=torch.float)
    class_weights = 1.0 / (class_counts + 1e-6)
    class_weights = class_weights / class_weights.sum() * num_classes
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

    model = AccentClassifier(input_dim, HIDDEN_DIM, num_classes, DROPOUT).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    best_val_f1 = 0

    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(x_train))
        x_train_shuffled = x_train[perm].to(DEVICE)
        y_train_shuffled = y_train_enc[perm].to(DEVICE)
        train_loss = 0

        for xb, yb in make_batches(x_train_shuffled, y_train_shuffled, BATCH_SIZE):
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_out = model(x_val.to(DEVICE))
            val_pred = torch.argmax(val_out, dim=1)
            val_f1 = f1_score(y_val_enc.cpu().numpy(), val_pred.cpu().numpy(), average="macro")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, f"{level}_best.pt"))

        print(f"{level.upper()} | Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.3f} | Val F1: {val_f1:.4f}")

    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, f"{level}_best.pt")))
    model.eval()

    with torch.no_grad():
        test_out = model(x_test.to(DEVICE))
        test_pred = torch.argmax(test_out, dim=1)
        test_acc = accuracy_score(y_test_enc.cpu().numpy(), test_pred.cpu().numpy())
        report = classification_report(y_test_enc.cpu().numpy(), test_pred.cpu().numpy(), target_names=le.classes_, digits=4)

    print(f"\n{level.upper()} Model Test Accuracy: {test_acc:.4f}")
    print(f"\nClassification Report:\n{report}")

for level in ["word", "sentence"]:
    train_model(level)
