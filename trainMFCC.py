import os
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB
import torch.nn as nn
import torch.optim as optim

class AccentAudioDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to train/test/val folder for adult or child (e.g., .../adult/train)
        Each subfolder in root_dir is a class label (e.g. andhra_pradesh, gujrat...)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []  # (filepath, label)
        self.classes = sorted(os.listdir(root_dir))  # list of class names (states)
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        for cls_name in self.classes:
            cls_folder = os.path.join(root_dir, cls_name)
            for fname in os.listdir(cls_folder):
                if fname.endswith(".wav"):  # assuming wav audio
                    self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls_name]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(filepath)  # Load audio

        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label

# Audio transforms: MelSpectrogram + Decibel scaling
audio_transform = nn.Sequential(
    MelSpectrogram(sample_rate=16000, n_mels=64),
    AmplitudeToDB()
)

# Function to collate batch with variable length audio (pad to max length in batch)
def collate_fn(batch):
    waveforms, labels = zip(*batch)
    lengths = [w.shape[1] for w in waveforms]
    max_len = max(lengths)
    padded_waveforms = []
    for w in waveforms:
        pad_len = max_len - w.shape[1]
        padded_waveforms.append(torch.nn.functional.pad(w, (0, pad_len)))
    return torch.stack(padded_waveforms), torch.tensor(labels)

# Paths
BASE_PATH = "D:/Savio Shaju/IIITH/Project/IndicAccentDS"

def get_dataloader(adult_or_child, split, batch_size=32):
    dataset_path = os.path.join(BASE_PATH, adult_or_child, split)
    dataset = AccentAudioDataset(dataset_path, transform=audio_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(split=='train'), collate_fn=collate_fn)
    return loader

# Example: load adult train loader
train_loader = get_dataloader("adult", "train")

# Simple CNN model for audio classification (modify for better accuracy)
class AudioClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(32*16*16, num_classes)  # Adjust dims depending on input size

    def forward(self, x):
        # x shape: (batch, channels=1, freq, time)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Instantiate model for number of states in adult dataset
num_classes = len(os.listdir(os.path.join(BASE_PATH, "adult", "train")))
model = AudioClassifier(num_classes).to('cuda' if torch.cuda.is_available() else 'cpu')

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop skeleton
def train_one_epoch(loader):
    model.train()
    running_loss = 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.to(model.device), labels.to(model.device)
        waveforms = waveforms.unsqueeze(1)  # add channel dim for CNN (batch,1, freq, time)
        optimizer.zero_grad()
        outputs = model(waveforms)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)

# Evaluation function
def evaluate(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for waveforms, labels in loader:
            waveforms, labels = waveforms.to(model.device), labels.to(model.device)
            waveforms = waveforms.unsqueeze(1)
            outputs = model(waveforms)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# Usage example for 10 epochs
train_loader = get_dataloader("adult", "train")
val_loader = get_dataloader("adult", "val")

for epoch in range(10):
    loss = train_one_epoch(train_loader)
    val_acc = evaluate(val_loader)
    print(f"Epoch {epoch+1}: Loss={loss:.4f}, Val Acc={val_acc:.4f}")
