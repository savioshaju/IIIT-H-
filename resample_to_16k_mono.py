import os
import torchaudio

SRC_DIR = "IndicAccentDB_clean_split"   
DST_DIR = "IndicAccentDB_16k"          
TARGET_SR = 16000

os.makedirs(DST_DIR, exist_ok=True)

for split in ["train", "val", "test"]:
    split_src = os.path.join(SRC_DIR, split)
    split_dst = os.path.join(DST_DIR, split)
    os.makedirs(split_dst, exist_ok=True)

    for accent in os.listdir(split_src):
        src_accent_dir = os.path.join(split_src, accent)
        dst_accent_dir = os.path.join(split_dst, accent)
        os.makedirs(dst_accent_dir, exist_ok=True)

        for file in os.listdir(src_accent_dir):
            if not file.endswith(".wav"):
                continue
            src_path = os.path.join(src_accent_dir, file)
            dst_path = os.path.join(dst_accent_dir, file)

            try:
                wav, sr = torchaudio.load(src_path)

                # Convert stereo → mono
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)

                # Resample to 16kHz if needed
                if sr != TARGET_SR:
                    wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)

                torchaudio.save(dst_path, wav, TARGET_SR)
            except Exception as e:
                print(f"❌ Failed: {src_path} | {e}")

print("✅ Conversion complete → All audio is now 16 kHz mono.")


# # # import torch

# # # # Assuming your model has these components
# # # encoder = model.encoder
# # # classifier = model.classifier

# # # # Combine into a single exportable module
# # # class AccentClassifier(torch.nn.Module):
# # #     def __init__(self, encoder, classifier):
# # #         super().__init__()
# # #         self.encoder = encoder
# # #         self.classifier = classifier

# # #     def forward(self, x):
# # #         feats = self.encoder(x)
# # #         out = self.classifier(feats)
# # #         return out

# # # # Build export model
# # # export_model = AccentClassifier(encoder, classifier).eval()

# # # # Save
# # # torch.save(export_model.state_dict(), "accent_classifier_16k.pt")
# # # print("✅ Model exported: accent_classifier_16k.pt")


# # import torchaudio
# # import torch
# # import os

# # AUG_DIR = "IndicAccentDB_augmented/train/gujarat"
# # os.makedirs(AUG_DIR, exist_ok=True)

# # def augment_audio(path, out_path):
# #     waveform, sr = torchaudio.load(path)
# #     # Add noise
# #     noise = torch.randn_like(waveform) * 0.005
# #     waveform = waveform + noise
# #     # Pitch shift
# #     waveform = torchaudio.functional.pitch_shift(waveform, sr, n_steps=2)
# #     torchaudio.save(out_path, waveform, sr)

# # for file in os.listdir("IndicAccentDB_clean_split/train/gujarat"):
# #     if file.endswith(".wav"):
# #         augment_audio(f"IndicAccentDB_clean_split/train/gujarat/{file}",
# #                       f"{AUG_DIR}/aug_{file}")

# import torch
# import os

# splits = ["train", "val", "test"]
# base_dir = "hubert_features"

# for split in splits:
#     lab_path = os.path.join(base_dir, split, f"{split}_labels.pt")
#     if not os.path.exists(lab_path):
#         print(f"❌ Missing: {lab_path}")
#         continue

#     labels = torch.load(lab_path)
#     print(f"\n--- {split.upper()} ---")
#     print("Total samples:", len(labels))
#     print("Label counts:", torch.bincount(labels).tolist())
