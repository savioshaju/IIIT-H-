import os
import shutil
import random
import parselmouth
import numpy as np

DATASET_DIR = 'IndicAccentDB'  # Original dataset folder
OUTPUT_DIR = 'IndicAccentDS'  # Output folder: age -> split -> accent

SPLIT_RATIOS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
F0_THRESHOLD = 200  # Hz threshold for child vs adult classification
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

def create_dirs(base_path, age_groups, splits, accents):
    for age in age_groups:
        for split in splits:
            for accent in accents:
                path = os.path.join(base_path, age, split, accent)
                os.makedirs(path, exist_ok=True)

def get_median_f0(file_path):
    snd = parselmouth.Sound(file_path)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    # Filter out unvoiced frames (0 frequency)
    voiced_pitches = pitch_values[pitch_values > 0]
    if len(voiced_pitches) == 0:
        return 0
    median_f0 = np.median(voiced_pitches)
    return median_f0

def split_files(files):
    random.shuffle(files)
    n = len(files)
    train_end = int(SPLIT_RATIOS['train'] * n)
    val_end = train_end + int(SPLIT_RATIOS['val'] * n)
    return {
        'train': files[:train_end],
        'val': files[train_end:val_end],
        'test': files[val_end:]
    }

def main():
    accents = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    age_groups = ['child', 'adult']
    splits = list(SPLIT_RATIOS.keys())

    create_dirs(OUTPUT_DIR, age_groups, splits, accents)

    # Data holders by age group and accent
    data_age_accent = {age: {accent: [] for accent in accents} for age in age_groups}

    print("Starting fundamental frequency based age classification...")

    for accent in accents:
        accent_dir = os.path.join(DATASET_DIR, accent)
        files = [f for f in os.listdir(accent_dir) if os.path.isfile(os.path.join(accent_dir, f))]

        for file_name in files:
            file_path = os.path.join(accent_dir, file_name)
            try:
                median_f0 = get_median_f0(file_path)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
            
            age_group = 'child' if median_f0 >= F0_THRESHOLD else 'adult'
            data_age_accent[age_group][accent].append(file_name)

        print(f"{accent} classification done: Child={len(data_age_accent['child'][accent])}, Adult={len(data_age_accent['adult'][accent])}")

    print("Starting train/val/test split and file copy...")

    for age in age_groups:
        for accent in accents:
            files = data_age_accent[age][accent]
            if not files:
                continue

            splits_dict = split_files(files)

            for split, split_files_list in splits_dict.items():
                for file_name in split_files_list:
                    src = os.path.join(DATASET_DIR, accent, file_name)
                    dst = os.path.join(OUTPUT_DIR, age, split, accent, file_name)
                    shutil.copy2(src, dst)

            print(f"{age} - {accent}: Train={len(splits_dict['train'])}, Val={len(splits_dict['val'])}, Test={len(splits_dict['test'])}")

    print("Dataset splitting complete.")

if __name__ == '__main__':
    main()
