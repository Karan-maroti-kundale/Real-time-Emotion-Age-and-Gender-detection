# src/datasets.py
import os
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 128

# TRAIN transforms include augmentation
train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# VAL / INFER transforms — deterministic
val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# fixed order expected by model / inference
FIXED_EMOTION_ORDER = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


# ============================================================
# EMOTION DATASET
# ============================================================
class EmotionDataset(Dataset):
    """
    Loads emotion dataset arranged as:
    root/<label_folder>/*.jpg
    The loader uses FIXED_EMOTION_ORDER to ensure labels align with model.
    phase: "train" or "val" or "infer"
    """

    def __init__(self, root, phase="train"):
        assert phase in ("train", "val", "infer")
        self.samples = []
        self.phase = phase

        # find available classes in the fixed order
        available = [c for c in FIXED_EMOTION_ORDER if os.path.isdir(os.path.join(root, c))]
        if len(available) == 0:
            raise RuntimeError(f"No emotion folders found inside {root}. Expected subfolders: {FIXED_EMOTION_ORDER}")

        # map class name -> index (0..)
        self.class_to_idx = {c: i for i, c in enumerate(available)}

        for cname in available:
            folder = os.path.join(root, cname)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(folder, fname), self.class_to_idx[cname]))

        self.transform = train_transform if phase == "train" else val_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # if unreadable, return a black image (rare)
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = self.transform(img)
        return img_t, torch.tensor(label, dtype=torch.long)


# ============================================================
# AGE + GENDER (UTKFace-style) dataset
# ============================================================
class AgeGenderDataset(Dataset):
    """
    Accepts either a path string or list of folders with UTKFace filenames:
    filenames like: 25_0_0_20170109150557335.jpg
    (age_gender_...)
    """

    def __init__(self, roots, phase="train"):
        if isinstance(roots, str):
            roots = [roots]
        self.samples = []
        self.phase = phase

        for root in roots:
            for rootdir, _, files in os.walk(root):
                for f in files:
                    if not f.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    parts = f.split("_")
                    if len(parts) < 2:
                        continue
                    try:
                        age = int(parts[0])
                        gender = int(parts[1])
                    except Exception:
                        continue
                    if gender not in (0, 1):
                        continue
                    if age < 0 or age > 120:
                        continue
                    self.samples.append((os.path.join(rootdir, f), age, gender))

        self.transform = train_transform if phase == "train" else val_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, age, gender = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_t = self.transform(img)
        return img_t, (torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long))
