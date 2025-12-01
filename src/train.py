# src/train.py
import os
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# support running as module (`python -m src.train`) and as a script (`python src/train.py`)
try:
    from .datasets import EmotionDataset, AgeGenderDataset, FIXED_EMOTION_ORDER
    from .model import MultiTaskCNN  # your model.py
except Exception:
    from datasets import EmotionDataset, AgeGenderDataset, FIXED_EMOTION_ORDER
    from model import MultiTaskCNN  # your model.py

# -----------------------
# Config
# -----------------------
DATA_ROOT = "data"
EMOTION_ROOT = os.path.join(DATA_ROOT, "Emotion", "train")
AGEG_ROOTS = [os.path.join(DATA_ROOT, "Age_Gender", "UTKFace"),
              os.path.join(DATA_ROOT, "Age_Gender", "crop_part1")]

BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_EMOTIONS = len(FIXED_EMOTION_ORDER)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# Seed
# -----------------------
def set_seed(s=42):
    random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

set_seed(42)

# -----------------------
# Datasets / Loaders
# -----------------------
print("Loading datasets...")
emotion_full = EmotionDataset(EMOTION_ROOT, phase="train")
print("Emotion samples:", len(emotion_full))
if len(emotion_full) == 0:
    raise RuntimeError("No emotion samples found; check path.")

# create simple val split 90/10
val_split = int(0.9 * len(emotion_full))
train_subset = torch.utils.data.Subset(emotion_full, list(range(0, val_split)))
val_subset = torch.utils.data.Subset(emotion_full, list(range(val_split, len(emotion_full))))

# compute class weights (inverse frequency)
labels = [emotion_full.samples[i][1] for i in range(len(emotion_full))]
cnt = Counter(labels)
class_weights = [0.0] * NUM_EMOTIONS
for cls in range(NUM_EMOTIONS):
    class_weights[cls] = 1.0 / (cnt.get(cls, 1))

# Weighted sampler for train to balance classes
train_labels = [emotion_full.samples[i][1] for i in range(0, val_split)]
sample_weights = [class_weights[l] for l in train_labels]
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# Age/Gender dataset optional for multi-task training (you can ignore if not present)
agegender_exists = any(os.path.isdir(root) for root in AGEG_ROOTS)
if agegender_exists:
    age_ds = AgeGenderDataset([r for r in AGEG_ROOTS if os.path.isdir(r)], phase="train")
    age_loader = DataLoader(age_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    print("Age/Gender samples:", len(age_ds))
else:
    age_loader = None
    print("Age/Gender data not found; training emotion-only.")

# -----------------------
# Model / Loss / Opt
# -----------------------
model = MultiTaskCNN(num_emotions=NUM_EMOTIONS).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=4, verbose=True)
ce_loss = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(DEVICE))
mse_loss = nn.MSELoss()
ce_gender = nn.CrossEntropyLoss()

best_val = float("inf")

# -----------------------
# Training loop
# -----------------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} (train)")
    for imgs, labels in pbar:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        out = model(imgs)

        loss_em = ce_loss(out["emotion"], labels)
        loss = loss_em

        # if age/gender available - optional multitask (we simply skip if not)
        if age_loader is not None:
            # sample a batch from age_loader (simple round-robin)
            try:
                age_batch = next(age_iter)
            except:
                age_iter = iter(age_loader)
                age_batch = next(age_iter)
            imgs_a, (age_t, gender_t) = age_batch
            imgs_a = imgs_a.to(DEVICE)
            age_t = age_t.to(DEVICE)
            gender_t = gender_t.to(DEVICE)

            out_a = model(imgs_a)
            loss_age = mse_loss(out_a["age"], age_t)
            loss_gender = ce_gender(out_a["gender"], gender_t)
            # weight tasks (emotion heavier)
            loss = loss_em + 0.2 * loss_age + 0.2 * loss_gender

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{running_loss / (pbar.n+1):.4f}"})

    # -----------------------
    # Validation
    # -----------------------
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs_v, labels_v in val_loader:
            imgs_v = imgs_v.to(DEVICE)
            labels_v = labels_v.to(DEVICE)
            out_v = model(imgs_v)
            l_em = ce_loss(out_v["emotion"], labels_v)
            val_loss += l_em.item() * imgs_v.size(0)

            preds = torch.argmax(out_v["emotion"], dim=1)
            correct += (preds == labels_v).sum().item()
            total += imgs_v.size(0)

    val_loss /= max(1, total)
    val_acc = 100.0 * correct / max(1, total)
    scheduler.step(val_loss)

    print(f"Epoch {epoch}/{EPOCHS} -> TrainLoss: {running_loss/len(train_loader):.4f}  ValLoss: {val_loss:.4f}  ValAcc: {val_acc:.2f}%")

    # save best
    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "multitask_cnn.pth"))
        print("Saved best model ->", os.path.join(MODEL_DIR, "multitask_cnn.pth"))

print("Training finished. Best val loss:", best_val)
