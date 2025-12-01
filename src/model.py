# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiTaskCNN(nn.Module):
    def __init__(self, num_emotions=7):
        super(MultiTaskCNN, self).__init__()

        # Feature extractor (bigger & better)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # Flatten: 128 → 64 → 32 → 16 → 8
        self.fc_shared = nn.Sequential(
            nn.Linear(256 * 8 * 8, 1024),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Heads
        self.emotion_head = nn.Linear(512, num_emotions)
        self.age_head = nn.Linear(512, 1)
        self.gender_head = nn.Linear(512, 2)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        x = self.fc_shared(x)

        emotion_logits = self.emotion_head(x)
        age = self.age_head(x).squeeze(1)
        gender_logits = self.gender_head(x)

        return {
            "emotion": emotion_logits,
            "age": age,
            "gender": gender_logits
        }

if __name__ == "__main__":
    test = MultiTaskCNN()
    dummy = torch.randn(2, 3, 128, 128)
    out = test(dummy)
    print({k: v.shape for k, v in out.items()})
