"""Students replace Net, preserve the I/O contract, change ARCHITECTURE, and retrain."""
import torch.nn as nn


class Net(nn.Module):
    ARCHITECTURE = 'small-cnn-v1'

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(16 * 16 * 16, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, image):
        return self.classifier(self.features(image))  # Raw logits, order O then X.
