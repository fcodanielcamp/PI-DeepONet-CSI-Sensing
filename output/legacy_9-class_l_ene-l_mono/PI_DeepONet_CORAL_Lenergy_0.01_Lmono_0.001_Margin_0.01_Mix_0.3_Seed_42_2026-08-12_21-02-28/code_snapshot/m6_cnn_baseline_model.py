import torch
import torch.nn as nn
import torch.nn.functional as F

class PureCNN2DBaseline(nn.Module):
    """
    Arquitectura Baseline Conv2D pura (3 capas convolucionales)
    Recibe tensor (Batch, 2, 25, 241) -> Produce Logits de Clasificación (Batch, num_classes)
    """
    def __init__(self, num_classes=9):
        super(PureCNN2DBaseline, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=16, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 5), padding=(1, 2))
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.act = nn.Mish()

        # Dimensión tras convs y pooling: (25, 241) -> Pool1: (12, 120) -> Pool2: (6, 60) -> Pool3: (3, 30)
        self.classifier = nn.Sequential(
            nn.Linear(64 * 3 * 30, 128),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.act(self.bn1(self.conv1(x))))
        x = self.pool(self.act(self.bn2(self.conv2(x))))
        x = self.pool(self.act(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        logits = self.classifier(x)
        return logits
