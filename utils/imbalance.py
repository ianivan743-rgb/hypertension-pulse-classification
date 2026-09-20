import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def compute_class_weights(y, num_classes=None):
    y = np.asarray(y)
    C = int(y.max())+1 if num_classes is None else num_classes
    counts = np.bincount(y, minlength=C).astype(np.float64)
    counts[counts == 0] = 1.0
    w = counts.sum() / (C * counts)
    return torch.tensor(w, dtype=torch.float32)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1-pt)**self.gamma * ce).mean()
