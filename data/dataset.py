import numpy as np, torch
from torch.utils.data import Dataset

class PulseDataset(Dataset):
    def __init__(self, X, F=None, M=None, y=None):
        self.X = torch.from_numpy(X.astype(np.float32))  # (N,C,T)
        self.F = torch.from_numpy(F.astype(np.float32)) if F is not None else None
        self.M = torch.from_numpy(M.astype(np.float32)) if M is not None else None
        self.y = torch.from_numpy(y.astype(np.int64))   if y is not None else None
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        item = {"x": self.X[i]}
        if self.F is not None: item["f"] = self.F[i]
        if self.M is not None: item["m"] = self.M[i]
        if self.y is not None: item["y"] = self.y[i]
        return item
