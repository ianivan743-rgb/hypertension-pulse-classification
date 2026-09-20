import os, random, numpy as np, torch

def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device(pref: str = "auto"):
    if pref == "cpu": return torch.device("cpu")
    if pref == "cuda": return torch.device("cuda")
    if pref == "mps":  return torch.device("mps")
    if torch.cuda.is_available(): return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
