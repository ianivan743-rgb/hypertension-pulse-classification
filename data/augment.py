import numpy as np, torch

def jitter(x, sigma=0.03):
    return x + sigma*torch.randn_like(x)

def scaling(x, sigma=0.1):
    factor = 1 + sigma*torch.randn(x.size(0), x.size(1), 1, device=x.device)
    return x * factor

def time_warp(x, sigma=0.1):
    # 简易实现: 沿时间轴按随机偏移做小幅重采样
    B, C, T = x.shape
    steps = torch.linspace(0, T-1, T, device=x.device)
    shift = sigma*torch.randn(B, 1, device=x.device)*T*0.02
    idx = torch.clamp((steps.unsqueeze(0)+shift).long(), 0, T-1)
    idx = idx.unsqueeze(1).expand(B, C, T)
    return torch.gather(x, 2, idx)

def cutout(x, p=0.2, max_len=40):
    B, C, T = x.shape
    if torch.rand(1).item() > p: return x
    L = torch.randint(5, max_len, (1,)).item()
    s = torch.randint(0, T-L, (1,)).item()
    x2 = x.clone(); x2[:, :, s:s+L] = 0
    return x2

def apply_aug(x, train=True):
    if not train: return x
    x = jitter(x); x = scaling(x); x = cutout(x)
    return x
