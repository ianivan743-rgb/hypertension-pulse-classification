"""1D 编码器: ResNet1D 与轻量 Transformer1D. 输入 (B,C,T), 输出 (B,H)."""
import torch, torch.nn as nn

class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 7, stride=stride, padding=3, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 7, stride=1, padding=3, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)
        self.short = (nn.Identity() if in_ch == out_ch and stride == 1
                      else nn.Sequential(
                          nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                          nn.BatchNorm1d(out_ch)))
        self.act = nn.GELU()
    def forward(self, x):
        r = self.short(x)
        x = self.act(self.bn1(self.conv1(x)))
        x = self.drop(self.bn2(self.conv2(x)))
        return self.act(x + r)

class ResNet1D(nn.Module):
    def __init__(self, in_ch=24, hidden=128, depth=6, dropout=0.2):
        super().__init__()
        widths = [max(32, hidden//4), hidden//2, hidden, hidden]
        blocks = []
        c_prev = in_ch
        strides = [2] + [2]*(depth-1)
        for i in range(depth):
            c_out = widths[min(i, len(widths)-1)]
            blocks.append(BasicBlock1D(c_prev, c_out, stride=strides[i], dropout=dropout))
            c_prev = c_out
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.out_dim = c_prev
    def forward(self, x):
        return self.pool(self.net(x)).squeeze(-1)

class Transformer1D(nn.Module):
    def __init__(self, in_ch=24, hidden=128, depth=4, dropout=0.2, patch=20):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv1d(in_ch, hidden, kernel_size=patch, stride=patch)
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                                            dim_feedforward=hidden*2,
                                            dropout=dropout, batch_first=True,
                                            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden))
        self.out_dim = hidden
    def forward(self, x):
        h = self.proj(x).transpose(1, 2)                # (B, L, H)
        cls = self.cls.expand(h.size(0), -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.enc(h)
        return h[:, 0]

def build_encoder(name, in_ch, hidden, depth, dropout, patch=20):
    if name == "resnet1d":
        return ResNet1D(in_ch, hidden, depth, dropout)
    if name == "transformer1d":
        return Transformer1D(in_ch, hidden, depth, dropout, patch)
    raise ValueError(name)

class ClsHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden=None, dropout=0.2):
        super().__init__()
        h = hidden or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, h), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(h, num_classes))
    def forward(self, x): return self.net(x)
