"""多模态融合: 深度分支(SSL/微调 encoder) + 手工特征分支 + 基础信息分支 -> 分类头."""
import torch, torch.nn as nn
from models.encoder1d import build_encoder

class MLP(nn.Module):
    def __init__(self, d_in, d_out, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        mods = []
        prev = d_in
        for _ in range(layers):
            mods += [nn.Linear(prev, hidden), nn.GELU(), nn.Dropout(dropout)]
            prev = hidden
        mods += [nn.Linear(prev, d_out)]
        self.net = nn.Sequential(*mods)
    def forward(self, x): return self.net(x)

class FusionModel(nn.Module):
    def __init__(self, encoder, feat_dim, meta_dim, num_classes,
                 hidden=128, dropout=0.2):
        super().__init__()
        self.encoder = encoder
        enc_dim = encoder.out_dim
        self.feat_mlp = MLP(feat_dim, hidden, hidden, layers=2, dropout=dropout)
        self.meta_mlp = MLP(meta_dim, hidden, hidden, layers=2, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(enc_dim+hidden+hidden, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes))
    def forward(self, x, f, m):
        z = self.encoder(x)
        zf = self.feat_mlp(f)
        zm = self.meta_mlp(m)
        return self.head(torch.cat([z, zf, zm], dim=-1))
