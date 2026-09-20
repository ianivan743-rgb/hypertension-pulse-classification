"""Masked Autoencoder (1D) 自监督预训练:
- 沿时间轴切 patch, 随机遮盖 patch, 让 encoder + 轻量 decoder 重建原始信号.
- 只需要波形, 不需要任何标签; 用你 914 例全量做预训练.
"""
import torch, torch.nn as nn

class MAE1D(nn.Module):
    def __init__(self, in_ch=24, seq_len=800, patch=20, hidden=128,
                 enc_depth=6, dec_depth=2, mask_ratio=0.5, dropout=0.1):
        super().__init__()
        assert seq_len % patch == 0, "seq_len 必须被 patch 整除"
        self.patch = patch
        self.n_patches = seq_len // patch
        self.mask_ratio = mask_ratio

        # patch 嵌入
        self.embed = nn.Conv1d(in_ch, hidden, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.n_patches, hidden))

        enc_layer = nn.TransformerEncoderLayer(hidden, 4, hidden*2,
                                                dropout=dropout, batch_first=True,
                                                activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_depth)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        dec_layer = nn.TransformerEncoderLayer(hidden, 4, hidden*2,
                                                dropout=dropout, batch_first=True,
                                                activation="gelu")
        self.decoder = nn.TransformerEncoder(dec_layer, num_layers=dec_depth)
        self.recon = nn.Linear(hidden, in_ch*patch)

        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)

    def _random_mask(self, B, L, device):
        n_keep = int(L*(1-self.mask_ratio))
        noise = torch.rand(B, L, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :n_keep]
        mask = torch.ones(B, L, device=device); mask[:, :n_keep] = 0
        mask = torch.gather(mask, 1, ids_restore)      # 1 = masked
        return ids_keep, ids_restore, mask

    def forward(self, x):
        # x: (B, C, T)
        B, C, T = x.shape
        h = self.embed(x).transpose(1, 2)              # (B, L, H)
        h = h + self.pos
        L = h.size(1)
        ids_keep, ids_restore, mask = self._random_mask(B, L, x.device)
        h_keep = torch.gather(h, 1, ids_keep.unsqueeze(-1).expand(-1, -1, h.size(-1)))
        z = self.encoder(h_keep)

        # decoder: 插入 mask token 并按原顺序拼回
        n_masked = L - z.size(1)
        mtoks = self.mask_token.expand(B, n_masked, z.size(-1))
        z_full = torch.cat([z, mtoks], dim=1)
        z_full = torch.gather(z_full, 1,
                              ids_restore.unsqueeze(-1).expand(-1, -1, z.size(-1)))
        z_full = z_full + self.pos
        z_dec  = self.decoder(z_full)
        pred = self.recon(z_dec)                       # (B, L, C*patch)
        pred = pred.reshape(B, L, C, self.patch).permute(0, 2, 1, 3).reshape(B, C, T)

        loss = ((pred - x)**2)                          # 只在 masked patch 计算
        m = mask.unsqueeze(1).repeat_interleave(self.patch, dim=1) \
                              .unsqueeze(1).squeeze(1)  # (B, L*patch) -> (B, T)
        m = mask.repeat_interleave(self.patch, dim=1).unsqueeze(1)  # (B,1,T)
        loss = (loss * m).sum() / (m.sum()*C + 1e-8)
        return loss, pred

    @torch.no_grad()
    def encode(self, x):
        h = self.embed(x).transpose(1, 2) + self.pos
        z = self.encoder(h)
        return z.mean(dim=1)
