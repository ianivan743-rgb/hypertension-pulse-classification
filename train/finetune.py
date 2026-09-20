"""有标签微调: 从 SSL 预训练的 MAE encoder 加载权重, 加分类头, 5-fold 训练评估."""
import argparse, os, sys, math, copy
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_config, load_dataset
from data.dataset import PulseDataset
from data.augment import apply_aug
from models.ssl_mae import MAE1D
from models.encoder1d import build_encoder, ClsHead
from utils.seed import set_seed, get_device
from utils.cv import outer_split, stratified_folds
from utils.imbalance import compute_class_weights, FocalLoss
from utils.metrics import classification_metrics, print_metrics


class MAEEncoderAdapter(nn.Module):
    """把训好的 MAE 包装成 encoder(x)->(B,H)."""
    def __init__(self, mae):
        super().__init__()
        self.embed = mae.embed
        self.pos   = mae.pos
        self.encoder = mae.encoder
        self.out_dim = mae.embed.out_channels
    def forward(self, x):
        h = self.embed(x).transpose(1, 2) + self.pos
        z = self.encoder(h)
        return z.mean(dim=1)


def build_from_pretrained(cfg, pretrained_path, num_classes):
    if pretrained_path and os.path.exists(pretrained_path):
        state = torch.load(pretrained_path, map_location="cpu")
        mae = MAE1D(in_ch=cfg["data"]["n_channels"],
                    seq_len=cfg["data"]["seq_len"],
                    patch=cfg["model"]["mae_patch_len"],
                    hidden=cfg["model"]["hidden_dim"],
                    enc_depth=cfg["model"]["depth"],
                    mask_ratio=cfg["model"]["mae_mask_ratio"],
                    dropout=cfg["model"]["dropout"])
        mae.load_state_dict(state["model"])
        encoder = MAEEncoderAdapter(mae)
    else:
        print("[warn] 无预训练权重, 使用 from-scratch encoder")
        encoder = build_encoder(cfg["model"]["encoder"],
                                cfg["data"]["n_channels"],
                                cfg["model"]["hidden_dim"],
                                cfg["model"]["depth"],
                                cfg["model"]["dropout"])
    head = ClsHead(encoder.out_dim, num_classes, dropout=cfg["model"]["dropout"])
    return encoder, head


def train_one_fold(cfg, X, y, tr, va, num_classes, pretrained, device):
    encoder, head = build_from_pretrained(cfg, pretrained, num_classes)
    model = nn.Sequential()
    model.add_module("enc", encoder); model.add_module("head", head)
    model = model.to(device)

    ds_tr = PulseDataset(X[tr], y=y[tr])
    ds_va = PulseDataset(X[va], y=y[va])
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       num_workers=cfg["train"]["num_workers"], drop_last=False)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"])

    weight = compute_class_weights(y[tr], num_classes).to(device) \
             if cfg["imbalance"]["strategy"] in ("class_weight","focal") else None
    if cfg["imbalance"]["strategy"] == "focal":
        crit = FocalLoss(gamma=cfg["imbalance"]["focal_gamma"], weight=weight)
    else:
        crit = nn.CrossEntropyLoss(weight=weight)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    best = {"f1": -1, "state": None, "epoch": 0}
    epochs = cfg["train"]["epochs_ft"]; warm = cfg["train"]["warmup_epochs"]
    patience = cfg["train"]["early_stop_patience"]; bad = 0
    for ep in range(epochs):
        if ep < warm: lr = cfg["train"]["lr"]*(ep+1)/warm
        else:
            t = (ep-warm)/max(1, epochs-warm)
            lr = 0.5*cfg["train"]["lr"]*(1+math.cos(math.pi*t))
        for g in opt.param_groups: g["lr"] = lr

        model.train()
        for batch in dl_tr:
            x = batch["x"].to(device); yy = batch["y"].to(device)
            x = apply_aug(x, train=True)
            z = model.enc(x); logit = model.head(z)
            loss = crit(logit, yy)
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval(); ys, ps, probs = [], [], []
        with torch.no_grad():
            for batch in dl_va:
                x = batch["x"].to(device); yy = batch["y"].to(device)
                logit = model.head(model.enc(x))
                prob  = torch.softmax(logit, dim=-1)
                ps.append(prob.argmax(1).cpu().numpy())
                probs.append(prob.cpu().numpy()); ys.append(yy.cpu().numpy())
        yt = np.concatenate(ys); yp = np.concatenate(ps); pr = np.concatenate(probs)
        m = classification_metrics(yt, yp, pr)
        print(f"  ep{ep+1:03d} lr={lr:.2e} valF1={m['f1']:.4f} valAcc={m['accuracy']:.4f}")
        if m["f1"] > best["f1"]:
            best = {"f1": m["f1"], "state": copy.deepcopy(model.state_dict()),
                    "epoch": ep, "metric": m}
            bad = 0
        else:
            bad += 1
            if bad >= patience: print("  early stop"); break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--task", choices=["binary","grade","syndrome"], default="syndrome")
    ap.add_argument("--pretrained", default="")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    device = get_device(cfg["device"])

    b = load_dataset(cfg)
    groups = b["groups"]
    X = b["X"]
    mu = X.mean(-1, keepdims=True); sd = X.std(-1, keepdims=True)+1e-6
    X = ((X-mu)/sd).astype(np.float32)
    y = b["y"][args.task]
    num_classes = int(y.max())+1

    tr_idx, te_idx = outer_split(y, cfg["cv"]["test_ratio"], cfg["seed"], groups=groups)
    fold_metrics = []
    for k, (a, v) in enumerate(stratified_folds(y[tr_idx], cfg["cv"]["n_folds"], cfg["seed"], groups=groups[tr_idx])):
        print(f"\n=== fold {k+1}/{cfg['cv']['n_folds']} ===")
        best = train_one_fold(cfg, X[tr_idx], y[tr_idx], a, v,
                              num_classes, args.pretrained, device)
        fold_metrics.append(best["metric"])
    macro = {k: float(np.mean([m[k] for m in fold_metrics]))
             for k in ["accuracy","precision","recall","f1"]}
    print("\n[CV macro]", macro)

if __name__ == "__main__":
    main()
