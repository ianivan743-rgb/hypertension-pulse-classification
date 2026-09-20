"""SSL 预训练 (MAE): 用全部 914 例波形, 不需要标签.
产出 outputs/models/ssl.pt, 供 finetune / fusion 加载."""
import argparse, os, sys, math
import numpy as np, torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_config, load_dataset
from data.dataset import PulseDataset
from data.augment import apply_aug
from models.ssl_mae import MAE1D
from utils.seed import set_seed, get_device

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    device = get_device(cfg["device"]); print("device:", device)

    b = load_dataset(cfg)
    X = b["X"]                                         # (N,C,T)
    # 波形逐样本 z-score
    mu = X.mean(-1, keepdims=True); sd = X.std(-1, keepdims=True)+1e-6
    X = (X-mu)/sd
    ds = PulseDataset(X)
    dl = DataLoader(ds, batch_size=cfg["train"]["batch_size"], shuffle=True,
                    num_workers=cfg["train"]["num_workers"], drop_last=True)

    model = MAE1D(in_ch=cfg["data"]["n_channels"],
                  seq_len=cfg["data"]["seq_len"],
                  patch=cfg["model"]["mae_patch_len"],
                  hidden=cfg["model"]["hidden_dim"],
                  enc_depth=cfg["model"]["depth"],
                  mask_ratio=cfg["model"]["mae_mask_ratio"],
                  dropout=cfg["model"]["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                            weight_decay=cfg["train"]["weight_decay"])
    epochs = cfg["train"]["epochs_ssl"]; warm = cfg["train"]["warmup_epochs"]

    for ep in range(epochs):
        # cosine + warmup
        if ep < warm: lr = cfg["train"]["lr"]*(ep+1)/warm
        else:
            t = (ep-warm)/max(1, epochs-warm)
            lr = 0.5*cfg["train"]["lr"]*(1+math.cos(math.pi*t))
        for g in opt.param_groups: g["lr"] = lr

        model.train(); losses = []
        for batch in dl:
            x = batch["x"].to(device)
            x = apply_aug(x, train=True)
            loss, _ = model(x)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        print(f"[SSL] epoch {ep+1}/{epochs}  lr={lr:.2e}  loss={np.mean(losses):.4f}")

    out_dir = cfg["output"]["models"]; os.makedirs(out_dir, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "cfg": cfg}, os.path.join(out_dir, "ssl.pt"))
    print("saved:", os.path.join(out_dir, "ssl.pt"))

if __name__ == "__main__":
    main()
