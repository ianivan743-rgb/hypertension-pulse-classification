"""多模态融合训练: 深度分支(可加载 SSL 权重) + 手工特征 MLP + 基础信息 MLP -> 分类头."""
import argparse, os, sys, math, copy
import numpy as np, torch, torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.loader import load_config, load_dataset
from data.dataset import PulseDataset
from data.augment import apply_aug
from models.ssl_mae import MAE1D
from models.encoder1d import build_encoder
from models.fusion import FusionModel
from utils.seed import set_seed, get_device
from utils.cv import outer_split, stratified_folds
from utils.imbalance import compute_class_weights, FocalLoss
from utils.metrics import classification_metrics
from train.finetune import MAEEncoderAdapter


def make_encoder(cfg, pretrained):
    if pretrained and os.path.exists(pretrained):
        state = torch.load(pretrained, map_location="cpu")
        mae = MAE1D(in_ch=cfg["data"]["n_channels"],
                    seq_len=cfg["data"]["seq_len"],
                    patch=cfg["model"]["mae_patch_len"],
                    hidden=cfg["model"]["hidden_dim"],
                    enc_depth=cfg["model"]["depth"],
                    mask_ratio=cfg["model"]["mae_mask_ratio"],
                    dropout=cfg["model"]["dropout"])
        mae.load_state_dict(state["model"])
        return MAEEncoderAdapter(mae)
    return build_encoder(cfg["model"]["encoder"], cfg["data"]["n_channels"],
                         cfg["model"]["hidden_dim"], cfg["model"]["depth"],
                         cfg["model"]["dropout"])


def run(cfg, task, pretrained):
    device = get_device(cfg["device"])
    b = load_dataset(cfg)
    groups = b["groups"]
    X = b["X"].astype(np.float32)
    mu = X.mean(-1, keepdims=True); sd = X.std(-1, keepdims=True)+1e-6
    X = (X-mu)/sd
    F = b["F"].astype(np.float32)
    M = b["M"].astype(np.float32)
    # 全局标准化 F, M
    F = (F - F.mean(0)) / (F.std(0)+1e-6)
    M = (M - M.mean(0)) / (M.std(0)+1e-6)
    y = b["y"][task]; num_classes = int(y.max())+1
    tr, te = outer_split(y, cfg["cv"]["test_ratio"], cfg["seed"], groups=groups)

    metrics = []
    for k, (a, v) in enumerate(stratified_folds(y[tr], cfg["cv"]["n_folds"], cfg["seed"], groups=groups[tr])):
        encoder = make_encoder(cfg, pretrained)
        model = FusionModel(encoder, F.shape[1], M.shape[1], num_classes,
                             hidden=cfg["model"]["fusion_mlp_hidden"],
                             dropout=cfg["model"]["dropout"]).to(device)
        ds_tr = PulseDataset(X[tr][a], F=F[tr][a], M=M[tr][a], y=y[tr][a])
        ds_va = PulseDataset(X[tr][v], F=F[tr][v], M=M[tr][v], y=y[tr][v])
        dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                            num_workers=cfg["train"]["num_workers"])
        dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"])

        weight = compute_class_weights(y[tr][a], num_classes).to(device) \
                 if cfg["imbalance"]["strategy"] in ("class_weight","focal") else None
        crit = FocalLoss(cfg["imbalance"]["focal_gamma"], weight) \
               if cfg["imbalance"]["strategy"]=="focal" else nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                weight_decay=cfg["train"]["weight_decay"])
        best = {"f1":-1, "metric":None}
        epochs = cfg["train"]["epochs_ft"]; warm = cfg["train"]["warmup_epochs"]
        patience = cfg["train"]["early_stop_patience"]; bad = 0
        for ep in range(epochs):
            if ep < warm: lr = cfg["train"]["lr"]*(ep+1)/warm
            else:
                t = (ep-warm)/max(1,epochs-warm)
                lr = 0.5*cfg["train"]["lr"]*(1+math.cos(math.pi*t))
            for g in opt.param_groups: g["lr"]=lr

            model.train()
            for batch in dl_tr:
                x = apply_aug(batch["x"].to(device), True)
                f = batch["f"].to(device); m = batch["m"].to(device)
                yy = batch["y"].to(device)
                logit = model(x, f, m)
                loss = crit(logit, yy)
                opt.zero_grad(); loss.backward(); opt.step()

            model.eval(); ys, ps, pr = [], [], []
            with torch.no_grad():
                for batch in dl_va:
                    x = batch["x"].to(device); f = batch["f"].to(device); m = batch["m"].to(device)
                    logit = model(x, f, m)
                    prob = torch.softmax(logit, -1)
                    ps.append(prob.argmax(1).cpu().numpy()); pr.append(prob.cpu().numpy())
                    ys.append(batch["y"].numpy())
            m_ep = classification_metrics(np.concatenate(ys),
                                          np.concatenate(ps),
                                          np.concatenate(pr))
            if m_ep["f1"] > best["f1"]:
                best = {"f1":m_ep["f1"], "metric":m_ep}; bad = 0
            else:
                bad += 1
                if bad >= patience: break
        print(f"[fold {k+1}] best F1={best['f1']:.4f}")
        metrics.append(best["metric"])
    print("\n[CV] macro F1 =", float(np.mean([m["f1"] for m in metrics])))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--task", choices=["binary","grade","syndrome"], default="syndrome")
    ap.add_argument("--pretrained", default="")
    args = ap.parse_args()
    cfg = load_config(args.config); set_seed(cfg["seed"])
    run(cfg, args.task, args.pretrained)
