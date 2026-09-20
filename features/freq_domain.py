"""频域特征: 总能量 E, 总频谱能量 TSEn, 近似熵 ApEn, 主频等."""
import numpy as np
from scipy.signal import welch

def approximate_entropy(u, m=2, r=None):
    u = np.asarray(u, dtype=float); N = len(u)
    if N < m+2: return 0.0
    if r is None: r = 0.2*np.std(u)
    def _phi(m):
        x = np.array([u[i:i+m] for i in range(N-m+1)])
        C = np.sum(np.max(np.abs(x[:,None]-x[None,:]), axis=2) <= r, axis=0)/(N-m+1)
        return np.sum(np.log(np.clip(C, 1e-12, None)))/(N-m+1)
    return float(_phi(m)-_phi(m+1))

def channel_freq_features(sig, fs=200):
    f, Pxx = welch(sig, fs=fs, nperseg=min(256, len(sig)))
    E = float((sig**2).sum())
    TSEn = float(Pxx.sum())
    main_freq = float(f[np.argmax(Pxx)])
    ApEn = approximate_entropy(sig[:min(400, len(sig))])
    return dict(E=E, TSEn=TSEn, ApEn=ApEn, main_freq=main_freq)

def array_freq_features(X, fs=200):
    energies = (X**2).sum(1)
    ch = int(np.argmax(energies))
    return channel_freq_features(X[ch], fs)
