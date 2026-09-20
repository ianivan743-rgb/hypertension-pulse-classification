"""从 24 通道波形计算时域指标 (与论文一致近似)."""
import numpy as np
from scipy.signal import find_peaks

def _peaks_valleys(sig):
    peaks, _ = find_peaks(sig, distance=30)
    return peaks

def one_cycle_features(sig, fs=200):
    peaks = _peaks_valleys(sig)
    if len(peaks) < 2:
        return dict(h1=float(sig.max()), h3=0.0, h4=0.0, h5=0.0,
                    t1=0.0, t4=0.0, t5=0.0, w1_t=0.0, w2_t=0.0)
    p0, p1 = peaks[0], peaks[1]
    seg = sig[p0:p1]
    h1 = float(seg.max())
    idx_h1 = int(np.argmax(seg))
    tail = seg[idx_h1:]
    if len(tail) > 10:
        h3 = float(tail.max())
        h5 = float(tail[-min(5, len(tail)):].mean())
    else:
        h3, h5 = 0.0, 0.0
    h4 = float((h1+h3)/2)
    T = len(seg)/fs
    t1 = idx_h1/fs
    t5 = T
    t4 = t5*0.6
    half = h1*0.5
    above = np.where(seg > half)[0]
    w1_t = (above[-1]-above[0])/fs if len(above) > 0 else 0.0
    w2_t = w1_t*0.7
    return dict(h1=h1, h3=h3, h4=h4, h5=h5, t1=t1, t4=t4, t5=t5,
                w1_t=w1_t, w2_t=w2_t)

def channel_time_features(x_ch, fs=200):
    d = one_cycle_features(x_ch, fs)
    d["h3_h1"] = d["h3"]/max(d["h1"], 1e-6)
    d["h4_h1"] = d["h4"]/max(d["h1"], 1e-6)
    return d

def array_time_features(X, fs=200):
    """X: (C,T) -> best channel (h1 最大) 的时域特征."""
    energies = (X**2).sum(1)
    ch = int(np.argmax(energies))
    return channel_time_features(X[ch], fs)
