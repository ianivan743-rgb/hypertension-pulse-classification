"""阵列脉图容积 (Array Pulse Volume) 指标."""
import numpy as np

def apv_features(X, fs=200):
    """X: (C,T). APV = 各通道波形对时间和空间的积分."""
    C, T = X.shape
    apv = X.sum(0)                                  # (T,) 空间求和
    peak = apv.argmax()
    sAPVt1 = float(apv[max(peak-int(0.05*fs),0)])
    sAPVt4 = float(apv[min(peak+int(0.10*fs),T-1)])
    sAPVt5 = float(apv[min(peak+int(0.30*fs),T-1)])
    APVh1  = float(apv.max())
    APVh3  = float(np.percentile(apv, 90))
    return dict(sAPVt1=sAPVt1, sAPVt4=sAPVt4, sAPVt5=sAPVt5,
                APVh1=APVh1, APVh3=APVh3)

def extract_all_hand_features(X, fs=200):
    """整合 time + freq + APV, 返回 dict 与 22 维向量."""
    from features.time_domain import array_time_features
    from features.freq_domain import array_freq_features
    d = {}
    d.update(array_time_features(X, fs))
    d.update(array_freq_features(X, fs))
    d.update(apv_features(X, fs))
    order = ["h1","h3","h4","h5","t1","t4","t5","w1_t","w2_t","h3_h1","h4_h1",
             "E","TSEn","ApEn",
             "sAPVt1","sAPVt4","sAPVt5","APVh1","APVh3"]
    vec = np.array([d.get(k, 0.0) for k in order], dtype=np.float32)
    return d, vec
