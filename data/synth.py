"""合成 demo 数据: 24 通道脉搏波 + 手工特征 + 三种标签, 用于 pipeline 冒烟测试."""
import numpy as np, pandas as pd, os

def synth_wave(n_ch=24, seq_len=800, hr_bpm=72, fs=200, hyper=False, rng=None):
    rng = rng or np.random.default_rng()
    t = np.arange(seq_len)/fs
    period = 60.0/hr_bpm
    phi = rng.uniform(0, 2*np.pi)
    # 主波+重搏前波+重搏波 三高斯 + 通道空间衰减
    x = np.zeros((n_ch, seq_len))
    for ch in range(n_ch):
        atten = 0.7 + 0.6*np.exp(-((ch-11)/6.0)**2)
        h1 = (1.4 if hyper else 1.0) * atten * (1+0.05*rng.standard_normal())
        h3 = (0.6 if hyper else 0.4) * atten
        h5 = 0.25 * atten
        cycles = np.floor(seq_len/(fs*period)).astype(int) + 1
        for k in range(cycles):
            c0 = k*period + phi/(2*np.pi)*period
            x[ch] += h1*np.exp(-((t-c0-0.08)/0.03)**2)
            x[ch] += h3*np.exp(-((t-c0-0.22)/0.05)**2)
            x[ch] += h5*np.exp(-((t-c0-0.42)/0.06)**2)
        x[ch] += 0.02*rng.standard_normal(seq_len)
    return x.astype(np.float32)

def make_dataset(n=200, n_ch=24, seq_len=800, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        hyper = rng.random() < 0.72   # 高血压占比接近 659/914
        grade = rng.choice([1,2,3,4], p=[0.35,0.30,0.15,0.20]) if hyper else 0
        syn   = rng.choice([0,1,2,3], p=[0.32,0.30,0.34,0.04]) if hyper else 0
        age = rng.integers(30, 80)
        sbp = rng.integers(140, 200) if hyper else rng.integers(100, 135)
        dbp = rng.integers(85, 120) if hyper else rng.integers(60, 84)
        wave = synth_wave(n_ch, seq_len, hyper=hyper, rng=rng)
        # 手工特征(粗略近似, 只求 pipeline 能跑)
        h1 = float(wave.max(axis=1).mean())
        h3 = float(np.percentile(wave, 90))
        h5 = float(np.percentile(wave, 70))
        row = {
            "sample_id": f"S{i:04d}",
            "label_hypertension": int(hyper),
            "bp_grade": int(grade),
            "tcm_syndrome": int(syn),
            "age": int(age), "sex": int(rng.integers(0,2)),
            "sbp": int(sbp), "dbp": int(dbp), "bmi": float(rng.normal(24,3)),
            "h1": h1, "h2": h1*0.6, "h3": h3, "h4": h3*0.8, "h5": h5,
            "t1": 0.08, "t2": 0.15, "t3": 0.22, "t4": 0.35, "t5": 0.60,
            "w1_t": 0.03, "w2_t": 0.05, "h3_h1": h3/max(h1,1e-6), "h4_h1": (h3*0.8)/max(h1,1e-6),
            "E": float((wave**2).sum()),
            "TSEn": float(np.log(1+np.abs(wave).sum())),
            "ApEn": float(np.random.rand()*0.5+0.5),
            "sAPVt1": h1*0.9, "sAPVt4": h1*1.2, "sAPVt5": h1*1.3,
            "APVh1": h1*24, "APVh3": h3*24,
        }
        # 展开波形列 wave_<ch>_<t>
        for ch in range(n_ch):
            for tt in range(seq_len):
                row[f"wave_{ch}_{tt}"] = float(wave[ch, tt])
        rows.append(row)
    return pd.DataFrame(rows)

def save_synth_csv(path, n=200, n_ch=24, seq_len=800, seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = make_dataset(n=n, n_ch=n_ch, seq_len=seq_len, seed=seed)
    df.to_csv(path, index=False)
    return path
