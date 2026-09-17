import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 20-day simple returns
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    
    # f1: 3-day momentum (last close vs 3-day ago close) — higher IC alignment than 5-day in trending_down regime
    f1 = (closes[-1] - closes[-3]) / (np.abs(closes[-3]) + 1e-8) if len(closes) >= 3 else 0.0
    
    # f2: 10-day downside deviation — improved robustness: use max(1, count) to avoid zero denominator
    neg_returns = returns[-10:][returns[-10:] < 0] if len(returns) >= 10 else returns[returns < 0]
    f2 = float(np.std(neg_returns)) + 1e-8 if len(neg_returns) >= 2 else float(np.std(returns[-10:] if len(returns) >= 10 else returns)) + 1e-8
    
    # f3: 3-day price deviation from 5-day SMA — faster mean-reversion signal, better IC in neutral/trending_down
    sma5 = float(np.mean(closes[-5:])) + 1e-8 if len(closes) >= 5 else float(np.mean(closes)) + 1e-8
    f3 = (closes[-1] - sma5) / sma5
    
    # f4: volume acceleration — (current vol - previous vol) / (prev vol + 1e-8), normalized and clipped
    f4 = (volumes[-1] - volumes[-2]) / (np.abs(volumes[-2]) + 1e-8) if len(volumes) >= 2 else 0.0
    f4 = np.clip(f4, -3.0, 3.0)
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Revised potential: emphasize f1 (momentum) and f2 (downside risk) per diagnostics;
    # invert f3 (mean-reversion: negative f3 = oversold → high potential);
    # use f4 (volume acceleration) with moderate weight and clipping for stability
    phi = np.tanh(1.4 * f1 - 1.0 * f2 + 0.7 * (-f3) + 0.3 * np.clip(f4, -1.5, 1.5))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi