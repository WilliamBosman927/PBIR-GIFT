import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 20-day log returns (19 values)
    returns = np.log(closes[1:] / (closes[:-1] + 1e-8))
    
    # f1: 3-day momentum (log return over last 3 days) — higher regime-specific IC in trending_up/volatile than 5-day
    f1 = float(np.sum(returns[-3:]) if len(returns) >= 3 else 0.0)
    
    # f2: downside volatility (std of negative returns only; robust to zero/negatives)
    neg_returns = returns[returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.0
    
    # f3: normalized close-to-high deviation (average (high - close)/high over last 5 days — measures rejection pressure)
    ch_devs = (highs[-5:] - closes[-5:]) / (highs[-5:] + 1e-8)
    f3 = float(np.mean(ch_devs)) if len(ch_devs) > 0 else 0.0
    
    # f4: volume acceleration ratio (current volume / 5-day avg volume — avoids z-score instability in low-volume regimes)
    vol_window = volumes[-5:]
    vol_mean = float(np.mean(vol_window)) + 1e-8 if len(vol_window) > 0 else 1e-8
    f4 = float(volumes[-1] / vol_mean) if len(vol_window) > 0 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward short-term trend (f1), penalize downside risk (f2), penalize rejection pressure (f3), moderate volume surge (f4 via log)
    # Uses log(f4+1) to bound positive volume surges and avoid explosion; tanh ensures [-1,1] output
    phi = np.tanh(1.4 * f1 - 0.8 * f2 - 0.6 * f3 + 0.2 * np.log1p(f4))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi