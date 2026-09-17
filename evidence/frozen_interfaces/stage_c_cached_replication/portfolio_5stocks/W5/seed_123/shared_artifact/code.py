import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # 20-day returns (19 values)
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    
    # f1: 5-day momentum (last close vs 5-day prior close)
    f1 = (closes[-1] - closes[-5]) / (np.abs(closes[-5]) + 1e-8) if len(closes) >= 5 else 0.0
    
    # f2: realized downside risk (std of negative returns only; fallback to full std if no negatives)
    negative_returns = returns[returns < 0]
    f2 = float(np.std(negative_returns)) + 1e-8 if len(negative_returns) > 1 else float(np.std(returns)) + 1e-8
    
    # f3: z-score of current price relative to 20-day mean/std (using adjusted_close for robustness)
    price_mean = float(np.mean(adjusted_closes))
    price_std = float(np.std(adjusted_closes)) + 1e-8
    f3 = (adjusted_closes[-1] - price_mean) / price_std
    
    # f4: normalized volume acceleration: (recent vol avg - prior vol avg) / (prior vol std + eps)
    vol_recent = np.mean(volumes[-5:]) if len(volumes) >= 5 else 0.0
    vol_prior = np.mean(volumes[-15:-5]) if len(volumes) >= 15 else np.mean(volumes)
    vol_prior_std = np.std(volumes[-15:-5]) + 1e-8 if len(volumes) >= 15 else np.std(volumes) + 1e-8
    f4 = (vol_recent - vol_prior) / vol_prior_std if vol_prior_std > 1e-8 else 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi: reward positive momentum (f1), penalize downside risk (f2), reward mean-reversion (|f3| near zero), neutral on volume acceleration (f4)
    # Defensive regime guidance: prioritize low downside risk and moderate price deviation
    phi = np.tanh(1.0 * f1 - 0.8 * f2 - 0.6 * np.abs(f3))
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi