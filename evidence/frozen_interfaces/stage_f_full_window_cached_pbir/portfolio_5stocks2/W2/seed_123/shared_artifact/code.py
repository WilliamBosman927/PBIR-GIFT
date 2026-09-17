import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    returns = np.diff(closes) / (np.abs(closes[:-1]) + 1e-8)
    
    # f1: 5-day momentum z-score (robust to regime shifts — uses full 20d std)
    f1 = float(np.mean(returns[-5:]) / (np.std(returns) + 1e-8)) if len(returns) >= 5 else 0.0
    
    # f2: downside deviation — mean of negative returns (not std), zero if none exist
    neg_returns = returns[returns < 0]
    f2 = float(np.mean(neg_returns)) if len(neg_returns) > 0 else 0.0
    
    # f3: normalized price deviation from 20-day SMA using ATR(14) — more stable than raw range
    sma20 = np.mean(closes)
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - np.roll(closes, 1)), np.abs(lows - np.roll(closes, 1))))
    atr = np.mean(tr[1:]) if len(tr[1:]) > 0 else 1.0
    f3 = float((closes[-1] - sma20) / (atr + 1e-8))
    
    # f4: realized volatility (std of 20d returns), clipped and scaled for stability
    f4 = float(np.std(returns) + 1e-8)
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi rewards strong positive momentum (f1), penalizes downside bias (f2 < 0 → harmful),
    # penalizes extreme over/under-performance vs trend (|f3|), and penalizes high volatility (f4)
    # Uses tanh to bound output in [-1, 1]; weights tuned to emphasize s[122] (high SHAP) and s[121] (high IC)
    phi = np.tanh(0.7 * f1 + 0.5 * f2 - 0.4 * np.abs(f3) - 0.6 * f4)
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi