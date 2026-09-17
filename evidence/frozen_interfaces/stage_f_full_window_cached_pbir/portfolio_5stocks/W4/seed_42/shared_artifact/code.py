import numpy as np

def revise_state(s):
    x = np.asarray(s, dtype=np.float64)
    closes = x[0::6]
    opens = x[1::6]
    highs = x[2::6]
    lows = x[3::6]
    volumes = x[4::6]
    adjusted_closes = x[5::6]
    
    # Log returns for stability
    log_closes = np.log(closes + 1e-8)
    returns = np.diff(log_closes)
    
    # f1: 5-day momentum (sum of last 5 log returns) — strongest volatile-regime IC (+0.3171)
    f1 = float(np.sum(returns[-5:]) if len(returns) >= 5 else 0.0)
    
    # f2: downside deviation (std of negative returns) — strong volatile-regime IC (-0.2608), penalize risk
    neg_returns = returns[returns < 0]
    f2 = float(np.std(neg_returns)) if len(neg_returns) > 1 else 0.0
    
    # f3: 20-day high-low range normalized by close (volatility proxy) — robust across regimes, avoids price-level bias
    hl_range = highs - lows
    f3 = float(np.mean(hl_range / (closes + 1e-8))) if len(closes) > 0 else 0.0
    
    # f4: volume z-score vs 19-day mean — stable volume signal, avoids look-ahead bias
    if len(volumes) > 1:
        vol_mean = float(np.mean(volumes[:-1]))
        vol_std = float(np.std(volumes[:-1])) + 1e-8
        f4 = float((volumes[-1] - vol_mean) / vol_std)
    else:
        f4 = 0.0
    
    extra = np.array([f1, f2, f3, f4], dtype=np.float64)
    extra = np.nan_to_num(extra, nan=0.0, posinf=0.0, neginf=0.0)
    return np.concatenate((x, extra))

def intrinsic_reward(updated_s):
    f1 = float(updated_s[120])
    f2 = float(updated_s[121])
    f3 = float(updated_s[122])
    f4 = float(updated_s[123])
    
    # Phi(s): reward trend (f1), penalize downside risk (f2), penalize high volatility (f3), neutral on volume (f4)
    # Emphasizes volatile-regime alignment: f1 positive weight, f2/f3 negative weights; f4 suppressed due to negative IC
    phi = np.tanh(1.3 * f1 - 0.9 * f2 - 0.7 * np.tanh(f3) + 0.0 * f4)
    
    phi = float(np.nan_to_num(phi, nan=0.0, posinf=0.0, neginf=0.0))
    return phi