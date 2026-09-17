"""Mathematical and configuration tests for PBIR-GIFT."""

import pickle
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from portfolio_env import PortfolioEnv
from prompts import build_init_prompt


def _write_market(path: Path, n_days: int = 25) -> None:
    market = {}
    start = date(2020, 1, 1)
    for idx in range(n_days):
        day = (start + timedelta(days=idx)).isoformat()
        price = 100.0 + idx
        market[day] = {
            'price': {
                'AAA': {
                    'close': price,
                    'open': price,
                    'high': price,
                    'low': price,
                    'volume': 1_000.0,
                    'adjusted_close': price,
                }
            }
        }
    with path.open('wb') as handle:
        pickle.dump(market, handle)


def _config(enabled=True, gamma=0.9, configured_gamma=None,
            terminal_zero=True):
    pbir = {
        'enabled': enabled,
        'scale': 1.0,
        'potential_clip': 10.0,
        'shaping_clip': 0.0,
        'terminal_zero': terminal_zero,
    }
    if configured_gamma is not None:
        pbir['gamma'] = configured_gamma
    return {
        'data': {'tickers': ['AAA']},
        'ppo': {'gamma': gamma},
        'portfolio': {'default_lambda': 0.0},
        'pbir': pbir,
    }


def _identity(state):
    return state


def _price_potential(state):
    # At reset and later, the final close in the 20-day window is index 114.
    return float(state[114] / 100.0)


def _env(tmp_path, enabled=True, gamma=0.9):
    data_path = tmp_path / 'market.pkl'
    _write_market(data_path)
    return PortfolioEnv(
        str(data_path),
        _config(enabled=enabled, gamma=gamma),
        revise_state_fn=_identity,
        intrinsic_reward_fn=_price_potential,
    )


def test_pbir_uses_potential_difference(tmp_path):
    env = _env(tmp_path)
    env.reset()
    _, _, done, info = env.step(np.array([0.5, 0.5]))

    assert not done
    assert info['potential_current'] == pytest.approx(1.20)
    assert info['potential_next'] == pytest.approx(1.21)
    assert info['intrinsic_reward'] == pytest.approx(0.9 * 1.21 - 1.20)


def test_terminal_potential_is_zero_and_discounted_sum_telescopes(tmp_path):
    gamma = 0.9
    env = _env(tmp_path, gamma=gamma)
    env.reset()
    shaping = []
    terminal_info = None
    done = False
    while not done:
        _, _, done, terminal_info = env.step(np.array([0.5, 0.5]))
        shaping.append(terminal_info['intrinsic_reward'])

    assert terminal_info['potential_next'] == 0.0
    assert terminal_info['raw_potential_next'] == pytest.approx(1.24)
    assert terminal_info['intrinsic_reward'] == pytest.approx(-1.23)
    discounted = sum((gamma ** idx) * value
                     for idx, value in enumerate(shaping))
    assert discounted == pytest.approx(-1.20)


def test_terminal_zero_ablation_retains_raw_terminal_potential(tmp_path):
    data_path = tmp_path / 'market.pkl'
    _write_market(data_path)
    env = PortfolioEnv(
        str(data_path),
        _config(enabled=True, gamma=0.9, terminal_zero=False),
        revise_state_fn=_identity,
        intrinsic_reward_fn=_price_potential,
    )
    env.reset()
    done = False
    terminal_info = None
    while not done:
        _, _, done, terminal_info = env.step(np.array([0.5, 0.5]))
    assert terminal_info['potential_next'] == pytest.approx(1.24)
    assert terminal_info['pbir_terminal_zero'] is False


def test_disabled_pbir_reproduces_pure_gift_direct_signal(tmp_path):
    env = _env(tmp_path, enabled=False)
    env.reset()
    _, _, _, info = env.step(np.array([0.5, 0.5]))
    assert info['intrinsic_reward'] == pytest.approx(1.21)


def test_pbir_gamma_cannot_diverge_from_ppo_gamma(tmp_path):
    data_path = tmp_path / 'market.pkl'
    _write_market(data_path)
    with pytest.raises(ValueError, match='PBIR gamma must equal ppo.gamma'):
        PortfolioEnv(
            str(data_path),
            _config(enabled=True, gamma=0.99, configured_gamma=0.90),
            revise_state_fn=_identity,
            intrinsic_reward_fn=_price_potential,
        )


def test_generation_prompt_defines_intrinsic_reward_as_phi():
    prompt = build_init_prompt('synthetic market statistics')
    assert 'state potential Phi(s)' in prompt
    assert 'gamma * Phi(s_next) - Phi(s_t)' in prompt
    assert 'DO NOT encode any of the following in Phi(s)' in prompt
    assert 'realized portfolio return' in prompt


def test_pure_gift_prompt_defines_direct_intrinsic_reward():
    prompt = build_init_prompt(
        'synthetic market statistics', pbir_enabled=False)
    assert 'DIRECT INTRINSIC REWARD SEMANTICS (PURE GIFT CONTROL)' in prompt
    assert 'directly produces the' in prompt
    assert 'does NOT compute gamma*Phi' in prompt
    assert 'POTENTIAL-BASED INTRINSIC REWARD SEMANTICS' not in prompt
