"""Rebuild manuscript figures from existing, read-only experiment artifacts.

No training, LLM calls, result mutation, or new inferential resampling occurs.
Source fields prefixed test_ in Stage A are the internal validation evaluation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'manuscript/revision_20260906'
FIG = OUT / 'figures'
EVI = OUT / 'evidence'
FIG.mkdir(parents=True, exist_ok=True)
EVI.mkdir(parents=True, exist_ok=True)
SOURCES = {}
CATALOG = []
METHODS = ('pbir', 'pure_gift')
LABEL = {'pbir': 'PBIR-GIFT', 'pure_gift': 'Pure GIFT'}
COLOR = {'pbir': '#0072B2', 'pure_gift': '#D55E00'}
STYLE = {'pbir': '-', 'pure_gift': '--'}
MARKER = {'pbir': 'o', 'pure_gift': 's'}
DATASETS = ('portfolio_5stocks', 'portfolio_5stocks2')
PANEL = dict(zip(DATASETS, ('D1', 'D2')))
SEEDS = (42, 123, 456)
WINDOWS = tuple(f'W{i}' for i in range(1, 7))
ORDER = ('epochs_per_update', 'max_episodes', 'actor_lr', 'clip_epsilon', 'entropy_coef')
PARAM_LABEL = dict(zip(ORDER, ('PPO update epochs', 'Training episodes', 'Actor learning rate', 'PPO clipping coefficient', 'Entropy coefficient')))
METRICS = ('sharpe', 'sortino', 'total_return', 'max_drawdown')
METRIC_LABEL = dict(zip(METRICS, ('Sharpe ratio ↑', 'Sortino ratio ↑', 'Total return (%) ↑', 'Maximum drawdown (%) ↓')))
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                     'axes.labelsize': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.edgecolor': '#5D6570', 'axes.labelcolor': '#242A33',
                     'text.color': '#242A33', 'xtick.color': '#424B57', 'ytick.color': '#424B57',
                     'grid.color': '#DCE1E7', 'grid.linewidth': .6, 'axes.axisbelow': True,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white'})


def track(path):
    path = Path(path).resolve()
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    key = str(path.relative_to(ROOT)).replace('\\', '/')
    if key in SOURCES:
        assert SOURCES[key] == digest, f'Source changed during read: {key}'
    SOURCES[key] = digest
    return path


def read_json(path):
    return json.loads(track(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    return pd.read_csv(track(path))


def json_out(name, obj):
    (EVI / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def save(fig, name, purpose, caption, observation, caveat):
    for ext in ('png', 'pdf'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=320, bbox_inches='tight')
    plt.close(fig)
    CATALOG.append({'file': name, 'purpose': purpose, 'caption': caption,
                    'observation': observation, 'caveat': caveat})


def axes_style(ax):
    ax.grid(axis='y', alpha=.8)
    ax.tick_params(labelsize=9)


def legend(fig, handles=None, labels=None, y=.905):
    if handles is None:
        handles, labels = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, y),
               ncol=2, frameon=False, fontsize=10)


stage_a_root = ROOT / 'results/stage_a_tuning/portfolio_5stocks/W1/seed_42'
a = read_csv(stage_a_root / 'paired_sweep_summary.csv')
selection = read_json(stage_a_root / 'selection/selection_manifest.json')
selected_yaml = yaml.safe_load(track(stage_a_root / 'selection/selected_ppo.yaml').read_text(encoding='utf-8'))
assert len(a) == 25 and set(a['status']) == {'completed'}
assert set(a['seed']) == {42} and all(a.groupby('parameter').size() == 5)
assert list(a.parameter.drop_duplicates()) == list(ORDER)
assert selected_yaml['ppo'] == selection['selected_ppo']
assert all(a[f'{m}_failed_iterations'].sum() == 0 for m in METHODS)
renamed = a.rename(columns={c: c.replace('_test_', '_validation_') for c in a.columns if '_test_' in c})
renamed.to_csv(EVI / 'stage_a_validation_all_25_trials.csv', index=False)
stage_a_periods = []
for _, trial in a.iterrows():
    for method in METHODS:
        trial_comp = read_json(stage_a_root / 'trials' / trial.trial_name / method / 'final_comparison.json')
        stage_a_periods.append({'trial':trial.trial_name, 'method':method, 'retrain_period':trial_comp['retrain_period'],
                               'eval_period':trial_comp['eval_period'], 'n_returns':len(trial_comp['daily_returns']['method'])})
assert all(p['retrain_period'] == ['2020-01-02', '2020-03-31'] for p in stage_a_periods)
assert all(p['eval_period'] == ['2020-04-01', '2020-06-30'] for p in stage_a_periods)
for n, param in enumerate(ORDER, 1):
    group = a[a.parameter == param].sort_values('level')
    values = group.value.to_numpy()
    x = np.arange(len(group))
    winner = selection['selected_ppo'][param]
    chosen_idx = int(np.where(np.isclose(values, winner))[0][0])
    fig, axs = plt.subplots(2, 2, figsize=(9.2, 6.7))
    fig.suptitle(f'{n}. Validation sensitivity: {PARAM_LABEL[param]}', y=.99, fontsize=14, fontweight='bold')
    fig.text(.5, .948, 'D1 final internal validation: Apr–Jun 2020 (after lookback) | seed 42', ha='center', fontsize=10)
    for ax, metric in zip(axs.flat, METRICS):
        for method in METHODS:
            ax.plot(x, group[f'{method}_test_{metric}'], color=COLOR[method], linestyle=STYLE[method],
                    marker=MARKER[method], markersize=5.5, linewidth=1.8, label=LABEL[method])
        ax.axvline(chosen_idx, color='#606770', linewidth=1, linestyle=':', alpha=.8)
        ax.set_xticks(x, [f'{v:g}' for v in values])
        ax.set_xlabel(PARAM_LABEL[param])
        ax.set_ylabel(METRIC_LABEL[metric])
        axes_style(ax)
    legend(fig, y=.916)
    fig.text(.5, .022, f'Dotted line: selected value {winner:g}. Lines connect observed levels only; no confidence interval.', ha='center', fontsize=9)
    fig.subplots_adjust(top=.84, bottom=.13, hspace=.42, wspace=.34)
    save(fig, f'fig_{n:02d}_sweep_{param}',
         f'Compare both algorithms across the five observed {param} levels in sequential OFAT tuning.',
         f'Four internal-validation metrics for PBIR-GIFT and Pure GIFT. The nominal January–June 2020 validation interval is split into January 2–March 31 adaptation and April 1–June 30 final validation, with 20-day lookback warm-up. One seed (42), five levels, sequential OFAT; previously selected groups are frozen. Independent LLM code generation occurs in each branch/trial. Dotted line marks the level selected by the mean validation Sharpe of both methods. Lines are visual guides; no uncertainty interval is estimated.',
         f'Selected {param}={winner:g}; mean validation Sharpe={selection["selected_rows"][param]["mean_validation_sharpe"]:.6f}.',
         'These curves are tuning diagnostics, not held-out final test results or an isolated causal estimate of the hyperparameter effect. LLM code changes are a confound.')

# Portrait rendering of the final tuning group, which shares page 5 with the
# cross-window evidence in the two-column manuscript.
param = 'entropy_coef'
group = a[a.parameter == param].sort_values('level')
values = group.value.to_numpy()
x = np.arange(len(group))
winner = selection['selected_ppo'][param]
chosen_idx = int(np.where(np.isclose(values, winner))[0][0])
fig, axs = plt.subplots(2, 2, figsize=(3.35, 4.05))
fig.suptitle('5. Validation sensitivity: entropy coefficient', y=.995, fontsize=8.2, fontweight='bold')
fig.text(.5, .968, 'D1 internal validation: Apr–Jun 2020 | seed 42', ha='center', fontsize=5.5)
for ax, metric in zip(axs.flat, METRICS):
    for method in METHODS:
        ax.plot(x, group[f'{method}_test_{metric}'], color=COLOR[method],
                linestyle=STYLE[method], marker=MARKER[method], markersize=2.2,
                linewidth=.85, label=LABEL[method])
    ax.axvline(chosen_idx, color='#606770', linewidth=.65, linestyle=':', alpha=.8)
    ax.set_xticks(x, [f'{v:g}' for v in values])
    ax.set_xlabel('Entropy coefficient', fontsize=5.5)
    ax.set_ylabel(METRIC_LABEL[metric], fontsize=5.5)
    ax.grid(axis='y', alpha=.8)
    ax.tick_params(labelsize=5.1, pad=1.2)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .946),
           ncol=2, frameon=False, fontsize=5.8, handlelength=1.8)
fig.text(.5, .018, f'Dotted line: selected value {winner:g}; observed levels only.', ha='center', fontsize=5.2)
fig.subplots_adjust(top=.89, bottom=.10, left=.15, right=.985, hspace=.40, wspace=.38)
save(fig, 'fig_05_sweep_entropy_coef_column',
     'Provide a portrait one-column rendering of the entropy-coefficient sensitivity.',
     'The five observed levels and four internal-validation metrics are identical to fig_05_sweep_entropy_coef.',
     'The portrait geometry fills the journal column while preserving the paired tuning comparison.',
     'This is a layout variant, not an additional analysis.')

froot = ROOT / 'results/stage_f_full_window_cached_pbir/analysis_bundle'
cells = read_csv(froot / 'stage_f_cell_metrics.csv')
seeds = read_csv(froot / 'stage_f_seed_metrics.csv')
inference = read_json(froot / 'stage_f_inference.json')
assert len(cells) == 12 and len(seeds) == 72
assert not seeds.duplicated(['dataset','window','seed','method']).any()
assert set(cells.n_seeds) == {3}
assert set(seeds.seed) == set(SEEDS)
assert np.isclose(cells.delta_sharpe.mean(), inference['primary']['mean_cell_delta_sharpe'])
assert (cells.delta_sharpe > 0).sum() == inference['primary']['positive_cells']
cells.to_csv(EVI / 'fixed_code_cell_metrics_12.csv', index=False)
seeds.to_csv(EVI / 'fixed_code_seed_metrics_72.csv', index=False)

# Raw run outputs establish daily return trajectories, weights, timings and code sharing.
raw = {d: pd.read_pickle(track(ROOT / f'data/{d}.pkl')) for d in DATASETS}
track(ROOT / 'portfolio_env.py')
runs = {}
run_records, weight_records, daily_records, period_records = [], [], [], []
for dataset in DATASETS:
    for window in WINDOWS:
        stage = 'stage_c_cached_replication' if int(window[1:]) % 2 else 'stage_f_full_window_cached_pbir'
        all_date_lists = []
        code_hashes = []
        for seed in SEEDS:
            for method in METHODS:
                base = ROOT / 'results' / stage / dataset / window / f'seed_{seed}' / 'methods' / method
                summary = read_json(base / 'summary.json')
                comp = read_json(base / 'final_comparison.json')
                protocol = read_json(base / 'run_protocol.json')
                config = yaml.safe_load(track(base / 'resolved_config.yaml').read_text(encoding='utf-8'))
                days = sorted(d for d in raw[dataset] if comp['eval_period'][0] <= d <= comp['eval_period'][1])
                dates = days[21:]
                returns = np.asarray(comp['daily_returns']['method'], dtype=float)
                assert len(dates) == len(returns) and np.all(np.isfinite(returns))
                assert np.isclose((np.prod(1 + returns) - 1) * 100, comp['test_result']['test_total_return'])
                assert summary['failed_iterations'] == 0
                assert summary['llm_call_attempts'] == 0
                all_date_lists.append(dates)
                code_hashes.append((seed, method, protocol['shared_code_sha256']))
                runs[dataset, window, seed, method] = comp
                run_records.append({'dataset': dataset, 'window': window, 'seed': seed, 'method': method,
                    'source_stage': stage, 'wall_time_seconds': summary['wall_time_seconds'],
                    'llm_call_attempts': summary['llm_call_attempts'], 'llm_call_successes': summary['llm_call_successes'],
                    'n_returns': len(returns), 'date_first_return': dates[0], 'date_last_return': dates[-1],
                    'shared_code_sha256': protocol['shared_code_sha256']})
                weights = comp['test_result']['test_avg_weights']
                assert np.isclose(sum(weights.values()), 1, atol=1e-5)
                for ticker, weight in weights.items():
                    weight_records.append({'dataset': dataset, 'window': window, 'seed': seed, 'method': method,
                                           'ticker': ticker, 'average_evaluation_weight': weight})
                wealth = 100 * np.concatenate(([1.], np.cumprod(1 + returns)))
                for date, value in zip([days[20]] + dates, wealth):
                    daily_records.append({'dataset': dataset, 'window': window, 'seed': seed, 'method': method,
                                          'date': date, 'wealth_index': value})
        assert all(d == all_date_lists[0] for d in all_date_lists)
        for seed in SEEDS:
            assert len(set(h for s, m, h in code_hashes if s == seed)) == 1
        period_records.append({'dataset': dataset, 'window': window, 'eval_period': comp['eval_period'],
                               'date_initial_wealth': days[20], 'date_first_return': dates[0],
                               'date_last_return': dates[-1], 'n_daily_returns': len(dates)})
runtime = pd.DataFrame(run_records)
weights = pd.DataFrame(weight_records)
daily = pd.DataFrame(daily_records)
runtime.to_csv(EVI / 'fixed_code_runtime_72_runs.csv', index=False)
weights.to_csv(EVI / 'output_weights_432_asset_run_means.csv', index=False)
daily.to_csv(EVI / 'output_wealth_7632_seed_dates.csv', index=False)
json_out('evaluation_dates.json', period_records)

# Cross-window small multiples; uncertainty is seed SD, not CI.
fig, axs = plt.subplots(2, 4, figsize=(13.4, 6.7), sharex=True)
fig.suptitle('Fixed-code comparison across datasets and rolling windows', y=.995, fontsize=14, fontweight='bold')
fig.text(.5, .95, 'Three matched seeds per cell | identical LLM code within each pair | error bars: ±1 seed SD', ha='center', fontsize=10)
for row, dataset in enumerate(DATASETS):
    group = cells[cells.dataset == dataset].set_index('window').loc[list(WINDOWS)]
    for col, metric in enumerate(METRICS):
        key = 'max_drawdown_improvement' if metric == 'max_drawdown' else metric
        ax = axs[row, col]
        for method in METHODS:
            ax.errorbar(np.arange(6), group[f'{method}_mean_{key}'], yerr=group[f'{method}_seed_sd_{key}'],
                        color=COLOR[method], linestyle=STYLE[method], marker=MARKER[method], markersize=4,
                        linewidth=1.6, capsize=2.3, label=LABEL[method])
        ax.set_xticks(np.arange(6), WINDOWS)
        ax.set_title(f'{PANEL[dataset]} · {METRIC_LABEL[metric]}', loc='left', fontsize=10)
        ax.set_xlabel('Rolling window')
        axes_style(ax)
legend(fig, y=.915)
fig.subplots_adjust(top=.82, bottom=.11, wspace=.31, hspace=.40)
save(fig, 'fig_06_cross_dataset_window_metrics', 'Assess temporal and panel heterogeneity in held-out fixed-code results.',
     'Each line is one algorithm; dots are means over seeds 42, 123, and 456, and error bars are one sample standard deviation across seeds. The 12 dataset-window cells use Stage C for W1/W3/W5 and Stage F for W2/W4/W6. Returns and MDD are percentages; lower MDD is better. Lines connect ordered windows for readability, not a continuously traded portfolio.',
     'Mean Sharpe differences are positive in 10/12 cells, including 5/6 odd and 5/6 even-window cells.',
     'Adjacent windows share training/adaptation history; D1 and D2 share three stocks. Seed error bars are not confidence intervals for cross-market generalization.')

# Single-column portrait version used by the journal layout.  The arrangement
# changes from datasets-by-rows to metrics-by-rows so every panel remains
# legible at one-column width without altering any values.
fig, axs = plt.subplots(4, 2, figsize=(3.35, 7.45), sharex=True)
fig.suptitle('Fixed-code metrics across rolling windows', y=.995, fontsize=8.6, fontweight='bold')
fig.text(.5, .970, 'Three matched seeds; whiskers: ±1 seed SD', ha='center', fontsize=5.8)
for row, metric in enumerate(METRICS):
    for col, dataset in enumerate(DATASETS):
        group = cells[cells.dataset == dataset].set_index('window').loc[list(WINDOWS)]
        key = 'max_drawdown_improvement' if metric == 'max_drawdown' else metric
        ax = axs[row, col]
        for method in METHODS:
            ax.errorbar(np.arange(6), group[f'{method}_mean_{key}'],
                        yerr=group[f'{method}_seed_sd_{key}'], color=COLOR[method],
                        linestyle=STYLE[method], marker=MARKER[method], markersize=2.2,
                        linewidth=.85, capsize=1.25, capthick=.65, label=LABEL[method])
        ax.set_title(f'{PANEL[dataset]} · {METRIC_LABEL[metric]}', loc='left', fontsize=6.3, pad=2)
        ax.set_xticks(np.arange(6), WINDOWS)
        ax.grid(axis='y', alpha=.8)
        ax.tick_params(labelsize=5.5, pad=1.5)
        if row == len(METRICS) - 1:
            ax.set_xlabel('Rolling window', fontsize=5.8, labelpad=2)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .949),
           ncol=2, frameon=False, fontsize=6.2, handlelength=2.1)
fig.subplots_adjust(top=.915, bottom=.065, left=.13, right=.985, wspace=.37, hspace=.43)
save(fig, 'fig_06_cross_dataset_window_metrics_column',
     'Provide a portrait rendering of the cross-window metrics for one-column typesetting.',
     'The values and uncertainty summaries are identical to fig_06_cross_dataset_window_metrics; only panel geometry and typography differ.',
     'All four recorded metrics remain visible for both datasets at journal column width.',
     'This is a layout variant, not an additional analysis.')

fig, axs = plt.subplots(1, 2, figsize=(9.3, 3.9), sharey=True)
fig.suptitle('Out-of-sample Sharpe across rolling windows', y=1.01, fontsize=13, fontweight='bold')
for ax, dataset in zip(axs, DATASETS):
    group = cells[cells.dataset == dataset].set_index('window').loc[list(WINDOWS)]
    for method in METHODS:
        ax.errorbar(np.arange(6), group[f'{method}_mean_sharpe'], yerr=group[f'{method}_seed_sd_sharpe'],
                    marker=MARKER[method], color=COLOR[method], linestyle=STYLE[method], capsize=3,
                    label=LABEL[method])
    ax.set_title(PANEL[dataset], loc='left')
    ax.set_xticks(np.arange(6), WINDOWS)
    ax.set_ylabel('Sharpe ratio')
    axes_style(ax)
legend(fig, y=.963)
fig.subplots_adjust(top=.78, bottom=.19, wspace=.22)
fig.text(.5, .02, 'Means ±1 seed SD (n=3); same generated code per pair. Connecting lines are visual guides.', ha='center', fontsize=9)
save(fig, 'fig_07_fixed_code_sharpe', 'Provide a compact primary-metric view of all 12 cells.',
     'Out-of-sample Sharpe by dataset and window; both methods use the same cached LLM artifact within each seed. Error bars show ±1 sample SD across three seeds, not inferential confidence intervals.',
     'D1 average Sharpe improvement is +0.219860 and D2 improvement is +0.336366.',
     'The hierarchical bootstrap 95% interval for the overall difference includes zero; this chart alone does not establish significant superiority.')

fig, axs = plt.subplots(2, 1, figsize=(3.35, 4.45), sharex=True, sharey=True)
fig.suptitle('Out-of-sample Sharpe by rolling window', y=.995, fontsize=8.6, fontweight='bold')
for ax, dataset in zip(axs, DATASETS):
    group = cells[cells.dataset == dataset].set_index('window').loc[list(WINDOWS)]
    for method in METHODS:
        ax.errorbar(np.arange(6), group[f'{method}_mean_sharpe'],
                    yerr=group[f'{method}_seed_sd_sharpe'], marker=MARKER[method],
                    color=COLOR[method], linestyle=STYLE[method], linewidth=1,
                    markersize=2.6, capsize=1.6, label=LABEL[method])
    ax.set_title(PANEL[dataset], loc='left', fontsize=7, pad=2)
    ax.set_xticks(np.arange(6), WINDOWS)
    ax.set_ylabel('Sharpe ratio', fontsize=6.5)
    ax.grid(axis='y', alpha=.8)
    ax.tick_params(labelsize=6, pad=1.5)
axs[-1].set_xlabel('Rolling window', fontsize=6.5)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945),
           ncol=2, frameon=False, fontsize=6.4)
fig.text(.5, .015, 'Means ±1 seed SD (n=3); identical generated code within each pair.',
         ha='center', fontsize=5.6)
fig.subplots_adjust(top=.885, bottom=.095, left=.16, right=.98, hspace=.28)
save(fig, 'fig_07_fixed_code_sharpe_column',
     'Provide a portrait one-column rendering of fixed-code Sharpe results.',
     'The plotted means and seed standard deviations are identical to fig_07_fixed_code_sharpe.',
     'The stacked panels use the column height more efficiently in a two-column manuscript.',
     'This is a layout variant, not an additional analysis.')

# Input market data, display common active experiment years and adjusted close normalization.
input_facts = {}
input_rows = []
asset_palette = ['#0072B2', '#D55E00', '#CC79A7', '#8A8D35', '#707070']
asset_styles = ['-', '--', '-.', ':', (0, (5, 1, 1, 1))]
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
fig.suptitle('Input market data: indexed adjusted closing prices', y=.995, fontsize=14, fontweight='bold')
fig.text(.5, .935, 'Observed trading dates, 2018-01-02–2024-06-28 | first displayed close = 100 | logarithmic y-axis', ha='center', fontsize=9.5)
for ax, dataset in zip(axs, DATASETS):
    dates = sorted(raw[dataset])
    assets = sorted(raw[dataset][dates[0]]['price'])
    shown_dates = [d for d in dates if '2018-01-01' <= d <= '2024-06-30']
    for ticker, color, style in zip(assets, asset_palette, asset_styles):
        values = np.array([raw[dataset][d]['price'][ticker]['adjusted_close'] for d in shown_dates])
        assert np.all(np.isfinite(values)) and np.all(values > 0)
        indexed = 100 * values / values[0]
        ax.plot(pd.to_datetime(shown_dates), indexed, color=color, linestyle=style, linewidth=1.4, label=ticker)
        input_rows.extend({'dataset': dataset, 'ticker': ticker, 'date': d, 'adjusted_close': float(v), 'indexed_adjusted_close': float(z)} for d, v, z in zip(shown_dates, values, indexed))
    input_facts[dataset] = {'stored_trading_dates': len(dates), 'stored_start': dates[0], 'stored_end': dates[-1],
                            'assets': assets, 'display_start': shown_dates[0], 'display_end': shown_dates[-1],
                            'display_trading_dates': len(shown_dates)}
    ax.set_title(PANEL[dataset], loc='left')
    ax.set_yscale('log')
    ax.set_ylabel('Indexed adjusted close (log scale)')
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper left', ncol=3, fontsize=8, frameon=False)
    axes_style(ax)
fig.subplots_adjust(top=.78, bottom=.17, wspace=.18)
save(fig, 'fig_08_input_market_data', 'Show the actual market inputs and different panel composition.',
     'Adjusted closing prices from the two supplied pickle files are normalized to 100 on 2018-01-02 and plotted on a log scale through 2024-06-28. D1 comprises AMZN/JNJ/MSFT/NFLX/TSLA; D2 comprises AMZN/CAT/JNJ/MSFT/XOM. The full files span 2010-06-29 to 2024-06-28 with 3,524 dates each.',
     'The asset paths exhibit visibly different growth and drawdown regimes; three stocks are shared between panels.',
     'This is an input illustration, not a portfolio backtest or evidence of predictive accuracy. No values are imputed.')

fig, axs = plt.subplots(2, 1, figsize=(3.35, 6.80), sharex=True, sharey=True)
fig.suptitle('Input market data: indexed adjusted closes', y=.995, fontsize=8.3, fontweight='bold')
for ax, dataset in zip(axs, DATASETS):
    dates = sorted(raw[dataset])
    assets = sorted(raw[dataset][dates[0]]['price'])
    shown_dates = [d for d in dates if '2018-01-01' <= d <= '2024-06-30']
    for ticker, color, style in zip(assets, asset_palette, asset_styles):
        values = np.array([raw[dataset][d]['price'][ticker]['adjusted_close'] for d in shown_dates])
        indexed = 100 * values / values[0]
        ax.plot(pd.to_datetime(shown_dates), indexed, color=color, linestyle=style,
                linewidth=.85, label=ticker)
    ax.set_title(PANEL[dataset], loc='left', fontsize=7, pad=2)
    ax.set_yscale('log')
    ax.set_ylabel('Indexed close (log)', fontsize=6.2)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.legend(loc='upper left', ncol=3, fontsize=5.2, frameon=False,
              handlelength=1.5, columnspacing=.7)
    ax.grid(axis='y', alpha=.8)
    ax.tick_params(labelsize=6, pad=1.5)
fig.text(.5, .018, '2018-01-02–2024-06-28; first displayed close = 100.', ha='center', fontsize=5.6)
fig.subplots_adjust(top=.94, bottom=.09, left=.17, right=.985, hspace=.26)
save(fig, 'fig_08_input_market_data_column',
     'Provide a portrait one-column rendering of the two market-input panels.',
     'The dates, prices, normalization, and logarithmic scale are identical to fig_08_input_market_data.',
     'Stacked panels preserve readable asset trajectories at journal column width.',
     'This is a layout variant, not an additional analysis.')
pd.DataFrame(input_rows).to_csv(EVI / 'input_adjusted_close_display.csv', index=False)

# Each window restarts at 100; 12 independent panels avoid false stitching.
daily_mean = daily.groupby(['dataset', 'window', 'method', 'date'], as_index=False).agg(wealth_index_mean=('wealth_index', 'mean'), wealth_index_sd=('wealth_index', 'std'))
daily_mean.to_csv(EVI / 'output_wealth_mean_by_cell_date.csv', index=False)
def wealth_figure(datasets, name):
    ncols = len(datasets)
    if ncols == 2:
        fig, axs = plt.subplots(6, 2, figsize=(11.2, 14.4), sharey=True)
        positions = [(axs[w, d], dataset, window) for w, window in enumerate(WINDOWS) for d, dataset in enumerate(datasets)]
    else:
        fig, axs = plt.subplots(3, 2, figsize=(10.2, 8.7), sharey=True)
        positions = [(ax, datasets[0], window) for ax, window in zip(axs.flat, WINDOWS)]
    fig.suptitle('Output portfolio wealth in each evaluation window', y=.995, fontsize=14, fontweight='bold')
    fig.text(.5, .967, 'Initial wealth = 100 per window | mean of three seed-specific wealth paths | no stitching', ha='center', fontsize=9.5)
    for ax, dataset, window in positions:
        for method in METHODS:
            z = daily_mean[(daily_mean.dataset == dataset) & (daily_mean.window == window) & (daily_mean.method == method)]
            ax.plot(pd.to_datetime(z.date), z.wealth_index_mean, color=COLOR[method], linestyle=STYLE[method], linewidth=1.7, label=LABEL[method])
        ax.axhline(100, color='#676E77', linewidth=.7)
        ax.set_title(f'{PANEL[dataset]} · {window}   {str(z.date.iloc[0])[:4]}–{str(z.date.iloc[-1])[:4]}', loc='left', fontsize=10)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.tick_params(axis='x', labelsize=8)
        ax.set_ylabel('Wealth index')
        axes_style(ax)
    legend(fig, y=.95)
    fig.subplots_adjust(top=.9, bottom=.045, hspace=.53, wspace=.18)
    save(fig, name, 'Visualize realized output paths without selecting the best window or seed.',
         'Net daily returns from all matched Stage C/F runs are separately compounded from initial wealth 100, then the three seed-specific wealth indices are arithmetically averaged within each method and cell. Each window is plotted separately. Dates map to the supplied trading calendar after the 20-day lookback; the initial point is day index 20 and first return day index 21. No smoothing or confidence ribbon is applied.',
         'Trajectories expose gains and losses in every included cell and show the timing of separation between algorithms.',
         'The mean of wealth paths is not a traded aggregate portfolio. Evaluation windows are not stitched. The chart does not depict full-year returns.')

wealth_figure(DATASETS, 'fig_09_output_wealth_all_12_cells')
wealth_figure((DATASETS[0],), 'fig_09a_output_wealth_D1')
wealth_figure((DATASETS[1],), 'fig_09b_output_wealth_D2')

# Portrait all-cell wealth matrix for direct placement in a journal column.
fig, axs = plt.subplots(6, 2, figsize=(3.35, 8.80))
fig.suptitle('Portfolio wealth in each evaluation window', y=.995, fontsize=8.1, fontweight='bold')
fig.text(.5, .976, 'Initial wealth = 100; mean of three seeds; windows restart', ha='center', fontsize=5.2)
for row, window in enumerate(WINDOWS):
    for col, dataset in enumerate(DATASETS):
        ax = axs[row, col]
        for method in METHODS:
            z = daily_mean[(daily_mean.dataset == dataset) &
                           (daily_mean.window == window) &
                           (daily_mean.method == method)]
            ax.plot(pd.to_datetime(z.date), z.wealth_index_mean, color=COLOR[method],
                    linestyle=STYLE[method], linewidth=.70, label=LABEL[method])
        ax.axhline(100, color='#676E77', linewidth=.35)
        ax.set_title(f'{PANEL[dataset]} · {window}', loc='left', fontsize=5.7, pad=1.2)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax.tick_params(axis='both', labelsize=4.6, pad=1)
        if col == 0:
            ax.set_ylabel('Wealth', fontsize=4.9, labelpad=1.5)
        ax.grid(axis='y', alpha=.75, linewidth=.35)
handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .958),
           ncol=2, frameon=False, fontsize=5.5, handlelength=1.8)
fig.subplots_adjust(top=.925, bottom=.035, left=.14, right=.985, wspace=.24, hspace=.52)
save(fig, 'fig_09_output_wealth_all_12_cells_column',
     'Provide a portrait one-column rendering of all 12 evaluation wealth paths.',
     'The 12 cells, method means, dates, initial wealth, and absence of smoothing are identical to fig_09_output_wealth_all_12_cells.',
     'The 6-by-2 geometry keeps every dataset-window path visible beside the input/output discussion.',
     'This is a layout variant, not an additional analysis.')

avg_weights = weights.groupby(['dataset','method','ticker'], as_index=False).agg(mean_weight=('average_evaluation_weight','mean'), sd_run_mean_weight=('average_evaluation_weight','std'), n_runs=('seed','size'))
assert set(avg_weights.n_runs) == {18}
avg_weights.to_csv(EVI / 'output_average_weights.csv', index=False)
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6), sharex=True)
fig.suptitle('Output allocation: mean evaluation-period portfolio weights', y=.995, fontsize=13, fontweight='bold')
fig.text(.5, .94, 'Equal weight per run: 6 windows × 3 seeds per method and dataset | bars: means; whiskers: run-level SD', ha='center', fontsize=9)
for ax, dataset in zip(axs, DATASETS):
    assets = sorted(set(weights[weights.dataset == dataset].ticker) - {'CASH'}) + ['CASH']
    y = np.arange(len(assets))
    for method, shift in zip(METHODS, (-.18, .18)):
        z = avg_weights[(avg_weights.dataset == dataset) & (avg_weights.method == method)].set_index('ticker').loc[assets]
        ax.barh(y + shift, z.mean_weight * 100, height=.32, xerr=z.sd_run_mean_weight * 100,
                color=COLOR[method], alpha=.9 if method == 'pbir' else .65, edgecolor=COLOR[method],
                hatch='' if method == 'pbir' else '//', error_kw={'linewidth':.8,'capsize':2}, label=LABEL[method])
    ax.set_yticks(y, assets)
    ax.invert_yaxis()
    ax.set_xlabel('Portfolio weight (%)')
    ax.set_title(PANEL[dataset], loc='left')
    ax.grid(axis='x', alpha=.8)
    ax.set_xlim(left=0)
legend(fig, y=.90)
fig.subplots_adjust(top=.79, bottom=.13, wspace=.19)
save(fig, 'fig_10_output_average_weights', 'Show the allocation output across all runs rather than a selected best case.',
     'Average evaluation-period asset weights are averaged equally over all 18 window-seed runs per method and dataset. Whiskers show SD of the 18 run-average weights. Cash is included, and the asset weights sum to one per run. No best-window or best-seed selection is used.',
     'The output includes cash allocation and asset diversification; the exact values are provided in output_average_weights.csv.',
     'Run SD describes heterogeneous, partly dependent conditions and is not a confidence interval. The summaries cannot establish that allocation changes caused the Sharpe difference.')

fig, axs = plt.subplots(2, 1, figsize=(3.35, 3.95), sharex=True)
fig.suptitle('Mean evaluation-period portfolio weights', y=.995, fontsize=8.3, fontweight='bold')
for ax, dataset in zip(axs, DATASETS):
    assets = sorted(set(weights[weights.dataset == dataset].ticker) - {'CASH'}) + ['CASH']
    y = np.arange(len(assets))
    for method, shift in zip(METHODS, (-.17, .17)):
        z = avg_weights[(avg_weights.dataset == dataset) & (avg_weights.method == method)].set_index('ticker').loc[assets]
        ax.barh(y + shift, z.mean_weight * 100, height=.30,
                xerr=z.sd_run_mean_weight * 100, color=COLOR[method],
                alpha=.9 if method == 'pbir' else .65, edgecolor=COLOR[method],
                hatch='' if method == 'pbir' else '//',
                error_kw={'linewidth':.55, 'capsize':1.5}, label=LABEL[method])
    ax.set_yticks(y, assets)
    ax.invert_yaxis()
    ax.set_title(PANEL[dataset], loc='left', fontsize=7, pad=2)
    ax.grid(axis='x', alpha=.8)
    ax.tick_params(labelsize=5.8, pad=1.5)
    ax.set_xlim(left=0)
axs[-1].set_xlabel('Portfolio weight (%)', fontsize=6.2)
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .945),
           ncol=2, frameon=False, fontsize=6.2)
fig.subplots_adjust(top=.87, bottom=.105, left=.17, right=.98, hspace=.30)
save(fig, 'fig_10_output_average_weights_column',
     'Provide a portrait one-column rendering of average portfolio allocations.',
     'The bar means, run-level standard deviations, assets, and cash entries are identical to fig_10_output_average_weights.',
     'Stacked dataset panels avoid undersized horizontal panels in the two-column paper.',
     'This is a layout variant, not an additional analysis.')

pairs = runtime.pivot(index=['dataset','window','seed'], columns='method', values='wall_time_seconds').reset_index()
pairs['pbir_over_pure_gift_ratio'] = pairs.pbir / pairs.pure_gift
pairs.to_csv(EVI / 'runtime_36_matched_pairs.csv', index=False)
efficiency = {'n_runs': len(runtime), 'n_pairs': len(pairs),
              'llm_calls_replay_total': int(runtime.llm_call_attempts.sum()),
              'mean_pair_runtime_ratio': float(pairs.pbir_over_pure_gift_ratio.mean()),
              'median_pair_runtime_ratio': float(pairs.pbir_over_pure_gift_ratio.median()),
              'ratio_of_mean_runtimes': float(pairs.pbir.mean() / pairs.pure_gift.mean()),
              'methods': {m: {'mean_wall_time_seconds': float(runtime[runtime.method == m].wall_time_seconds.mean()),
                              'median_wall_time_seconds': float(runtime[runtime.method == m].wall_time_seconds.median()),
                              'min_wall_time_seconds': float(runtime[runtime.method == m].wall_time_seconds.min()),
                              'max_wall_time_seconds': float(runtime[runtime.method == m].wall_time_seconds.max()),
                              'total_wall_time_hours': float(runtime[runtime.method == m].wall_time_seconds.sum()/3600)} for m in METHODS},
              'limitation': 'Historical application wall time includes source training/adaptation/evaluation and unstandardized hardware load, concurrency and run environment; not a controlled GPU benchmark or proof of speedup. Both cached branches have zero LLM calls during replay, excluding earlier artifact generation.'}
fig, ax = plt.subplots(figsize=(6.0,5.3))
for dataset, marker in zip(DATASETS, ('o','s')):
    z = pairs[pairs.dataset == dataset]
    ax.scatter(z.pure_gift / 60, z.pbir / 60, s=45, marker=marker, color='#0072B2',
               facecolors='#0072B2' if marker == 'o' else 'none', linewidths=1.3, label=PANEL[dataset])
low = min(pairs.pbir.min(), pairs.pure_gift.min()) / 60 * .82
high = max(pairs.pbir.max(), pairs.pure_gift.max()) / 60 * 1.18
ax.plot([low,high], [low,high], linestyle=':', color='#555E6B', linewidth=1, label='Equal wall time')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set(xlim=(low,high), ylim=(low,high), xlabel='Pure GIFT wall time (minutes; log scale)', ylabel='PBIR-GIFT wall time (minutes; log scale)')
ax.set_title('Recorded runtime of 36 matched replay pairs', loc='left', fontsize=12, pad=23)
ax.text(0, 1.025, f'Historical logs | median paired ratio = {efficiency["median_pair_runtime_ratio"]:.3f}', transform=ax.transAxes, fontsize=9)
ax.legend(frameon=False, fontsize=9, loc='upper left')
axes_style(ax)
fig.text(.5,.012, 'Descriptive application wall time; hardware load and concurrency were not standardized.', ha='center', fontsize=8)
fig.subplots_adjust(bottom=.15,top=.84)
save(fig, 'fig_11_efficiency_recorded_runtime', 'Audit available efficiency evidence without implying a controlled speed comparison.',
     'Each point represents one dataset-window-seed pair; log-scaled axes show application wall_time_seconds divided by 60 from each cached replay summary. The dotted diagonal marks equal runtime. All 72 replay summaries report zero LLM calls. Runtime includes training and downstream evaluation and was not measured under standardized hardware load or concurrency.',
     f'The median PBIR/Pure GIFT runtime ratio is {efficiency["median_pair_runtime_ratio"]:.4f}; both branches avoid repeated LLM calls during replay.',
     'Historical timings cannot establish computational superiority, cost neutrality, or negligible overhead; earlier LLM artifact generation costs are excluded.')

panel_means = []
for dataset in DATASETS:
    z = cells[cells.dataset == dataset]
    record = {'dataset': dataset, 'cells':len(z), 'positive_sharpe_cells':int((z.delta_sharpe>0).sum())}
    for metric in METRICS:
        key = 'max_drawdown_improvement' if metric == 'max_drawdown' else metric
        for method in METHODS:
            record[f'{method}_mean_{metric}'] = float(z[f'{method}_mean_{key}'].mean())
        record[f'delta_{metric}_pbir_minus_gift'] = record[f'pbir_mean_{metric}'] - record[f'pure_gift_mean_{metric}']
    panel_means.append(record)
pd.DataFrame(panel_means).to_csv(EVI / 'cross_dataset_descriptive_summary.csv', index=False)
stage_i = read_json(ROOT / 'results/stage_i_final_comparison/stage_i_inference.json')
contrasts = pd.DataFrame(stage_i['key_contrasts'])
contrasts.to_csv(EVI / 'stage_i_six_sharpe_contrasts.csv', index=False)
fig, ax = plt.subplots(figsize=(11.6,5.3))
y = np.arange(len(contrasts))
for i, r in contrasts.iterrows():
    c = '#0072B2' if i == 0 else '#68727E'
    ax.errorbar(r.mean_cell_delta_sharpe, i,
                xerr=[[r.mean_cell_delta_sharpe-r.bootstrap_ci95_low],[r.bootstrap_ci95_high-r.mean_cell_delta_sharpe]],
                color=c, marker='o' if i == 0 else 's', capsize=4, markersize=6, linewidth=1.5)
    ax.text(1.6, i, f'{r.mean_cell_delta_sharpe:+.3f}    [{r.bootstrap_ci95_low:+.3f}, {r.bootstrap_ci95_high:+.3f}]    {r.holm_adjusted_p_six_contrasts:.4f}',
            va='center', fontsize=9, fontfamily='DejaVu Sans Mono')
ax.set_yticks(y, [r.label + (' [fixed code]' if i == 0 else ' [pipeline]') for i, r in contrasts.iterrows()])
ax.invert_yaxis()
ax.axvline(0, color='#49525C', linestyle=':', linewidth=1)
ax.set_xlim(-.55,3.5)
ax.set_xticks([-.5,0,.5,1,1.5])
ax.set_xlabel('Mean cell-level Sharpe difference (12 cells)')
ax.set_title('Six final Sharpe contrasts and uncertainty',loc='left',fontsize=13,pad=35)
ax.text(1.6, -.72, 'Delta        95% bootstrap CI     Holm p',fontsize=9,fontfamily='DejaVu Sans Mono')
ax.grid(axis='x',alpha=.55)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y',length=0)
fig.subplots_adjust(left=.35,right=.985,top=.8,bottom=.16)
fig.text(.5,.03,'PBIR contrast: Stage C/F fixed-code replay. External baselines: pipeline GIFT. Holm correction across six contrasts.',ha='center',fontsize=9)
save(fig, 'fig_12_six_final_sharpe_contrasts', 'Expose baseline sensitivity and interval evidence in the final six-contrast family.',
     'Points show mean cell-level Sharpe differences across 12 dataset-window cells; horizontal intervals are the recorded 95% hierarchical block-bootstrap intervals (10,000 resamples; block length 20). The right column reports Holm-adjusted one-sided exact sign-flip p values across the final six-contrast family. PBIR–Pure GIFT uses fixed-code Stage C/F, whereas the external baseline contrasts use full-pipeline GIFT sources from their respective stages.',
     'The PBIR point estimate is positive, but its bootstrap interval includes zero. The independent PPO-Tuned contrast is negative. XGBoost has a positive bootstrap lower bound but does not pass the six-contrast Holm threshold.',
     'Different rows have different source protocols; do not treat the first row as a transitive PBIR-versus-external-baseline comparison. CI and adjusted-p criteria are both disclosed.')

# Portrait forest plot with an audit table beneath it.  Short display labels
# preserve the same six contrasts while retaining every delta, interval, and
# adjusted p value at one-column width.
short_labels = [
    'PBIR − Pure [fixed]',
    'Pure − PPO-M [pipe]',
    'Pure − PPO-T [pipe]',
    'Pure − SMA [pipe]',
    'Pure − BB [pipe]',
    'Pure − XGB [pipe]',
]
fig, ax = plt.subplots(figsize=(3.35, 5.25))
y = np.arange(len(contrasts))
for i, r in contrasts.iterrows():
    c = '#0072B2' if i == 0 else '#68727E'
    ax.errorbar(r.mean_cell_delta_sharpe, i,
                xerr=[[r.mean_cell_delta_sharpe-r.bootstrap_ci95_low],
                      [r.bootstrap_ci95_high-r.mean_cell_delta_sharpe]],
                color=c, marker='o' if i == 0 else 's', capsize=2.2,
                markersize=3.5, linewidth=1)
ax.set_yticks(y, short_labels)
ax.invert_yaxis()
ax.axvline(0, color='#49525C', linestyle=':', linewidth=.8)
ax.set_xlim(-.58, 1.58)
ax.set_xlabel('Mean cell-level Sharpe difference', fontsize=6.2)
ax.set_title('Six final Sharpe contrasts', loc='left', fontsize=8.3, pad=6)
ax.grid(axis='x', alpha=.55)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0, labelsize=5.5, pad=2)
ax.tick_params(axis='x', labelsize=5.6)
fig.subplots_adjust(left=.39, right=.98, top=.94, bottom=.47)
fig.text(.07, .305, 'Contrast             Delta       95% bootstrap CI      Holm p',
         fontsize=5.0, fontfamily='DejaVu Sans Mono')
for i, (short, (_, r)) in enumerate(zip(short_labels, contrasts.iterrows())):
    row = f'{short:<20} {r.mean_cell_delta_sharpe:+.3f}  [{r.bootstrap_ci95_low:+.3f},{r.bootstrap_ci95_high:+.3f}]  {r.holm_adjusted_p_six_contrasts:.4f}'
    fig.text(.07, .270 - i*.039, row, fontsize=4.8, fontfamily='DejaVu Sans Mono')
fig.text(.5, .018, 'Fixed = Stage C/F cached replay; pipe = recorded full-pipeline source.',
         ha='center', fontsize=4.9)
save(fig, 'fig_12_six_final_sharpe_contrasts_column',
     'Provide a portrait one-column rendering of the six final Sharpe contrasts.',
     'The six deltas, hierarchical-bootstrap intervals, and Holm-adjusted p values are identical to fig_12_six_final_sharpe_contrasts.',
     'The forest plot and audit table preserve numeric traceability at journal column width.',
     'This is a layout variant, not an additional analysis.')
evidence = {'stage_a': {'n_trials':25, 'seeds':[42], 'validation_period':selection['validation_period'],
                       'actual_adaptation_period':stage_a_periods[0]['retrain_period'],
                       'actual_validation_period':stage_a_periods[0]['eval_period'],
                       'n_return_days':stage_a_periods[0]['n_returns'],
                       'selected_ppo':selection['selected_ppo'], 'selected_rows':selection['selected_rows'],
                       'parameter_order':list(ORDER), 'llm_generation':'Independent code generation per trial and method; sequential OFAT, not a pure hyperparameter causal experiment'},
            'fixed_code':{'n_method_runs':72,'n_pairs':36,'n_cells':12,'seeds':list(SEEDS),
                          'panel_means':panel_means,'primary':inference['primary'],'metrics':inference['metrics'],
                          'odd_even':inference['odd_even_generalization'],
                          'multiplicity_family':'Stage-F five-metric family; do not substitute Stage-I six-contrast family adjusted p values',
                          'caveats':inference['caveats']},
            'efficiency':efficiency, 'input':input_facts, 'stage_i_six_contrasts':stage_i['key_contrasts'],
            'output':{'n_daily_wealth_rows':len(daily),'n_asset_run_mean_rows':len(weights),
                      'mean_weights':avg_weights.to_dict('records'),'evaluation_dates':period_records,
                      'wealth_aggregation':'Compounded seed wealth from 100, then equal-weight arithmetic mean over three seeds; each window separately restarted.'}}
json_out('manuscript_numeric_evidence.json', evidence)
json_out('figure_catalog.json', CATALOG)

catalog_text = '# Figure catalog\n\nAll figures are real PNG (320 dpi) and vector PDF exports. Source values are supplied in evidence/*.csv.\n\n'
for item in CATALOG:
    catalog_text += f'## {item["file"]}\n\nPurpose: {item["purpose"]}\n\nCaption: {item["caption"]}\n\nObservation: {item["observation"]}\n\nInterpretation limit: {item["caveat"]}\n\n'
(EVI / 'figure-catalog.md').write_text(catalog_text, encoding='utf-8')
after = {key:hashlib.sha256((ROOT/key).read_bytes()).hexdigest() for key in SOURCES}
assert after == SOURCES, 'One or more source artifacts changed during figure generation.'
json_out('source_hash_audit.json', {'source_file_count':len(SOURCES), 'before':SOURCES,'after':after,'all_unchanged':True})
stats = '# Statistical provenance\n\nNo new hypothesis tests or resampling were run. All inference is read from the locked Stage-F analysis JSON. The unit is a dataset-window mean of three paired seeds (12 cells), with dependence limitations inherited from that analysis.\n\n'
stats += '| Metric contrast | Mean difference | Bootstrap 95% interval | Hedges g | Holm p (five metrics) |\n|---|---:|---:|---:|---:|\n'
for metric, v in inference['metrics'].items():
    stats += f'| {metric} | {v["mean_cell_delta"]:.6f} | [{v["bootstrap"]["ci95"][0]:.6f}, {v["bootstrap"]["ci95"][1]:.6f}] | {v["hedges_g_paired"]:.6f} | {v["holm_adjusted_p"]:.6f} |\n'
stats += '\nAll five bootstrap intervals contain zero. The prespecified strict full-window gate is false. Sign-flip tests and bootstrap intervals quantify uncertainty differently; neither can be omitted to manufacture significance. Stage-A curves are single-seed selection diagnostics.\n'
(EVI / 'stats-appendix.md').write_text(stats, encoding='utf-8')
report = '# Analysis report\n\nQuestion: Does fixed-code PBIR shaping change out-of-sample investment performance, and what can existing records say about runtime and cross-window robustness?\n\n'
report += 'Validated 25 Stage-A trial rows, 72 fixed-code method runs (36 pairs), 12 complete cells with three seeds, identical cached code within every pair, all return/date alignments, all weight sums, and zero replay LLM calls. Inputs remained byte-identical.\n\n'
report += '## Claim candidates\n\n- Claim: PBIR has a positive overall Sharpe point estimate and wins 10/12 cells.\n  - Source: Stage-F inference and all 72 run outputs.\n  - Allowed: directionally favorable, bounded by uncertainty.\n  - Forbidden: universally or conclusively significantly superior.\n  - Uncertainty: 95% bootstrap interval crosses zero.\n  - Next check: independent asset universes, prespecified validation.\n  - Decision: keep with boundary.\n\n'
report += '- Claim: Both cached methods make no LLM calls during controlled replay.\n  - Source: 72 summary files.\n  - Allowed: no repeated generation cost in these replays.\n  - Forbidden: PBIR uniquely eliminates LLM costs, trains faster, or has negligible measured overhead.\n  - Uncertainty: prior generation is excluded; timing conditions are not controlled.\n  - Next check: standardized hardware timing if a speed claim is required.\n  - Decision: keep descriptive.\n'
(EVI / 'analysis-report.md').write_text(report, encoding='utf-8')
print(json.dumps({'output':str(OUT),'figures':len(CATALOG),'sources_unchanged':len(SOURCES),'panel_means':panel_means,'efficiency':efficiency}, ensure_ascii=False, indent=2))
