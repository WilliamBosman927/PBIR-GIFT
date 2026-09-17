"""
Main entry point for portfolio optimization with GIFT.

Usage:
    python main.py --config configs/config.yaml --experiment_name my_experiment

Pipeline:
    1. Load the YAML config file.
    2. Construct a GIFTController.
    3. Run the full iterative optimization loop.
    4. Emit final test results and baseline comparison.
"""

import argparse
import os
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent / 'core'))

from gift_controller import GIFTController


def main():
    parser = argparse.ArgumentParser(description='GIFT Portfolio Optimization')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config YAML file')
    parser.add_argument('--experiment_name', type=str, default='gift_portfolio',
                        help='Experiment name for results directory')
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help=('Explicit result directory. This is useful for a mounted NAS and '
              'takes precedence over results/<experiment_name>.'))
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument(
        '--pbir-mode', choices=['config', 'enabled', 'disabled'],
        default='config',
        help=('Override config.pbir.enabled for PBIR versus legacy direct-'
              'intrinsic-reward ablations'))
    parser.add_argument(
        '--fixed-artifact', type=str, default=None,
        help=('Verified shared-code artifact manifest. When supplied, GIFT '
              'skips every LLM call and replays this exact code/rule set.'))
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    if args.pbir_mode != 'config':
        config['pbir'] = dict(config.get(
            'pbir', config.get('potential_based_intrinsic_reward', {})))
        config['pbir']['enabled'] = args.pbir_mode == 'enabled'
    if args.fixed_artifact:
        config.setdefault('experiment', {})['fixed_artifact_path'] = str(
            Path(args.fixed_artifact).resolve())

    # Create controller and run
    if args.output_dir:
        experiment_dir = str(Path(args.output_dir).resolve())
    else:
        results_root = Path(os.environ.get('GIFT_RESULTS_ROOT', 'results'))
        experiment_dir = str(results_root / args.experiment_name)
    controller = GIFTController(config, experiment_dir, seed=args.seed)
    controller.run()


if __name__ == '__main__':
    main()
