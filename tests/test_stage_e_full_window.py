from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from analyze_stage_e_full_window_reproduction import (
    DATASETS,
    METRICS,
    WINDOWS,
    _paper_replication,
)
from lock_stage_e_results import _validate_cells
from run_stage_e_full_window_baselines import _assert_isolated_output


class StageEFullWindowTests(unittest.TestCase):
    def test_stage_e_output_is_disjoint_from_locked_roots(self) -> None:
        project = Path(__file__).resolve().parents[1]
        stage_b = (project / "results" / "stage_b_final").resolve()
        with self.assertRaises(ValueError):
            _assert_isolated_output(
                (project / "results" / "stage_d_ppo_baselines").resolve(),
                stage_b,
            )
        with self.assertRaises(ValueError):
            _assert_isolated_output((project / "results").resolve(), stage_b)
        _assert_isolated_output(
            (project / "results" / "stage_e_full_window_baselines").resolve(),
            stage_b,
        )

    def test_incomplete_stage_e_cannot_be_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "incomplete Stage-E"):
                _validate_cells(Path(directory))

    def test_light_mix_directional_replication_rule(self) -> None:
        values = {}
        for window in WINDOWS:
            for metric in METRICS:
                gift = 2.0
                ppo = 1.0
                if metric == "test_max_drawdown":
                    gift, ppo = 1.0, 2.0
                values[(DATASETS[0], window, "pure_gift", metric)] = gift
                values[(DATASETS[0], window, "ppo_test_only", metric)] = ppo
        decision, rows = _paper_replication(values)
        self.assertTrue(decision["directional_replication_supported"])
        self.assertEqual(decision["positive_sharpe_windows"], 6)
        self.assertEqual(decision["windows_winning_at_least_4_of_5_metrics"], 6)
        self.assertTrue(all(row["gift_metric_wins"] == 5 for row in rows))


if __name__ == "__main__":
    unittest.main()
