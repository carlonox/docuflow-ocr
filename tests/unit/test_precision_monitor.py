"""Unit tests for the precision monitor."""

from __future__ import annotations

from docuflow.learning.precision_monitor import (
    RESULT_FULL,
    RESULT_NONE,
    PrecisionMonitor,
)


class TestPrecisionMonitor:
    def test_full_match_series(self, tmp_path):
        monitor = PrecisionMonitor(
            stats_path=str(tmp_path / "stats.json"),
            min_samples_for_alerts=10,
        )
        for _ in range(20):
            monitor.record_result([True, True])
        assert monitor.window_precision()["full"] == 100.0
        assert monitor.global_precision()["full"] == 100.0

    def test_partial_and_none(self, tmp_path):
        monitor = PrecisionMonitor(
            stats_path=str(tmp_path / "stats.json"),
            fields=["field_a", "field_b"],
        )
        monitor.record_result([True, False])  # partial
        monitor.record_result([False, False])  # none
        assert monitor.window_precision()["partial"] == 50.0
        assert monitor.window_precision()["none"] == 50.0

    def test_manual_results_are_excluded(self, tmp_path):
        monitor = PrecisionMonitor(stats_path=str(tmp_path / "stats.json"))
        monitor.record_result([True, True])
        monitor.record_result([False, False], manual=True)
        # Window has only one non-manual result.
        assert monitor.window_precision()["full"] == 100.0

    def test_persistence(self, tmp_path):
        stats_path = str(tmp_path / "stats.json")
        monitor = PrecisionMonitor(stats_path=stats_path)
        monitor.record_result([True, True])

        reloaded = PrecisionMonitor(stats_path=stats_path)
        assert reloaded.global_stats["total_processed"] == 1
        assert reloaded.global_stats["full"] == 1

    def test_drift_alert(self, tmp_path):
        monitor = PrecisionMonitor(
            stats_path=str(tmp_path / "stats.json"),
            min_samples_for_alerts=10,
        )
        for _ in range(15):
            monitor.record_result([True, True])
        for _ in range(10):
            monitor.record_result([False, False])

        alerts = monitor.check_alerts()
        assert alerts  # drift should trigger at least one alert

    def test_per_field_precision(self, tmp_path):
        monitor = PrecisionMonitor(
            stats_path=str(tmp_path / "stats.json"),
            fields=["field_a", "field_b"],
        )
        monitor.record_result([True, False])
        per_field = monitor.per_field_precision()
        assert per_field["field_a"] == 100.0
        assert per_field["field_b"] == 0.0
