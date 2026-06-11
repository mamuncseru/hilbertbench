"""
tests/analysis/test_builtin_analyzers.py

Tests for the function-based analysis layer (hilbertbench.analysis).
Traces are built deterministically with the tape; no quantum execution.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.trace import HilbertTrace
from hilbertbench.analysis import detect_barren_plateau, shot_noise_ratio, summary


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def build_trace(runs, outcomes, shots=None, precision=None):
    """One span per outcome; optional shots/precision on EXECUTION_COMPLETED."""
    dummy = "sha256:" + "a" * 64
    with HilbertTape(runs) as tape:
        for o in outcomes:
            with tape.execution_span(payload_ref=dummy) as span:
                span.outcome_ref = span.attach_inline(
                    json.dumps(o), kind="execution_outcome", encoding="json"
                )
                attrs = {}
                if shots is not None:
                    attrs["shots"] = shots
                if precision is not None:
                    attrs["precision"] = precision
                if attrs:
                    span.add_event("EXECUTION_COMPLETED", attrs)
    return tape.dir_path


# ── detect_barren_plateau ─────────────────────────────────────────────────────

class TestBarrenPlateau:

    def test_trainable(self, runs):
        r = detect_barren_plateau(build_trace(runs, [0.9, -0.8, 0.7, -0.6, 0.5]))
        assert r["status"] == "Trainable"
        assert r["variance"] > 0.005
        assert r["num_evaluations"] == 5

    def test_barren_plateau(self, runs):
        r = detect_barren_plateau(build_trace(runs, [0.0001, -0.0001, 0.0, 0.0001, -0.0001]))
        assert r["status"] == "Barren Plateau Detected"
        assert r["variance"] < 0.005

    def test_insufficient_data(self, runs):
        # span with no numeric outcome (counts dict)
        dummy = "sha256:" + "b" * 64
        with HilbertTape(runs) as tape:
            with tape.execution_span(payload_ref=dummy) as span:
                span.outcome_ref = span.attach_inline(
                    json.dumps({"meas": {"counts": {"00": 10}}}),
                    kind="execution_outcome", encoding="json",
                )
        r = detect_barren_plateau(tape.dir_path)
        assert r["status"] == "Insufficient Data"
        assert r["variance"] is None
        assert r["num_evaluations"] == 0

    def test_custom_threshold(self, runs):
        run_dir = build_trace(runs, [0.1, -0.1, 0.1, -0.1])  # variance ~0.01
        loose = detect_barren_plateau(run_dir, threshold=0.001)
        strict = detect_barren_plateau(run_dir, threshold=0.05)
        assert loose["status"] == "Trainable"
        assert strict["status"] == "Barren Plateau Detected"

    def test_accepts_path_and_trace_object(self, runs):
        run_dir = build_trace(runs, [0.5, -0.5, 0.3])
        from_path = detect_barren_plateau(run_dir)
        from_obj = detect_barren_plateau(HilbertTrace(run_dir))
        assert from_path["variance"] == from_obj["variance"]

    def test_variance_matches_numpy(self, runs):
        vals = [0.5, -0.3, 0.8, -0.6, 0.1, 0.9]
        r = detect_barren_plateau(build_trace(runs, vals))
        assert r["variance"] == pytest.approx(np.var(vals))


# ── shot_noise_ratio ──────────────────────────────────────────────────────────

class TestShotNoise:

    def test_with_recorded_shots(self, runs):
        # high trajectory variance, low shots → signal clear
        r = shot_noise_ratio(build_trace(runs, [0.9, -0.8, 0.7, -0.6], shots=100))
        assert r["mean_shots"] == 100
        assert r["theoretical_floor"] == pytest.approx(0.01)
        assert r["estimated_snr"] is not None
        assert "Signal Clear" in r["status"]

    def test_shot_noise_dominated(self, runs):
        # tiny trajectory variance, many shots → buried in noise
        r = shot_noise_ratio(build_trace(runs, [0.001, -0.001, 0.001, -0.001], shots=10))
        assert r["estimated_snr"] < 1.5
        assert "Shot Noise Dominated" in r["status"]

    def test_no_shots_recorded(self, runs):
        r = shot_noise_ratio(build_trace(runs, [0.5, -0.5, 0.3]))  # no shots tagged
        assert r["theoretical_floor"] is None
        assert "not recorded" in r["status"]
        assert r["empirical_variance"] is not None  # still computed

    def test_default_shots_fallback(self, runs):
        r = shot_noise_ratio(build_trace(runs, [0.5, -0.5, 0.3]), default_shots=1024)
        assert r["mean_shots"] == 1024
        assert r["estimated_snr"] is not None
        assert r["shots_source"] == "default"

    def test_precision_fallback(self, runs):
        # estimator runs record target precision, not shots; the floor
        # is precision^2 (here 0.1^2 = 0.01 → ~100 effective shots)
        r = shot_noise_ratio(
            build_trace(runs, [0.9, -0.8, 0.7, -0.6], precision=0.1)
        )
        assert r["shots_source"] == "precision"
        assert r["theoretical_floor"] == pytest.approx(0.01)
        assert r["mean_shots"] == pytest.approx(100.0)
        assert "Signal Clear" in r["status"]

    def test_recorded_shots_win_over_precision(self, runs):
        r = shot_noise_ratio(
            build_trace(runs, [0.9, -0.8, 0.7], shots=64, precision=0.1)
        )
        assert r["shots_source"] == "recorded"
        assert r["mean_shots"] == 64

    def test_insufficient_data(self, runs):
        r = shot_noise_ratio(build_trace(runs, [0.5]))  # < 2 outcomes
        assert r["status"] == "Insufficient Data"


# ── summary ───────────────────────────────────────────────────────────────────

class TestSummary:

    def test_combined_report(self, runs):
        s = summary(build_trace(runs, [0.9, -0.8, 0.7, -0.6], shots=50))
        assert s["trace"]["status"] == "SEALED_SUCCESS"
        assert s["trace"]["num_spans"] == 4
        assert "status" in s["trainability"]
        assert "status" in s["measurement"]

    def test_summary_accepts_path(self, runs):
        s = summary(build_trace(runs, [0.1, 0.2, 0.3]))
        assert s["trainability"]["num_evaluations"] == 3


# ── Extensibility demonstration ───────────────────────────────────────────────

class TestCustomAnalysis:

    def test_user_can_write_own_analysis(self, runs):
        """A user composes their own diagnostic on the same trace API."""
        run_dir = build_trace(runs, [0.1, 0.5, 0.9, 0.95, 0.97])
        trace = HilbertTrace(run_dir)

        # custom: is the outcome trajectory monotonically increasing?
        outcomes = trace.numeric_outcomes()
        is_monotonic = bool(np.all(np.diff(outcomes) >= 0))
        assert is_monotonic is True

        # custom metric alongside a built-in
        bp = detect_barren_plateau(trace)
        step_sizes = np.abs(np.diff(outcomes))
        assert bp["status"] == "Trainable"
        assert step_sizes.max() > 0
