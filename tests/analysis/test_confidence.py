"""
tests/analysis/test_confidence.py

Tests for the statistical-uncertainty measures added to the analyzers
(proposal Section 2.6: "reported with statistical uncertainty and confidence
measures, emphasizing transparency over definitive attribution").

Covers the bootstrap_ci helper and the confidence fields on
detect_barren_plateau, shot_noise_ratio, and kl_expressibility.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.analysis import detect_barren_plateau, shot_noise_ratio
from hilbertbench.analysis._util import bootstrap_ci


@pytest.fixture
def runs(tmp_path: Path) -> Path:
    d = tmp_path / "runs"
    d.mkdir()
    return d


def make_trace(runs, outcomes, shots=None):
    dummy = "sha256:" + "a" * 64
    with HilbertTape(runs) as tape:
        for o in outcomes:
            with tape.execution_span(payload_ref=dummy) as s:
                s.outcome_ref = s.attach_inline(
                    json.dumps(float(o)), kind="execution_outcome",
                    encoding="json",
                )
                if shots is not None:
                    s.add_event("EXECUTION_COMPLETED", {"shots": shots})
    return tape.dir_path


# ---------------------------------------------------------------------------
# bootstrap_ci helper
# ---------------------------------------------------------------------------

class TestBootstrapCI:

    def test_ci_brackets_statistic(self):
        rng = np.random.default_rng(0)
        data = rng.normal(0.0, 1.0, 500)
        low, high = bootstrap_ci(data, np.mean, n_boot=500, seed=1)
        assert low < np.mean(data) < high

    def test_degenerate_inputs_return_none(self):
        assert bootstrap_ci(np.array([1.0]), np.var) == (None, None)
        assert bootstrap_ci(np.array([]), np.var) == (None, None)

    def test_n_boot_zero_disables(self):
        data = np.arange(10.0)
        assert bootstrap_ci(data, np.var, n_boot=0) == (None, None)

    def test_reproducible_with_seed(self):
        data = np.random.default_rng(2).normal(size=200)
        a = bootstrap_ci(data, np.var, n_boot=300, seed=7)
        b = bootstrap_ci(data, np.var, n_boot=300, seed=7)
        assert a == b

    def test_wider_ci_for_higher_level(self):
        data = np.random.default_rng(3).normal(size=300)
        lo90, hi90 = bootstrap_ci(data, np.var, n_boot=500, ci=0.90, seed=4)
        lo99, hi99 = bootstrap_ci(data, np.var, n_boot=500, ci=0.99, seed=4)
        assert (hi99 - lo99) >= (hi90 - lo90)


# ---------------------------------------------------------------------------
# detect_barren_plateau confidence
# ---------------------------------------------------------------------------

class TestBarrenPlateauConfidence:

    def test_ci_brackets_variance(self, runs):
        rng = np.random.default_rng(0)
        run = make_trace(runs, rng.normal(0, 0.5, 200))
        r = detect_barren_plateau(run, seed=1)
        lo, hi = r["variance_ci"]
        assert lo < r["variance"] < hi
        assert r["confidence_level"] == 0.95

    def test_clear_trainable_high_confidence(self, runs):
        rng = np.random.default_rng(1)
        run = make_trace(runs, rng.normal(0, 0.5, 200))
        r = detect_barren_plateau(run, seed=1)
        assert r["status"] == "Trainable"
        assert r["verdict_confidence"] == "high"

    def test_clear_barren_high_confidence(self, runs):
        rng = np.random.default_rng(2)
        run = make_trace(runs, rng.normal(0, 0.01, 200))
        r = detect_barren_plateau(run, seed=1)
        assert r["status"] == "Barren Plateau Detected"
        assert r["verdict_confidence"] == "high"

    def test_near_threshold_low_confidence(self, runs):
        # variance engineered to sit right at the 0.005 threshold
        rng = np.random.default_rng(3)
        run = make_trace(runs, rng.normal(0, np.sqrt(0.005), 200))
        r = detect_barren_plateau(run, seed=1)
        assert r["verdict_confidence"] == "low"

    def test_n_boot_zero_skips_ci(self, runs):
        run = make_trace(runs, [0.5, -0.5, 0.3, -0.2, 0.4])
        r = detect_barren_plateau(run, n_boot=0)
        assert r["variance_ci"] == [None, None]
        assert r["verdict_confidence"] is None


# ---------------------------------------------------------------------------
# shot_noise_ratio confidence
# ---------------------------------------------------------------------------

class TestShotNoiseConfidence:

    def test_empirical_variance_ci_present(self, runs):
        run = make_trace(runs, [0.9, -0.8, 0.7, -0.6, 0.5], shots=100)
        r = shot_noise_ratio(run, seed=1)
        lo, hi = r["empirical_variance_ci"]
        assert lo < r["empirical_variance"] < hi
        assert r["confidence_level"] == 0.95

    def test_ci_present_even_without_shots(self, runs):
        run = make_trace(runs, [0.5, -0.5, 0.3, -0.2])
        r = shot_noise_ratio(run, seed=1)
        assert r["empirical_variance_ci"][0] is not None
