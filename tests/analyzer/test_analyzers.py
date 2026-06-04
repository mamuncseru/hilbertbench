"""
tests/analyzer/test_analyzers.py

Unit tests for Thrust 1 Offline Analyzers.
Mocks the Parquet reading layer to strictly verify the mathematics 
of Barren Plateaus, Expressibility, and Shot Noise.
"""
import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import pandas as pd

from hilbertbench.analyzer.trainability import AnsatzAnalyzer
from hilbertbench.analyzer.expressibility import ExpressibilityAnalyzer
from hilbertbench.analyzer.measurement import MeasurementAnalyzer


# --- Mock Data Generators ---

def mock_trace_table(outcomes_list, shots_list=None):
    """Generates a fake Pandas DataFrame simulating a Parquet trace."""
    if shots_list is None:
        shots_list = [1024] * len(outcomes_list)
        
    data = {
        "status": ["COMPLETED"] * len(outcomes_list),
        "outcome_ref": [f"sha256:mock_{i}" for i in range(len(outcomes_list))],
        "attributes": [json.dumps({"shots": s}) for s in shots_list]
    }
    
    mock_table = MagicMock()
    mock_table.to_pandas.return_value = pd.DataFrame(data)
    return mock_table


@pytest.fixture
def mock_dependencies():
    """Patches the file system and parquet readers."""
    with patch("pathlib.Path.exists", return_value=True), \
         patch("builtins.open"), \
         patch("json.load", return_value={"artifacts": {f"sha256:mock_{i}": {"file_path": f"mock_{i}.npy"} for i in range(100)}}), \
         patch("pyarrow.parquet.read_table") as mock_read_table, \
         patch("numpy.load") as mock_np_load:
        
        yield mock_read_table, mock_np_load


# --- 1. Trainability Analyzer Tests ---

def test_ansatz_analyzer_barren_plateau(mock_dependencies):
    mock_read_table, mock_np_load = mock_dependencies
    
    # 1. Simulate a Barren Plateau (Variance near 0)
    flat_gradients = [np.array([0.00001, -0.00002]), np.array([-0.00001, 0.00001])]
    mock_read_table.return_value = mock_trace_table(flat_gradients)
    mock_np_load.side_effect = flat_gradients

    analyzer = AnsatzAnalyzer("dummy_path")
    result = analyzer.calculate_trajectory_variance()
    
    assert result["status"] == "Barren Plateau Detected"
    assert result["variance"] < 1e-4

    # 2. Simulate a Trainable Ansatz (High Variance)
    healthy_gradients = [np.array([0.5, -0.4]), np.array([-0.6, 0.3])]
    mock_np_load.side_effect = healthy_gradients
    mock_read_table.return_value = mock_trace_table(healthy_gradients)
    
    result2 = analyzer.calculate_trajectory_variance()
    assert result2["status"] == "Trainable"
    assert result2["variance"] > 1e-4


# --- 2. Expressibility Analyzer Tests ---

def test_expressibility_kl_divergence(mock_dependencies):
    mock_read_table, mock_np_load = mock_dependencies
    
    # Simulate an extremely rigid ansatz (always outputs |00>)
    # KL Divergence against Haar will be massive.
    rigid_states = [np.array([1.0, 0.0, 0.0, 0.0]) for _ in range(10)]
    mock_read_table.return_value = mock_trace_table(rigid_states)
    mock_np_load.side_effect = rigid_states

    analyzer = ExpressibilityAnalyzer("dummy_path", num_qubits=2)
    result = analyzer.calculate_kl_divergence(num_bins=10)
    
    assert result["status"] == "Low Expressibility (Rigid Ansatz)"
    assert result["kl_divergence"] > 1.0


# --- 3. Measurement Analyzer Tests ---

def test_measurement_shot_noise_saturation(mock_dependencies):
    mock_read_table, mock_np_load = mock_dependencies
    
    # Simulate 100 shots. Theoretical floor = 0.01.
    # Empirical variance is 0.011 (barely above noise floor).
    outcomes = [np.array([0.1]), np.array([0.1]), np.array([-0.1])]
    shots_list = [100, 100, 100]
    
    mock_read_table.return_value = mock_trace_table(outcomes, shots_list)
    # Force variance calculations to roughly match shot noise
    mock_np_load.side_effect = [np.random.normal(0, np.sqrt(0.011), 100) for _ in range(3)]

    analyzer = MeasurementAnalyzer("dummy_path")
    result = analyzer.analyze_shot_noise_saturation()
    
    assert result["mean_theoretical_floor"] == 0.01
    assert "Shot Noise Dominated" in result["status"] or "Marginal" in result["status"]