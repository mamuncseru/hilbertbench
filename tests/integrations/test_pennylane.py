"""
tests/integrations/test_pennylane.py

Verifies the dynamic proxy integration for PennyLane.
Tests that strict ML type-checks are preserved and synchronous 
executions are correctly logged as single, unified spans.
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from hilbertbench.integrations.pennylane import HilbertPennyLaneDeviceProxy
from hilbertbench.models import Kind, SpanStatus
from hilbertbench.recorder.tape import HilbertTape


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def runs_dir(tmp_path: Path) -> Path:
    runs = tmp_path / "runs"
    runs.mkdir()
    return runs


class MockPennyLaneDeviceBase:
    """A dummy base class to test dynamic inheritance (isinstance checks)."""
    pass


@pytest.fixture
def mock_pl_device():
    class RealMockDevice(MockPennyLaneDeviceBase):
        def __init__(self):
            self.short_name = "default.qubit.mock"
            self.wires = 2
            
        def execute(self, tapes, execution_config=None):
            # Mock returning a list of numpy arrays (like PennyLane does)
            return [[0.1, 0.9] for _ in tapes]

    return RealMockDevice()


@pytest.fixture
def mock_tapes():
    # Mocking qml.tape.QuantumTape objects
    tape1 = MagicMock()
    tape1.operations = ["RX(0.5, wires=[0])"]
    tape1.measurements = ["expval(PauliZ(wires=[0]))"]
    return [tape1]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPennyLaneProxyTransparency:
    
    def test_dynamic_inheritance(self, runs_dir: Path, mock_pl_device):
        """Crucial for PennyLane QNodes: The proxy must pass isinstance()."""
        with HilbertTape(runs_dir) as tape:
            proxy = HilbertPennyLaneDeviceProxy(mock_pl_device, tape)
            
            assert isinstance(proxy, MockPennyLaneDeviceBase)
            assert proxy.short_name == "default.qubit.mock"
            assert proxy.wires == 2


class TestPennyLaneExecutionLifecycle:

    def test_synchronous_execution_span(self, runs_dir: Path, mock_pl_device, mock_tapes):
        """PennyLane evaluates synchronously. We should get exactly ONE span."""
        with HilbertTape(runs_dir) as hb_tape:
            proxy = HilbertPennyLaneDeviceProxy(mock_pl_device, hb_tape)
            
            # Execute synchronously
            results = proxy.execute(mock_tapes)
            
            assert len(results) == 1
            assert results[0] == [0.1, 0.9]
            
        # Verify exactly ONE span was created (unlike Qiskit's two async spans)
        spans_lines = (hb_tape.dir_path / "events.jsonl").read_text().splitlines()
        assert len(spans_lines) == 1
        
        span = json.loads(spans_lines[0])
        assert span["status"] == SpanStatus.COMPLETED.value
        assert span["backend_id"] == "default.qubit.mock"
        
        # Verify 4 events (REQUEST, EXECUTE_STARTED, EXECUTE_COMPLETED, RESULT)
        assert len(span["events"]) == 4
        assert span["events"][1]["event_type"] == "DEVICE_EXECUTE_STARTED"
        assert span["events"][1]["attributes"]["num_tapes"] == 1


class TestPennyLaneExceptionVisibility:

    def test_synchronous_exception_handling(self, runs_dir: Path, mock_tapes):
        """Verifies INV-007 for synchronous failures."""
        
        class MockQMLGradientError(Exception):
            pass
            
        class CrashingMockDevice(MockPennyLaneDeviceBase):
            def __init__(self):
                self.short_name = "crashing.device"
            def execute(self, tapes):
                raise MockQMLGradientError("Parameter shift failed")
                
        crashing_device = CrashingMockDevice()

        with pytest.raises(MockQMLGradientError, match="Parameter shift"):
            with HilbertTape(runs_dir) as hb_tape:
                proxy = HilbertPennyLaneDeviceProxy(crashing_device, hb_tape)
                proxy.execute(mock_tapes)
                
        # Verify the span caught the error
        spans_lines = (hb_tape.dir_path / "events.jsonl").read_text().splitlines()
        assert len(spans_lines) == 1
        
        failed_span = json.loads(spans_lines[0])
        assert failed_span["status"] == SpanStatus.FAILED.value
        
        error_event = next(e for e in failed_span["events"] if e["event_type"] == "ERROR")
        assert error_event["attributes"]["exception_type"] == "MockQMLGradientError"