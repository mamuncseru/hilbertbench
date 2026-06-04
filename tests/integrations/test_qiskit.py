"""
tests/integrations/test_qiskit.py

Verifies the transparent proxy integration for Qiskit.
Tests that circuits are serialized, spans are split (async mirroring),
and all underlying framework exceptions are properly propagated (INV-007).
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qiskit import QuantumCircuit

from hilbertbench.integrations.qiskit import HilbertQiskitBackendProxy, HilbertQiskitJobProxy
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


@pytest.fixture
def dummy_circuit() -> QuantumCircuit:
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


@pytest.fixture
def mock_qiskit_objects():
    """Provides a mocked backend, job, and result to avoid slow simulation overhead."""
    # Mock Result
    mock_result = MagicMock()
    mock_result.to_dict.return_value = {"success": True, "results": [{"shots": 1024}]}
    
    # Mock Job
    mock_job = MagicMock()
    mock_job.job_id.return_value = "mock-job-id-1234"
    mock_job.result.return_value = mock_result
    mock_job.status.return_value = "DONE"
    
    # Mock Backend
    mock_backend = MagicMock()
    mock_backend.name = "ibm_mock_kyiv"
    mock_backend.run.return_value = mock_job
    
    # Link job back to backend
    mock_job.backend.return_value = mock_backend
    
    return mock_backend, mock_job, mock_result


# ---------------------------------------------------------------------------
# 1. Passthrough & Transparency Tests
# ---------------------------------------------------------------------------

class TestProxyTransparency:
    
    def test_backend_proxy_passthrough(self, runs_dir: Path, mock_qiskit_objects):
        mock_backend, _, _ = mock_qiskit_objects
        mock_backend.configuration.return_value = {"n_qubits": 127}
        
        with HilbertTape(runs_dir) as tape:
            proxy = HilbertQiskitBackendProxy(mock_backend, tape)
            
            # The proxy must perfectly imitate the underlying backend properties
            assert proxy.name == "ibm_mock_kyiv"
            assert proxy.configuration()["n_qubits"] == 127
            # Verify the mock was actually called
            mock_backend.configuration.assert_called_once()

    def test_job_proxy_passthrough(self, runs_dir: Path, mock_qiskit_objects):
        _, mock_job, _ = mock_qiskit_objects
        
        with HilbertTape(runs_dir) as tape:
            job_proxy = HilbertQiskitJobProxy(mock_job, tape, "dummy_ref")
            
            assert job_proxy.job_id() == "mock-job-id-1234"
            assert job_proxy.status() == "DONE"
            mock_job.status.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Async Lifecycle & Artifact Attachment
# ---------------------------------------------------------------------------

class TestAsyncLifecycle:
    
    def test_successful_run_and_result(self, runs_dir: Path, dummy_circuit: QuantumCircuit, mock_qiskit_objects):
        mock_backend, mock_job, mock_result = mock_qiskit_objects
        
        with HilbertTape(runs_dir) as tape:
            backend = HilbertQiskitBackendProxy(mock_backend, tape)
            
            # 1. Trigger the run (SUBMIT SPAN)
            job = backend.run(dummy_circuit, shots=1024)
            
            # Verify backend.run was actually called on the underlying hardware
            mock_backend.run.assert_called_once_with(dummy_circuit, shots=1024)
            
            # 2. Trigger the fetch (RESULT SPAN)
            result = job.result(timeout=60)
            
            # Verify job.result was called
            mock_job.result.assert_called_once_with(timeout=60)
            assert result == mock_result

        # --- Post-Execution Trace Verification ---
        
        # Verify 2 separate spans were created
        spans_lines = (tape.dir_path / "events.jsonl").read_text().splitlines()
        assert len(spans_lines) == 2
        
        submit_span = json.loads(spans_lines[0])
        fetch_span = json.loads(spans_lines[1])
        
        # Verify Submit Span Details
        assert submit_span["status"] == SpanStatus.COMPLETED.value
        assert len(submit_span["events"]) == 4 # REQUEST, STARTED, COMPLETED, RESULT
        assert submit_span["events"][1]["event_type"] == "JOB_SUBMISSION_STARTED"
        assert submit_span["backend_id"] == "ibm_mock_kyiv"
        
        # Verify Fetch Span Details
        assert fetch_span["status"] == SpanStatus.COMPLETED.value
        assert fetch_span["backend_id"] == "ibm_mock_kyiv"
        
        # Verify Artifacts were generated correctly
        catalog = json.loads((tape.dir_path / "catalog.json").read_text())
        assert len(catalog["artifacts"]) == 2
        
        circuit_artifact_id = submit_span["payload_ref"]
        result_artifact_id = fetch_span["outcome_ref"]
        
        assert catalog["artifacts"][circuit_artifact_id]["kind"] == Kind.circuit_qasm.value
        assert catalog["artifacts"][result_artifact_id]["kind"] == Kind.execution_outcome.value


# ---------------------------------------------------------------------------
# 3. Exception Handling (INV-007)
# ---------------------------------------------------------------------------

class TestExceptionVisibility:
    
    def test_run_exception_visibility(self, runs_dir: Path, dummy_circuit: QuantumCircuit, mock_qiskit_objects):
        mock_backend, _, _ = mock_qiskit_objects
        
        # Simulate a crash during circuit translation/submission
        class MockQiskitCompilerError(Exception):
            pass
            
        mock_backend.run.side_effect = MockQiskitCompilerError("Failed to map circuit to hardware topology")
        
        with pytest.raises(MockQiskitCompilerError, match="hardware topology"):
            with HilbertTape(runs_dir) as tape:
                backend = HilbertQiskitBackendProxy(mock_backend, tape)
                backend.run(dummy_circuit, shots=1024)
                
        # Verify the span caught the error before propagating it
        spans_lines = (tape.dir_path / "events.jsonl").read_text().splitlines()
        assert len(spans_lines) == 1
        
        failed_span = json.loads(spans_lines[0])
        assert failed_span["status"] == SpanStatus.FAILED.value
        
        error_event = next(e for e in failed_span["events"] if e["event_type"] == "ERROR")
        assert error_event["attributes"]["exception_type"] == "MockQiskitCompilerError"

    def test_result_exception_visibility(self, runs_dir: Path, dummy_circuit: QuantumCircuit, mock_qiskit_objects):
        mock_backend, mock_job, _ = mock_qiskit_objects
        
        # Simulate a timeout while waiting for an IBM cloud job
        class MockIBMQTimeoutError(Exception):
            pass
            
        mock_job.result.side_effect = MockIBMQTimeoutError("Job 1234 timed out after 300s")
        
        with pytest.raises(MockIBMQTimeoutError):
            with HilbertTape(runs_dir) as tape:
                backend = HilbertQiskitBackendProxy(mock_backend, tape)
                job = backend.run(dummy_circuit) # This succeeds
                job.result() # This fails
                
        spans_lines = (tape.dir_path / "events.jsonl").read_text().splitlines()
        assert len(spans_lines) == 2
        
        submit_span = json.loads(spans_lines[0])
        fetch_span = json.loads(spans_lines[1])
        
        # Submission succeeded
        assert submit_span["status"] == SpanStatus.COMPLETED.value
        # Fetching failed
        assert fetch_span["status"] == SpanStatus.FAILED.value
        
        error_event = next(e for e in fetch_span["events"] if e["event_type"] == "ERROR")
        assert error_event["attributes"]["exception_type"] == "MockIBMQTimeoutError"