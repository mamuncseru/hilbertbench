"""
tests/e2e/test_qiskit_aer.py

End-to-End verification using a REAL Qiskit Aer simulator.
Proves the transparent proxy works with actual quantum execution
without breaking standard QML workflows.
"""
import json
from pathlib import Path

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit_aer import AerSimulator

from hilbertbench.integrations.qiskit import HilbertQiskitBackendProxy
from hilbertbench.recorder.tape import HilbertTape


def test_real_qml_parameterized_circuit(tmp_path: Path):
    """
    Runs a real parameterized circuit (typical of QML/VQE) through
    the AerSimulator, verifying the proxy handles real Qiskit objects.
    """
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # 1. Standard QML Setup (Untouched by our framework)
    theta = Parameter('θ')
    qc = QuantumCircuit(2)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    qc.measure_all()

    # Bind the parameter to a real value (e.g., during a gradient descent step)
    bound_qc = qc.assign_parameters({theta: 1.57}) # Pi/2

    # 2. Get the REAL simulator
    real_backend = AerSimulator()

    # 3. Wrap it in HilbertBench
    with HilbertTape(runs_dir) as tape:
        backend = HilbertQiskitBackendProxy(real_backend, tape)
        
        # Execute the real physics simulation!
        job = backend.run(bound_qc, shots=2000)
        result = job.result()
        
        # Verify the real math happened
        counts = result.get_counts()
        assert sum(counts.values()) == 2000
        # With an Rx(pi/2) and a CX, we expect roughly equal 00 and 11
        assert "00" in counts
        assert "11" in counts

    # 4. Verify HilbertBench successfully captured the physical reality
    spans_lines = (tape.dir_path / "events.jsonl").read_text().splitlines()
    assert len(spans_lines) == 2 # Submit and Fetch spans recorded
    
    catalog = json.loads((tape.dir_path / "catalog.json").read_text())
    artifacts = list(catalog["artifacts"].values())
    assert len(artifacts) == 2
    
    # We can actually verify the physical QASM was written to disk!
    qasm_artifact = next(a for a in artifacts if a["kind"] == "circuit_qasm")
    qasm_path = tape.dir_path / qasm_artifact["file_path"]
    
    assert qasm_path.exists()
    qasm_text = qasm_path.read_text()
    
    # The QASM should contain the compiled parameter value
    assert "rx(1.57)" in qasm_text