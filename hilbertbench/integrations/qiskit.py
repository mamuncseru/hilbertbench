"""
hilbertbench/integrations/qiskit.py

Transparent proxy integration for Qiskit.
Wraps standard Backends and Jobs to intercept execution data without 
altering the user's code structure or async execution flows.
"""

import json
import tempfile
from pathlib import Path
from typing import Any, List, Union

from qiskit import QuantumCircuit
from qiskit.providers import Backend, Job
from qiskit.qasm3 import dumps as qasm3_dumps

from hilbertbench.models import Kind, Encoding, Compression
from hilbertbench.recorder.tape import HilbertTape


class HilbertQiskitJobProxy:
    """
    Transparently wraps a qiskit Job.
    Delays recording the outcome until the user explicitly calls .result(),
    perfectly mirroring the real-world async nature of quantum hardware.
    """

    def __init__(self, real_job: Job, tape: HilbertTape, payload_ref: str):
        self._job = real_job
        self._tape = tape
        self._payload_ref = payload_ref
        
        # We store the underlying job ID as a tag, not a reproducible fact
        self._job_id = real_job.job_id()

    def __getattr__(self, name: str) -> Any:
        """Forward all non-intercepted calls to the real Job."""
        return getattr(self._job, name)

    def result(self, *args, **kwargs) -> Any:
        """
        Intercepts the result request. Opens a span to record the exact
        moment the data is retrieved from the hardware/simulator.
        """
        backend_name = self._job.backend().name if self._job.backend() else "unknown"
        
        with self._tape.execution_span(
            payload_ref=self._payload_ref, 
            backend_id=backend_name
        ) as handle:
            
            handle.add_event("JOB_RESULT_REQUESTED", {"qiskit_job_id": self._job_id})
            
            # 1. Execute the real fetch (Observer Effect)
            result = self._job.result(*args, **kwargs)
            
            handle.add_event("JOB_RESULT_RECEIVED")

            # 2. Serialize and hash the outcome
            result_dict = result.to_dict()
            
            # Write to a temp file to attach to the tape
            with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as tmp:
                json.dump(result_dict, tmp)
                tmp_path = Path(tmp.name)

            try:
                outcome_ref = self._tape.attach_artifact(
                    src_path=tmp_path,
                    kind=Kind.execution_outcome,
                    encoding=Encoding.json,
                    producer="qiskit"
                )
                handle.outcome_ref = outcome_ref
            finally:
                # Cleanup temp file
                if tmp_path.exists():
                    tmp_path.unlink()

            return result


class HilbertQiskitBackendProxy:
    """
    Transparently wraps a qiskit Backend (V1 or V2).
    Intercepts the .run() method to capture the circuit payload.
    """

    def __init__(self, real_backend: Backend, tape: HilbertTape):
        self._backend = real_backend
        self._tape = tape
        self._backend_name = getattr(real_backend, "name", "unknown")

    def __getattr__(self, name: str) -> Any:
        """Forward all other calls (like .configuration(), .properties()) to the real backend."""
        return getattr(self._backend, name)

    def run(self, run_input: Union[QuantumCircuit, List[QuantumCircuit]], **options) -> HilbertQiskitJobProxy:
        """
        Intercepts circuit submission. Serializes the circuit to OpenQASM 3.0,
        hashes it, and submits the job.
        """
        # 1. Serialize the payload
        # Standardizing to a list for unified serialization
        circuits = [run_input] if isinstance(run_input, QuantumCircuit) else run_input
        
        # We serialize to QASM3 for standard physical representation
        qasm_payload = "\n---\n".join([qasm3_dumps(c) for c in circuits])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".qasm", mode="w", encoding="utf-8") as tmp:
            tmp.write(qasm_payload)
            tmp_path = Path(tmp.name)
            
        try:
            payload_ref = self._tape.attach_artifact(
                src_path=tmp_path,
                kind=Kind.circuit_qasm,
                encoding=Encoding.openqasm,
                producer="qiskit"
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # 2. Record the submission span
        with self._tape.execution_span(
            payload_ref=payload_ref, 
            backend_id=self._backend_name
        ) as handle:
            
            handle.add_event("JOB_SUBMISSION_STARTED", {"options": str(options)})
            
            # Execute the real submission
            real_job = self._backend.run(run_input, **options)
            
            handle.add_event("JOB_SUBMISSION_COMPLETED", {"qiskit_job_id": real_job.job_id()})

        # 3. Return the wrapped job so we can catch the result later
        return HilbertQiskitJobProxy(real_job, self._tape, payload_ref)