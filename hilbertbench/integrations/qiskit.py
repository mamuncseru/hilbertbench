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
from qiskit.primitives import BaseEstimatorV2, StatevectorEstimator

import copy
import tempfile
import json
import os
import numpy as np
from qiskit import qasm3

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
    



class HilbertEstimatorProxy(BaseEstimatorV2):
    """
    A pure Python wrapper that perfectly mimics a Qiskit V2 primitive.
    Safely intercepts batched gradient evaluations, serializes circuits and 
    expectation values, and commits them to the HilbertBench flight recorder.
    """
    def __init__(self, tape):
        super().__init__()
        self.tape = tape
        self.real_estimator = StatevectorEstimator()

    @property
    def options(self):
        # QML explicitly checks options during gradient generation
        return self.real_estimator.options

    def __deepcopy__(self, memo):
        # Prevent Qiskit from duplicating the tape and closing the stream
        new_proxy = HilbertEstimatorProxy(self.tape)
        new_proxy.real_estimator = copy.deepcopy(self.real_estimator, memo)
        return new_proxy

    def run(self, pubs, **kwargs):
        # 1. Execute natively and wait for the results synchronously
        job = self.real_estimator.run(pubs, **kwargs)
        res = job.result()

        # 2. Log every single circuit in the batched gradient array
        for i, pub in enumerate(pubs):
            try:
                # --- FIX 1: Use the correct internal attribute (_closed) ---
                if getattr(self.tape, "_closed", True):
                    continue

                # Safely extract the circuit from the V2 Pub format
                circuit = pub[0] if isinstance(pub, tuple) else getattr(pub, 'circuit', pub)
                
                # Serialize the quantum circuit to OpenQASM 3.0
                try:
                    from qiskit import qasm3
                    qasm_str = qasm3.dumps(circuit)
                except Exception as e:
                    qasm_str = f"// QASM Serialization Failed: {e}"

                # Write QASM to temporary file
                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".qasm", encoding="utf-8") as f_qasm:
                    f_qasm.write(qasm_str)
                    tmp_qasm_path = f_qasm.name

                # --- FIX 2: Attach the QASM artifact FIRST to get the true payload hash ---
                qasm_hash = self.tape.attach_artifact(
                    src_path=tmp_qasm_path, 
                    kind="circuit_qasm", 
                    encoding="openqasm"
                )
                os.remove(tmp_qasm_path)

                # Open the span with the correct payload_ref
                with self.tape.execution_span(payload_ref=qasm_hash, backend_id="qiskit_estimator_v2") as span:
                    
                    # Extract expectation values and serialize to JSON
                    evs_list = np.array(res[i].data.evs).tolist()
                    evs_str = json.dumps(evs_list)

                    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json", encoding="utf-8") as f_json:
                        f_json.write(evs_str)
                        tmp_json_path = f_json.name

                    # Attach physical artifacts to the trace catalog
                    span.outcome_ref = self.tape.attach_artifact(
                        src_path=tmp_json_path, 
                        kind="execution_outcome",
                        encoding="json"
                    )
                    
                    span.add_event("EXECUTION_COMPLETED", {"shots": 1024, "batch_index": i})
                    os.remove(tmp_json_path)

            except Exception as e:
                print(f"  [HilbertBench] Warning: Failed to record span on Batch {i}: {e}")

        # Return the native job back to the optimizer
        return job
    
import os
import copy
import json
import tempfile
import numpy as np

from qiskit.primitives import BaseEstimatorV2

class HilbertEstimatorProxyIBM(BaseEstimatorV2):
    """
    A pure Python wrapper that perfectly mimics a Qiskit V2 primitive.
    Accepts any V2 Estimator (Local or Cloud) via dependency injection.
    """
    def __init__(self, real_estimator, tape):
        super().__init__()
        self.real_estimator = real_estimator
        self.tape = tape

    @property
    def options(self):
        return self.real_estimator.options

    def __deepcopy__(self, memo):
        # Prevent QML from duplicating the tape and corrupting the stream
        return HilbertEstimatorProxyIBM(
            real_estimator=copy.deepcopy(self.real_estimator, memo),
            tape=self.tape
        )

    def run(self, pubs, **kwargs):
        # 1. Execute natively (On IBM Cloud) and wait for the results
        job = self.real_estimator.run(pubs, **kwargs)
        res = job.result()

        # 2. Log every single circuit in the batched gradient array
        for i, pub in enumerate(pubs):
            try:
                if getattr(self.tape, "_closed", True):
                    continue

                circuit = pub[0] if isinstance(pub, tuple) else getattr(pub, 'circuit', pub)
                
                try:
                    from qiskit import qasm3
                    qasm_str = qasm3.dumps(circuit)
                except Exception as e:
                    qasm_str = f"// QASM Serialization Failed: {e}"

                with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".qasm", encoding="utf-8") as f_qasm:
                    f_qasm.write(qasm_str)
                    tmp_qasm_path = f_qasm.name

                qasm_hash = self.tape.attach_artifact(
                    src_path=tmp_qasm_path, 
                    kind="circuit_qasm", 
                    encoding="openqasm"
                )
                os.remove(tmp_qasm_path)

                # Tag the backend ID dynamically (e.g., 'ibm_marrakesh')
                backend_name = "unknown_backend"
                if hasattr(self.real_estimator, "backend"):
                    backend_name = getattr(self.real_estimator.backend, "name", "unknown_backend")

                with self.tape.execution_span(payload_ref=qasm_hash, backend_id=backend_name) as span:
                    evs_list = np.array(res[i].data.evs).tolist()
                    evs_str = json.dumps(evs_list)

                    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".json", encoding="utf-8") as f_json:
                        f_json.write(evs_str)
                        tmp_json_path = f_json.name

                    span.outcome_ref = self.tape.attach_artifact(
                        src_path=tmp_json_path, 
                        kind="execution_outcome",
                        encoding="json"
                    )
                    
                    span.add_event("EXECUTION_COMPLETED", {"shots": 1024, "batch_index": i})
                    os.remove(tmp_json_path)

            except Exception as e:
                print(f"  [HilbertBench] Warning: Failed to record span on Batch {i}: {e}")

        return job
    
