"""
hilbertbench/integrations/pennylane.py

Transparent proxy integration for PennyLane.
Dynamically wraps qml.Device or qml.devices.Device to intercept synchronous
executions (and gradient executions) without breaking JAX/Torch/Autograd pipelines.
"""

import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from hilbertbench.models import Kind, Encoding
from hilbertbench.recorder.tape import HilbertTape


def HilbertPennyLaneDeviceProxy(real_device: Any, tape: HilbertTape) -> Any:
    """
    Dynamically creates a wrapper that inherits from the real_device's exact class.
    This bypasses PennyLane's strict QNode isinstance() checks while allowing us
    to intercept the core execution methods.
    """
    
    class PennyLaneDeviceWrapper(real_device.__class__):
        
        def __getattribute__(self, name: str) -> Any:
            """Forward all property accesses to the real device instance."""
            # These are our injected methods/properties (Notice the added helpers here)
            if name in [
                "execute", 
                "batch_execute", 
                "_hb_tape", 
                "_original_dev", 
                "__class__",
                "_intercept_execution",  # <-- Added
                "_serialize_tapes"       # <-- Added
            ]:
                return object.__getattribute__(self, name)
            
            # Everything else goes to the real device
            return getattr(object.__getattribute__(self, "_original_dev"), name)

        def _serialize_tapes(self, tapes) -> str:
            """Safely extracts a textual representation of the quantum operations."""
            # PennyLane tapes contain complex framework-specific tensor objects.
            # We serialize to a safe generic string representation to avoid pickling errors.
            output = []
            for i, t in enumerate(tapes):
                output.append(f"--- Tape {i} ---")
                output.append(str(t.operations))
                output.append(str(t.measurements))
            return "\n".join(output)

        def _intercept_execution(self, method_name: str, tapes, *args, **kwargs) -> Any:
            """Core interception logic for both execute() and batch_execute()."""
            tape_engine = self._hb_tape
            backend_name = getattr(self._original_dev, "short_name", "unknown_pennylane_dev")

            # 1. Serialize and attach the circuit payload
            payload_str = self._serialize_tapes(tapes)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w", encoding="utf-8") as tmp:
                tmp.write(payload_str)
                tmp_path = Path(tmp.name)

            try:
                payload_ref = tape_engine.attach_artifact(
                    src_path=tmp_path,
                    kind=Kind.generic_blob,
                    encoding=Encoding.plaintext,
                    producer="pennylane"
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            # 2. Record the Synchronous Execution Span
            with tape_engine.execution_span(payload_ref=payload_ref, backend_id=backend_name) as handle:
                handle.add_event(f"DEVICE_{method_name.upper()}_STARTED", {"num_tapes": len(tapes)})
                
                # Execute the real underlying hardware/simulator method
                real_method = getattr(self._original_dev, method_name)
                results = real_method(tapes, *args, **kwargs)
                
                handle.add_event(f"DEVICE_{method_name.upper()}_COMPLETED")

                # 3. Serialize and attach the numerical outcome
                # Convert JAX/Torch tensors to numpy safely before saving
                try:
                    import pennylane.math as qml_math
                    safe_results = [qml_math.toarray(r) if hasattr(r, 'shape') else r for r in results]
                except ImportError:
                    safe_results = results # Fallback if pennylane isn't fully imported

                with tempfile.NamedTemporaryFile(delete=False, suffix=".npy") as tmp_npy:
                    # Allow pickle for complex tuples of arrays returned by PL
                    np.save(tmp_npy, np.array(safe_results, dtype=object), allow_pickle=True)
                    tmp_npy_path = Path(tmp_npy.name)

                try:
                    outcome_ref = tape_engine.attach_artifact(
                        src_path=tmp_npy_path,
                        kind=Kind.execution_outcome,
                        encoding=Encoding.numpy_binary,
                        producer="pennylane"
                    )
                    handle.outcome_ref = outcome_ref
                finally:
                    if tmp_npy_path.exists():
                        tmp_npy_path.unlink()

            return results

        def execute(self, tapes, *args, **kwargs):
            return self._intercept_execution("execute", tapes, *args, **kwargs)

        def batch_execute(self, tapes, *args, **kwargs):
            return self._intercept_execution("batch_execute", tapes, *args, **kwargs)

    # Instantiate our dynamic wrapper without calling the real __init__ again
    wrapper = PennyLaneDeviceWrapper.__new__(PennyLaneDeviceWrapper)
    wrapper._original_dev = real_device
    wrapper._hb_tape = tape
    
    return wrapper