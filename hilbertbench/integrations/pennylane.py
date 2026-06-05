#!/usr/bin/env python
#
# file: hilbertbench/integrations/pennylane.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Transparent proxy integration for PennyLane. Dynamically wraps
# qml.Device or qml.devices.Device to intercept synchronous executions
# without breaking JAX/Torch/Autograd pipelines.
# Adheres to INV-001 (no circuit re-execution) and INV-007 (no silent
# failures).
#------------------------------------------------------------------------------

# import system modules
#
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

# import third-party modules
#
import numpy as np

# import hilbertbench modules
#
from hilbertbench.models import Encoding, Kind
from hilbertbench.recorder.tape import HilbertTape

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# matches numeric literals (ints, floats, scientific notation, optional sign)
#
_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")

# matches gate-parameter groups: the (...) following a gate name
#
_PARENS_RE = re.compile(r"\(([^)]*)\)")

# inline storage threshold in bytes; larger outcomes fall back to .npy
#
_INLINE_BYTES = 65_536

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _qasm_to_template(qasm: str) -> str:
    """
    function: _qasm_to_template

    arguments:
     qasm: a concrete OpenQASM string with baked parameter values

    return:
     a structural template string with positional placeholders

    description:
     Converts concrete OpenQASM (with baked parameter values) into a
     structural template by replacing every numeric literal inside gate
     parentheses with a positional placeholder (_p0, _p1, ...).

     The template is identical across all steps of a training run (only
     parameter values change), so content-addressed storage deduplicates
     it to a single file. The concrete values are stored separately as
     the 'parameters' inline artifact, making the full circuit
     reconstructible.

     Wire indices like q[0] live inside square brackets, never
     parentheses, so they are never touched.
    """

    # use a mutable list as a counter so the nested closure can write it
    #
    counter = [0]

    def repl_parens(match: re.Match) -> str:
        def repl_num(_: re.Match) -> str:
            idx = counter[0]
            counter[0] += 1
            return f"_p{idx}"
        return "(" + _NUM_RE.sub(repl_num, match.group(1)) + ")"

    # exit gracefully
    #
    return _PARENS_RE.sub(repl_parens, qasm)
#
# end of function


def _pennylane_circuit_template(pl_tape: Any) -> tuple[str, bool]:
    """
    function: _pennylane_circuit_template

    arguments:
     pl_tape: a PennyLane tape object

    return:
     a tuple of (text, is_qasm) where:
      is_qasm=True  — text is a templated OpenQASM circuit suitable
                      for the file store (deduplicates across steps)
      is_qasm=False — text is an operations repr fallback stored as
                      a generic_blob inline artifact; used when QASM
                      export is unsupported (e.g. mid-circuit
                      measurements, qml.state())

    description:
     Produces a circuit representation for one PennyLane tape. Attempts
     QASM export via qml.to_openqasm and templates the result; falls
     back to the operations repr on any failure.
    """

    # attempt QASM export and template
    #
    try:
        import pennylane as qml
        concrete = qml.to_openqasm(pl_tape)
        return _qasm_to_template(concrete), True
    except Exception:
        pass

    # exit gracefully — fallback to operations repr
    #
    return str(pl_tape.operations), False
#
# end of function


def _serialize_pl_outcome(result: Any) -> str:
    """
    function: _serialize_pl_outcome

    arguments:
     result: any PennyLane measurement result

    return:
     a JSON string representation of the result

    description:
     Serialises any PennyLane measurement result to a JSON string.
     Handles the four common return types:
      - expval / var  : scalar float        -> bare number string
      - probs         : 1-D float array     -> JSON array
      - counts        : dict[str, int64]    -> JSON object (int64->int)
      - sample        : 2-D int array       -> JSON array of arrays
      - state         : complex array       -> [[re, im], ...] form
    """

    # handle counts dict separately — values are numpy.int64
    #
    if isinstance(result, dict):
        return json.dumps({str(k): int(v) for k, v in result.items()})

    # convert to numpy array for uniform handling
    #
    try:
        import pennylane.math as qml_math
        arr = (
            qml_math.toarray(result)
            if hasattr(result, "shape")
            else np.array(result)
        )
    except Exception:
        arr = np.array(result)

    # complex arrays (state / density_matrix) -> [[re, im], ...]
    #
    if np.iscomplexobj(arr):
        return json.dumps(
            [[float(x.real), float(x.imag)] for x in arr.flat]
        )

    # exit gracefully — real arrays and scalars
    #
    return json.dumps(arr.tolist())
#
# end of function


def HilbertPennyLaneDeviceProxy(
    real_device: Any,
    tape: HilbertTape,
) -> Any:
    """
    function: HilbertPennyLaneDeviceProxy

    arguments:
     real_device: the PennyLane device object to wrap
     tape:        the HilbertTape to record spans into

    return:
     a proxy object that inherits from the real device's exact class

    description:
     Dynamically creates a wrapper that inherits from the real device's
     exact class. This bypasses PennyLane's strict QNode isinstance()
     checks while still intercepting core execution methods.

     Records one span per tape (matching the Qiskit one-span-per-PUB
     pattern), capturing circuit structure, parameter bindings,
     observables, and outcomes. Adheres to INV-001 (passive only) and
     INV-007 (errors recorded and re-raised).
    """

    class PennyLaneDeviceWrapper(real_device.__class__):
        """
        Class: PennyLaneDeviceWrapper

        description:
         Inner proxy class dynamically created to inherit from the real
         device's class. Intercepts execute() and batch_execute() via
         _intercept_execution(), forwarding all other attribute access
         to the original device.
        """

        def __getattribute__(self, name: str) -> Any:
            """
            method: __getattribute__

            arguments:
             name: the attribute name being accessed

            return:
             the attribute from self for intercepted names, or from the
             wrapped original device for all others

            description:
             Selectively intercepts the attributes that belong to this
             proxy; forwards everything else to the real device.
            """

            # intercept only the proxy's own attributes
            #
            _own = {
                "execute",
                "batch_execute",
                "_hb_tape",
                "_original_dev",
                "__class__",
                "_intercept_execution",
            }
            if name in _own:
                return object.__getattribute__(self, name)
            return getattr(
                object.__getattribute__(self, "_original_dev"), name
            )
        #
        # end of method

        def _intercept_execution(
            self,
            method_name: str,
            tapes,
            *args,
            **kwargs,
        ) -> Any:
            """
            method: _intercept_execution

            arguments:
             method_name: 'execute' or 'batch_execute'
             tapes:       the list of PennyLane tapes to execute
             *args:       forwarded to the real device method
             **kwargs:    forwarded to the real device method

            return:
             the list of results from the real device method

            description:
             Executes the full batch on the real device and records one
             span per tape. If the batch itself raises, a single FAILED
             span is recorded before re-raising (INV-007). Per-tape
             recording failures are printed as warnings and do not
             interrupt processing.
            """

            tape_engine = self._hb_tape
            backend_name = getattr(
                self._original_dev, "short_name", "unknown_pennylane_dev"
            )
            real_method = getattr(self._original_dev, method_name)

            # execute the full batch; record a single FAILED span on
            # batch-level error so INV-007 is honoured
            #
            try:
                results = real_method(tapes, *args, **kwargs)
            except Exception:
                if (
                    not getattr(tape_engine, "_closed", True)
                    and tapes
                ):
                    circuit_str = str(
                        getattr(tapes[0], "operations", [])
                    )
                    circuit_hash = "sha256:" + hashlib.sha256(
                        circuit_str.encode("utf-8")
                    ).hexdigest()
                    with tape_engine.execution_span(
                        payload_ref=circuit_hash,
                        backend_id=backend_name,
                    ) as span:
                        span.add_event(
                            f"DEVICE_{method_name.upper()}_STARTED",
                            {"num_tapes": len(tapes)},
                        )
                        # tape's __exit__ records ERROR and flushes FAILED
                        raise
                raise

            # record one span per tape
            #
            for i, (pl_tape, result) in enumerate(zip(tapes, results)):
                try:
                    if getattr(tape_engine, "_closed", True):
                        continue

                    # produce circuit template (QASM or ops-repr fallback)
                    #
                    circuit_text, is_qasm = (
                        _pennylane_circuit_template(pl_tape)
                    )

                    if is_qasm:

                        # store templated QASM in the file store;
                        # content-hash deduplicates across training steps
                        #
                        with tempfile.NamedTemporaryFile(
                            delete=False,
                            mode="w",
                            suffix=".qasm",
                            encoding="utf-8",
                        ) as f_qasm:
                            f_qasm.write(circuit_text)
                            tmp_qasm_path = f_qasm.name

                        payload_ref = tape_engine.attach_artifact(
                            src_path=tmp_qasm_path,
                            kind=Kind.circuit_qasm,
                            encoding=Encoding.openqasm,
                            producer="pennylane",
                        )
                        Path(tmp_qasm_path).unlink(missing_ok=True)

                    else:

                        # inline fallback — payload_ref from span record
                        #
                        payload_ref = "sha256:" + hashlib.sha256(
                            circuit_text.encode("utf-8")
                        ).hexdigest()

                    # open the execution span
                    #
                    with tape_engine.execution_span(
                        payload_ref=payload_ref,
                        backend_id=backend_name,
                    ) as span:

                        # attach ops-repr inline for the fallback case
                        #
                        if not is_qasm:
                            span.attach_inline(
                                circuit_text,
                                kind="generic_blob",
                                encoding="plaintext",
                                producer="pennylane",
                            )

                        # emit the start event
                        #
                        span.add_event(
                            f"DEVICE_{method_name.upper()}_STARTED",
                            {
                                "num_tapes":   len(tapes),
                                "batch_index": i,
                            },
                        )

                        # store outcome inline; fall back to .npy for
                        # large statevectors exceeding _INLINE_BYTES
                        #
                        try:
                            outcome_str = _serialize_pl_outcome(result)
                            if len(outcome_str.encode()) <= _INLINE_BYTES:
                                span.outcome_ref = span.attach_inline(
                                    outcome_str,
                                    kind="execution_outcome",
                                    encoding="json",
                                    producer="pennylane",
                                )
                            else:
                                raise ValueError(
                                    "outcome too large for inline storage"
                                )
                        except Exception:
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=".npy"
                            ) as tmp_npy:
                                np.save(
                                    tmp_npy,
                                    np.array(result, dtype=object),
                                    allow_pickle=True,
                                )
                                tmp_npy_path = Path(tmp_npy.name)
                            try:
                                span.outcome_ref = (
                                    tape_engine.attach_artifact(
                                        src_path=tmp_npy_path,
                                        kind=Kind.execution_outcome,
                                        encoding=Encoding.numpy_binary,
                                        producer="pennylane",
                                    )
                                )
                            finally:
                                if tmp_npy_path.exists():
                                    tmp_npy_path.unlink()

                        # store parameter bindings as inline artifact;
                        # trainable_only=False captures gradient-shift
                        # tapes where shifted values are needed
                        #
                        try:
                            params = pl_tape.get_parameters(
                                trainable_only=False
                            )
                            if params:
                                params_str = json.dumps([
                                    float(np.array(p).flat[0])
                                    for p in params
                                ])
                                span.attach_inline(
                                    params_str,
                                    kind="parameters",
                                    encoding="json",
                                    producer="pennylane",
                                )
                        except Exception:
                            pass

                        # store observable strings as inline artifact
                        #
                        try:
                            obs_list = [
                                str(m.obs)
                                for m in pl_tape.measurements
                                if m.obs is not None
                            ]
                            if obs_list:
                                span.attach_inline(
                                    json.dumps(obs_list),
                                    kind="observables",
                                    encoding="json",
                                    producer="pennylane",
                                )
                        except Exception:
                            pass

                        # emit the completion event
                        #
                        span.add_event(
                            "EXECUTION_COMPLETED",
                            {"method": method_name, "batch_index": i},
                        )

                except Exception as e:
                    print(
                        f"  [HilbertBench] Warning: Failed to record "
                        f"span on Tape {i}: {e}"
                    )

            # exit gracefully
            #
            return results
        #
        # end of method

        def execute(self, tapes, *args, **kwargs):
            """
            method: execute

            arguments:
             tapes:    the list of PennyLane tapes to execute
             *args:    forwarded to _intercept_execution
             **kwargs: forwarded to _intercept_execution

            return:
             list of results from the real device

            description:
             Intercepts the execute() method and delegates to
             _intercept_execution for recording.
            """
            return self._intercept_execution(
                "execute", tapes, *args, **kwargs
            )
        #
        # end of method

        def batch_execute(self, tapes, *args, **kwargs):
            """
            method: batch_execute

            arguments:
             tapes:    the list of PennyLane tapes to execute
             *args:    forwarded to _intercept_execution
             **kwargs: forwarded to _intercept_execution

            return:
             list of results from the real device

            description:
             Intercepts the batch_execute() method and delegates to
             _intercept_execution for recording.
            """
            return self._intercept_execution(
                "batch_execute", tapes, *args, **kwargs
            )
        #
        # end of method
    #
    # end of class

    # construct the proxy without calling __init__; inject references
    #
    wrapper = PennyLaneDeviceWrapper.__new__(PennyLaneDeviceWrapper)
    wrapper._original_dev = real_device
    wrapper._hb_tape = tape

    # exit gracefully
    #
    return wrapper
#
# end of function

#
# end of file
