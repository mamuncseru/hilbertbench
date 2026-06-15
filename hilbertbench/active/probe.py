#!/usr/bin/env python
#
# file: hilbertbench/active/probe.py
#
# revision history:
#  20260604 (am): cleaned up to project coding standards
#
# Active Mode — controlled, explicitly-authorized circuit sampling.
#
# Passive recording observes whatever circuits an optimizer chooses to
# run. That is the right data for trainability but cannot measure
# expressibility: to compare an ansatz against the Haar measure you
# need output states under parameters drawn uniformly at random over
# the full parameter space, which a training trajectory never provides.
#
# Active Mode does exactly that: given a parameterized circuit and a
# way to obtain its statevector, it draws num_samples random parameter
# vectors, runs each, and records the resulting statevectors into a
# mode="active" trace. Feed that trace to kl_expressibility.
#
#   from hilbertbench.active import active_probe_qiskit
#   run_dir = active_probe_qiskit(
#       ansatz, num_samples=1000, output_root="runs"
#   )
#
# This is an explicit user action that runs new circuits — never
# invoked automatically. Honoring INV-001 (the passive recorder never
# re-executes circuits).
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

# import third-party modules
#
import numpy as np

# import hilbertbench modules
#
from hilbertbench.models import Encoding, Kind, Mode
from hilbertbench.recorder.tape import HilbertTape

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# statevectors at or below this serialized size are embedded inline;
# larger ones (deep circuits / many qubits) spill to a .npy file
#
_INLINE_BYTES = 65_536

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _serialize_statevector(psi: np.ndarray) -> str:
    """
    function: _serialize_statevector

    arguments:
     psi: a complex numpy array (the statevector)

    return:
     a JSON string in [[re, im], ...] form

    description:
     Serializes a complex statevector to JSON component-by-component.
     The [[re, im], ...] form round-trips exactly through
     _to_statevector in the expressibility analyzer.
    """

    # flatten and serialize real and imaginary parts component-wise
    #
    flat = np.asarray(psi).ravel()

    # exit gracefully
    #
    return json.dumps([[float(x.real), float(x.imag)] for x in flat])
#
# end of function


def probe_expressibility(
    state_fn: Callable[[np.ndarray], np.ndarray],
    num_params: int,
    num_samples: int,
    output_root: Path | str,
    *,
    circuit_qasm: Optional[str] = None,
    param_low: float = 0.0,
    param_high: float = 2.0 * math.pi,
    seed: Optional[int] = None,
    tags: Optional[dict] = None,
    backend_id: str = "active_probe",
) -> Path:
    """
    function: probe_expressibility

    arguments:
     state_fn      callable (theta: np.ndarray) -> complex statevector;
                   must return a 1-D complex array
     num_params    dimension of the parameter vector to sample
     num_samples   number of random parameter draws
     output_root   directory under which the run directory is created
     circuit_qasm  optional QASM string stored once as a circuit_qasm
                   artifact (deduplicates across all steps)
     param_low     lower bound of uniform parameter sampling (default 0)
     param_high    upper bound of uniform parameter sampling
                   (default 2π)
     seed          optional RNG seed for reproducibility
     tags          additional trace tags merged into run tags
     backend_id    backend label written into span records

    return:
     path to the created run directory

    description:
     Records an Active Mode expressibility trace. Draws num_samples
     random parameter vectors uniformly from [param_low, param_high],
     evaluates state_fn on each, and records the statevector inline
     (or in the file store for large states). One span per sample,
     each with an ACTIVE_SAMPLE event carrying the sample index.
    """

    # initialise the RNG and build merged trace tags
    #
    rng = np.random.default_rng(seed)
    run_tags = {"probe": "expressibility"}
    if tags:
        run_tags.update(tags)

    with HilbertTape(output_root, mode=Mode.active, tags=run_tags) as tape:

        # store the ansatz circuit once; fall back to a generic blob
        # descriptor when no QASM is available
        #
        if circuit_qasm is not None:
            with tempfile.NamedTemporaryFile(
                delete=False,
                mode="w",
                suffix=".qasm",
                encoding="utf-8",
            ) as f:
                f.write(circuit_qasm)
                tmp = f.name
            payload_ref = tape.attach_artifact(
                src_path=tmp,
                kind=Kind.circuit_qasm,
                encoding=Encoding.openqasm,
                producer="active_probe",
            )
            os.remove(tmp)
        else:
            descriptor = f"active_probe: num_params={num_params}"
            with tempfile.NamedTemporaryFile(
                delete=False,
                mode="w",
                suffix=".txt",
                encoding="utf-8",
            ) as f:
                f.write(descriptor)
                tmp = f.name
            payload_ref = tape.attach_artifact(
                src_path=tmp,
                kind=Kind.generic_blob,
                encoding=Encoding.plaintext,
                producer="active_probe",
            )
            os.remove(tmp)

        # draw samples and record one span per sample
        #
        for i in range(num_samples):
            theta = rng.uniform(param_low, param_high, size=num_params)
            psi = np.asarray(state_fn(theta)).ravel()

            with tape.execution_span(
                payload_ref=payload_ref,
                backend_id=backend_id,
            ) as span:

                # attach the parameter vector as an inline artifact
                #
                span.attach_inline(
                    json.dumps(theta.tolist()),
                    kind="parameters",
                    encoding="json",
                    producer="active_probe",
                )

                # attach statevector inline; spill to .npy when large
                #
                ser = _serialize_statevector(psi)
                if len(ser.encode()) <= _INLINE_BYTES:
                    span.outcome_ref = span.attach_inline(
                        ser,
                        kind="execution_outcome",
                        encoding="json",
                        producer="active_probe",
                    )
                else:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".npy"
                    ) as fn:
                        np.save(fn, psi)
                        npy_path = Path(fn.name)
                    try:
                        span.outcome_ref = tape.attach_artifact(
                            src_path=npy_path,
                            kind=Kind.execution_outcome,
                            encoding=Encoding.numpy_binary,
                            producer="active_probe",
                        )
                    finally:
                        npy_path.unlink(missing_ok=True)

                # emit the per-sample diagnostic event
                #
                span.add_event(
                    "ACTIVE_SAMPLE", {"sample_index": i}
                )

    # exit gracefully
    #
    return tape.dir_path
#
# end of function


def active_probe_qiskit(
    circuit: Any,
    num_samples: int,
    output_root: Path | str,
    *,
    seed: Optional[int] = None,
    tags: Optional[dict] = None,
) -> Path:
    """
    function: active_probe_qiskit

    arguments:
     circuit:     a parameterized Qiskit QuantumCircuit
     num_samples: number of random parameter draws
     output_root: directory under which the run directory is created
     seed:        optional RNG seed for reproducibility
     tags:        additional trace tags

    return:
     path to the created run directory

    description:
     Active Mode probe for a parameterized Qiskit circuit. Uses exact
     statevector simulation (no shots) — expressibility is a property
     of the state, not of sampling noise.

     The circuit is decomposed before QASM serialization so that
     library ansätze (e.g. RealAmplitudes) expose their underlying
     gates rather than an opaque compound gate. Falls back to the
     un-decomposed form, then to no QASM, on any failure.
    """

    # import Qiskit modules locally to respect INV-004
    #
    from qiskit import qasm3
    from qiskit.quantum_info import Statevector

    # build a state function that binds parameters and returns data
    #
    params = list(circuit.parameters)

    def state_fn(theta: np.ndarray) -> np.ndarray:
        """Bind theta into the circuit and return its statevector data."""
        bound = circuit.assign_parameters(dict(zip(params, theta)))
        return np.asarray(Statevector(bound).data)

    # serialize the decomposed circuit to QASM; fall back gracefully
    #
    try:
        circuit_qasm = qasm3.dumps(circuit.decompose())
    except Exception:
        try:
            circuit_qasm = qasm3.dumps(circuit)
        except Exception:
            circuit_qasm = None

    # exit gracefully
    #
    return probe_expressibility(
        state_fn,
        num_params=len(params),
        num_samples=num_samples,
        output_root=output_root,
        circuit_qasm=circuit_qasm,
        seed=seed,
        tags={"framework": "qiskit", **(tags or {})},
    )
#
# end of function


def active_probe_pennylane(
    circuit_fn: Callable,
    num_qubits: int,
    num_params: int,
    num_samples: int,
    output_root: Path | str,
    *,
    seed: Optional[int] = None,
    tags: Optional[dict] = None,
) -> Path:
    """
    function: active_probe_pennylane

    arguments:
     circuit_fn:  a function (theta) -> applies gates; the wrapper
                  appends qml.state() and runs on default.qubit
     num_qubits:  wire count
     num_params:  parameter-vector dimension
     num_samples: number of random parameter draws
     output_root: directory under which the run directory is created
     seed:        optional RNG seed for reproducibility
     tags:        additional trace tags

    return:
     path to the created run directory

    description:
     Active Mode probe for a PennyLane ansatz. Wraps circuit_fn with
     a default.qubit qml.state() qnode to obtain exact statevectors.
     Generates the QASM template once at the zero-parameter point and
     stores it as a circuit_qasm artifact for the file store to
     deduplicate across all samples.
    """

    # import PennyLane modules locally to respect INV-004; PennyLane is
    # an optional extra, so guide the user if it is not installed
    #
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError(
            "PennyLane is required for the PennyLane active probe. "
            "Install it with: pip install 'hilbertbench[pennylane]'"
        ) from exc
    from hilbertbench.integrations.pennylane import _qasm_to_template

    # build a statevector qnode wrapping the user's circuit function
    #
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)  # type: ignore[untyped-decorator]
    def qnode(theta: Any) -> Any:
        """Execute the user circuit at theta and return the statevector."""
        circuit_fn(theta)
        return qml.state()

    def state_fn(theta: np.ndarray) -> np.ndarray:
        """Return the circuit statevector at theta as a numpy array."""
        return np.asarray(qnode(theta))

    # generate the QASM template from the zero-parameter point;
    # falls back to None on any error (e.g. unsupported gates)
    #
    circuit_qasm: Optional[str] = None
    try:
        qscript = qml.tape.make_qscript(
            lambda: circuit_fn(np.zeros(num_params))
        )()
        circuit_qasm = _qasm_to_template(
            qml.to_openqasm(qscript, wires=range(num_qubits))
        )
    except Exception:
        circuit_qasm = None

    # exit gracefully
    #
    return probe_expressibility(
        state_fn,
        num_params=num_params,
        num_samples=num_samples,
        output_root=output_root,
        circuit_qasm=circuit_qasm,
        seed=seed,
        tags={"framework": "pennylane", **(tags or {})},
    )
#
# end of function

#
# end of file
