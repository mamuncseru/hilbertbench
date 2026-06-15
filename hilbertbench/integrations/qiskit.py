#!/usr/bin/env python
#
# file: hilbertbench/integrations/qiskit.py
#
# revision history:
#  20260610 (am): robust backend resolution + rate-limited calibration
#                 refresh so drifting devices yield a snapshot history
#  20260604 (am): cleaned up to project coding standards
#
# Transparent proxy integration for Qiskit. Wraps standard Backends, Jobs,
# and Estimator/Sampler primitives to intercept execution data without
# altering the user's circuit structure or async execution flows.
# Adheres to INV-001 (no circuit re-execution) and INV-007 (no silent
# failures).
#------------------------------------------------------------------------------

# import system modules
#
import copy
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, List, Optional, Union

# import third-party modules
#
import numpy as np
from qiskit import QuantumCircuit, qasm3
from qiskit.primitives import (
    BaseEstimatorV2,
    BaseSamplerV2,
    StatevectorEstimator,
    StatevectorSampler,
)
from qiskit.providers import Backend, Job
from qiskit.qasm3 import dumps as qasm3_dumps

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

# minimum seconds between calibration re-queries on a live backend;
# keeps the per-execution overhead amortised well below the 5ms budget
# while still catching intra-run device recalibration (drift)
#
CALIBRATION_REFRESH_S = 600.0

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _resolve_backend(primitive: Any) -> Optional[Any]:
    """
    function: _resolve_backend

    arguments:
     primitive: a Qiskit primitive, a Backend, or None

    return:
     the underlying Backend object, or None if not resolvable

    description:
     Locates the backend behind a V2 primitive. Handles the three
     conventions in the wild: a 'backend' property (qiskit
     BackendEstimatorV2/BackendSamplerV2), a 'backend()' bound method
     (qiskit-ibm-runtime primitives), and a private '_backend'
     attribute (qiskit-aer primitives). A Backend passed directly is
     returned as-is. Statevector primitives resolve to None.
    """

    # nothing to resolve
    #
    if primitive is None:
        return None

    # a backend passed directly exposes the Backend API surface
    #
    if hasattr(primitive, "target") or hasattr(primitive, "properties"):
        return primitive

    # the public attribute may be a property value or a bound method
    # (qiskit-ibm-runtime primitives expose backend() as a method)
    #
    backend = getattr(primitive, "backend", None)
    if callable(backend):
        try:
            backend = backend()
        except Exception:
            backend = None
    if backend is not None:
        return backend

    # fall back to the private attribute used by qiskit-aer primitives
    #
    return getattr(primitive, "_backend", None)
#
# end of function


def _serialize_calibration(
    backend: Any,
    refresh: bool = False,
) -> Optional[str]:
    """
    function: _serialize_calibration

    arguments:
     backend: a Qiskit backend object, or None
     refresh: when True, ask the provider to bypass its local cache
              (IBM runtime backends cache properties between calls)

    return:
     a JSON string of calibration data, or None

    description:
     Extracts device calibration (T1, T2, readout error, gate errors)
     from a Qiskit backend as a JSON string. Returns None for ideal
     simulators or any backend that does not expose calibration data.
     BackendProperties datetimes are serialised with default=str.
    """

    # return None for missing backends
    #
    if backend is None:
        return None

    # attempt to fetch BackendProperties, bypassing the provider cache
    # on refresh so calibration drift is actually observable
    #
    try:
        if refresh:
            try:
                props = backend.properties(refresh=True)
            except TypeError:
                props = backend.properties()
        else:
            props = backend.properties()
    except Exception:
        props = None

    # serialise to JSON if properties expose a real dict; anything
    # else (mocks, exotic providers) is rejected rather than recorded
    #
    if props is not None and hasattr(props, "to_dict"):
        try:
            as_dict = props.to_dict()
            if isinstance(as_dict, dict):
                return json.dumps(as_dict, default=str)
        except Exception:
            return None

    # exit gracefully — no calibration available
    #
    return None
#
# end of function


def _resolve_shot_evidence(
    estimator: Any,
    pub: Any,
    run_kwargs: dict,
) -> dict:
    """
    function: _resolve_shot_evidence

    arguments:
     estimator:  the wrapped V2 estimator primitive
     pub:        the PUB being recorded
     run_kwargs: the kwargs the user passed to run()

    return:
     dict with 'shots' and/or 'precision' keys when resolvable,
     empty otherwise

    description:
     Resolves the measurement budget the user requested for a PUB —
     this is intent, so it belongs in the trace. Precision comes from
     the PUB's fourth element, then the run() kwarg, then the
     estimator's default_precision option; shots come from the
     default_shots option (qiskit-ibm-runtime). Without this evidence
     the shot-noise analyzer cannot establish a noise floor for
     estimator-based runs.
    """

    # resolve the requested precision (PUB > run kwarg > option)
    #
    evidence: dict = {}
    precision = None
    if isinstance(pub, tuple) and len(pub) > 3 and pub[3] is not None:
        precision = pub[3]
    elif run_kwargs.get("precision") is not None:
        precision = run_kwargs["precision"]
    options = getattr(estimator, "options", None)
    if precision is None:
        default_precision = getattr(options, "default_precision", None)
        if (
            isinstance(default_precision, float)
            and default_precision > 0
        ):
            precision = default_precision

    # resolve a configured shot count (qiskit-ibm-runtime option)
    #
    default_shots = getattr(options, "default_shots", None)
    if isinstance(default_shots, (int, np.integer)) and default_shots > 0:
        evidence["shots"] = int(default_shots)

    # record the precision when it resolved to a usable number
    #
    if precision is not None:
        try:
            precision = float(precision)
            if precision > 0:
                evidence["precision"] = precision
        except (TypeError, ValueError):
            pass

    # exit gracefully
    #
    return evidence
#
# end of function


def _attach_calibration_json(
    tape: HilbertTape,
    cal_json: str,
) -> None:
    """
    function: _attach_calibration_json

    arguments:
     tape:     the HilbertTape to attach the calibration artifact to
     cal_json: the serialised calibration JSON string

    return:
     none

    description:
     Writes a calibration JSON string into the file store as a
     'calibration_snapshot' artifact. Silently does nothing if the
     tape is closed or the attach fails (recording must never break
     the user's run).
    """

    # skip if tape is already closed
    #
    if getattr(tape, "_closed", True):
        return

    # write to a temp file and attach to the tape
    #
    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            mode="w",
            suffix=".json",
            encoding="utf-8",
        ) as f:
            f.write(cal_json)
            tmp_path = f.name

        tape.attach_artifact(
            src_path=tmp_path,
            kind=Kind.calibration_snapshot,
            encoding=Encoding.json,
            producer="qiskit",
        )
        os.remove(tmp_path)

    except Exception:
        pass
#
# end of function


def _maybe_capture_calibration(
    proxy: Any,
    tape: HilbertTape,
    primitive: Any,
) -> None:
    """
    function: _maybe_capture_calibration

    arguments:
     proxy:     a Hilbert proxy carrying calibration capture state
                (calibration_refresh_s, _cal_next_check, _cal_last_hash)
     tape:      the HilbertTape to attach snapshots to
     primitive: the wrapped primitive or backend to resolve

    return:
     none

    description:
     Rate-limited calibration capture. The backend is queried at most
     once per calibration_refresh_s window, and a new artifact is
     attached only when the calibration content actually changed — so
     a drifting device yields a snapshot history while a stable device
     costs a single artifact. Only device metadata is queried; no
     circuits are executed (INV-001).
    """

    # rate-limit the backend query
    #
    now = time.monotonic()
    if now < proxy._cal_next_check:
        return
    first = proxy._cal_last_hash is None
    proxy._cal_next_check = now + proxy.calibration_refresh_s

    # serialise; bypass the provider cache on re-checks so drift in
    # the device calibration is actually observable
    #
    cal_json = _serialize_calibration(
        _resolve_backend(primitive),
        refresh=not first,
    )
    if not cal_json:
        return

    # attach only when the calibration content changed
    #
    digest = hashlib.sha256(cal_json.encode("utf-8")).hexdigest()
    if digest == proxy._cal_last_hash:
        return
    proxy._cal_last_hash = digest
    _attach_calibration_json(tape, cal_json)
#
# end of function

#------------------------------------------------------------------------------
#
# classes are listed here
#
#------------------------------------------------------------------------------

class HilbertQiskitJobProxy:
    """
    Class: HilbertQiskitJobProxy

    description:
     Transparently wraps a Qiskit Job. Delays recording the outcome
     until the user explicitly calls .result(), mirroring the real-world
     async nature of quantum hardware.
    """

    def __init__(
        self,
        real_job: Job,
        tape: HilbertTape,
        payload_ref: str,
    ) -> None:
        """
        method: constructor

        arguments:
         real_job:    the real Qiskit Job object to wrap
         tape:        the HilbertTape to record spans into
         payload_ref: the artifact hash of the submitted circuit

        return:
         none

        description:
         Stores references to the real job, tape, and circuit payload
         hash. Job ID is captured eagerly as it is always available
         before .result() is called.
        """

        # store the wrapped job and tape references
        #
        self._job = real_job
        self._tape = tape
        self._payload_ref = payload_ref

        # capture job_id eagerly — available without blocking
        #
        self._job_id = real_job.job_id()
    #
    # end of method

    def __getattr__(self, name: str) -> Any:
        """
        method: __getattr__

        arguments:
         name: the attribute name being accessed

        return:
         the attribute from the underlying real job

        description:
         Transparently forwards all unknown attribute accesses to the
         real job object, preserving the original API surface (INV-001).
        """
        return getattr(self._job, name)
    #
    # end of method

    def result(self, *args: Any, **kwargs: Any) -> Any:
        """
        method: result

        arguments:
         *args:   forwarded to the real job's .result()
         **kwargs: forwarded to the real job's .result()

        return:
         the real Qiskit Result object

        description:
         Intercepts .result() to record an outcome span. The result dict
         is written to a temp file, attached to the tape as a JSON
         artifact, then the real Result object is returned unchanged.
        """

        # resolve the backend name for the span record
        #
        backend_name = (
            self._job.backend().name
            if self._job.backend()
            else "unknown"
        )

        # open the outcome span and record result events
        #
        with self._tape.execution_span(
            payload_ref=self._payload_ref,
            backend_id=backend_name,
        ) as handle:

            # record that the result was requested
            #
            handle.add_event(
                "JOB_RESULT_REQUESTED",
                {"qiskit_job_id": self._job_id},
            )
            result = self._job.result(*args, **kwargs)
            handle.add_event("JOB_RESULT_RECEIVED")

            # serialise result dict to a temp file and attach
            #
            result_dict = result.to_dict()
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".json",
                mode="w",
                encoding="utf-8",
            ) as tmp:
                json.dump(result_dict, tmp)
                tmp_path = Path(tmp.name)

            try:
                handle.outcome_ref = self._tape.attach_artifact(
                    src_path=tmp_path,
                    kind=Kind.execution_outcome,
                    encoding=Encoding.json,
                    producer="qiskit",
                )
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

            return result
    #
    # end of method
#
# end of class


class HilbertQiskitBackendProxy:
    """
    Class: HilbertQiskitBackendProxy

    description:
     Transparently wraps a Qiskit Backend (V1 or V2). Intercepts .run()
     to capture the circuit payload as a QASM artifact before submission,
     then wraps the returned Job in a HilbertQiskitJobProxy.
    """

    def __init__(
        self,
        real_backend: Backend,
        tape: HilbertTape,
    ) -> None:
        """
        method: constructor

        arguments:
         real_backend: the real Qiskit Backend to wrap
         tape:         the HilbertTape to record spans into

        return:
         none

        description:
         Stores references to the real backend and tape. The backend
         name is captured eagerly for use in span records.
        """

        # store the wrapped backend and tape
        #
        self._backend = real_backend
        self._tape = tape
        self._backend_name = getattr(real_backend, "name", "unknown")

        # calibration capture state: queries are rate-limited and a
        # snapshot is attached only when the content hash changes
        #
        self.calibration_refresh_s = CALIBRATION_REFRESH_S
        self._cal_next_check = 0.0
        self._cal_last_hash: Optional[str] = None
    #
    # end of method

    def __getattr__(self, name: str) -> Any:
        """
        method: __getattr__

        arguments:
         name: the attribute name being accessed

        return:
         the attribute from the underlying real backend

        description:
         Transparently forwards all unknown attribute accesses to the
         real backend object, preserving the original API surface
         (INV-001).
        """
        return getattr(self._backend, name)
    #
    # end of method

    def run(
        self,
        run_input: Union[QuantumCircuit, List[QuantumCircuit]],
        **options: Any,
    ) -> HilbertQiskitJobProxy:
        """
        method: run

        arguments:
         run_input: a single QuantumCircuit or a list of them
         **options: forwarded to the real backend's .run()

        return:
         a HilbertQiskitJobProxy wrapping the real job

        description:
         Serialises the circuit(s) to OpenQASM 3.0, attaches to the
         tape, then records the job submission span. Returns a proxy
         that intercepts .result() for outcome capture.
        """

        # capture/refresh the calibration snapshot (rate-limited)
        #
        _maybe_capture_calibration(self, self._tape, self._backend)

        # normalise input to a list and serialise to QASM
        #
        circuits = (
            [run_input]
            if isinstance(run_input, QuantumCircuit)
            else run_input
        )
        qasm_payload = "\n---\n".join([qasm3_dumps(c) for c in circuits])

        # write QASM to a temp file and attach to the tape
        #
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".qasm",
            mode="w",
            encoding="utf-8",
        ) as tmp:
            tmp.write(qasm_payload)
            tmp_path = Path(tmp.name)

        try:
            payload_ref = self._tape.attach_artifact(
                src_path=tmp_path,
                kind=Kind.circuit_qasm,
                encoding=Encoding.openqasm,
                producer="qiskit",
            )
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        # record the job submission span
        #
        with self._tape.execution_span(
            payload_ref=payload_ref,
            backend_id=self._backend_name,
        ) as handle:
            handle.add_event(
                "JOB_SUBMISSION_STARTED",
                {"options": str(options)},
            )
            real_job = self._backend.run(run_input, **options)
            handle.add_event(
                "JOB_SUBMISSION_COMPLETED",
                {"qiskit_job_id": real_job.job_id()},
            )

        # exit gracefully — return the job proxy
        #
        return HilbertQiskitJobProxy(real_job, self._tape, payload_ref)
    #
    # end of method
#
# end of class


class HilbertEstimatorProxy(BaseEstimatorV2):  # type: ignore[misc]
    """
    Class: HilbertEstimatorProxy

    description:
     Transparent wrapper around any Qiskit V2 Estimator primitive.
     Works with both local simulators and IBM Cloud hardware via
     dependency injection. Pass real_estimator=None (the default) to
     use StatevectorEstimator locally, or pass any V2 Estimator for
     hardware.

     Usage:
      proxy = HilbertEstimatorProxy(tape)                  # local sim
      proxy = HilbertEstimatorProxy(tape, real_estimator)  # hardware
    """

    def __init__(
        self,
        tape: HilbertTape,
        real_estimator: Optional[BaseEstimatorV2] = None,
    ) -> None:
        """
        method: constructor

        arguments:
         tape:           the HilbertTape to record spans into
         real_estimator: optional V2 Estimator; defaults to
                         StatevectorEstimator for local simulation

        return:
         none

        description:
         Stores the tape and the underlying estimator. Initialises the
         calibration capture state (rate limit + content hash).
        """
        super().__init__()

        # store the tape and underlying estimator
        #
        self.tape = tape
        self.real_estimator = (
            real_estimator
            if real_estimator is not None
            else StatevectorEstimator()
        )

        # calibration capture state: queries are rate-limited and a
        # snapshot is attached only when the content hash changes
        #
        self.calibration_refresh_s = CALIBRATION_REFRESH_S
        self._cal_next_check = 0.0
        self._cal_last_hash: Optional[str] = None
    #
    # end of method

    @property
    def options(self) -> Any:
        """
        method: options

        arguments:
         none

        return:
         the options object from the underlying real estimator
        """
        return self.real_estimator.options
    #
    # end of method

    def __deepcopy__(self, memo: dict) -> "HilbertEstimatorProxy":
        """
        method: __deepcopy__

        arguments:
         memo: the deepcopy memo dict

        return:
         a new HilbertEstimatorProxy wrapping a deep-copied estimator

        description:
         Required by Qiskit optimizers that deepcopy primitives. The
         tape reference is shared (not copied) intentionally, and the
         calibration capture state is carried over so a deepcopy does
         not trigger a redundant backend query.
        """
        clone = HilbertEstimatorProxy(
            tape=self.tape,
            real_estimator=copy.deepcopy(self.real_estimator, memo),
        )
        clone.calibration_refresh_s = self.calibration_refresh_s
        clone._cal_next_check = self._cal_next_check
        clone._cal_last_hash = self._cal_last_hash
        return clone
    #
    # end of method

    def _backend_id(self) -> str:
        """
        method: _backend_id

        arguments:
         none

        return:
         a human-readable backend identifier string

        description:
         Returns the backend name if available, otherwise falls back to
         the estimator class name.
        """

        # prefer the backend's name attribute
        #
        backend = getattr(self.real_estimator, "backend", None)
        if backend is not None:
            return getattr(backend, "name", str(backend))
        return type(self.real_estimator).__name__
    #
    # end of method

    def run(self, pubs: Any, **kwargs: Any) -> Any:
        """
        method: run

        arguments:
         pubs:    list of primitive unified blocs (PUBs) to execute
         **kwargs: forwarded to the real estimator's .run()

        return:
         the real estimator job object

        description:
         Intercepts each PUB after batch execution. For each PUB:
          - Circuit is serialised to QASM and stored in the file store
          - Expectation values are stored as inline outcome artifacts
          - Parameter bindings are stored as inline artifacts
          - Observable strings are stored as inline artifacts
         Failures per-PUB are printed as warnings and do not interrupt
         the batch (INV-007 is honoured at the batch level).
        """

        # capture/refresh the calibration snapshot (rate-limited;
        # handles property-, method-, and private-attr backend access)
        #
        _maybe_capture_calibration(self, self.tape, self.real_estimator)

        # execute the batch via the real estimator
        #
        job = self.real_estimator.run(pubs, **kwargs)
        res = job.result()

        # record one span per PUB
        #
        for i, pub in enumerate(pubs):
            try:
                if getattr(self.tape, "_closed", True):
                    continue

                # extract the circuit from the PUB
                #
                circuit = (
                    pub[0]
                    if isinstance(pub, tuple)
                    else getattr(pub, "circuit", pub)
                )

                # serialise the circuit to QASM for the file store
                #
                try:
                    qasm_str = qasm3.dumps(circuit)
                except Exception as e:
                    qasm_str = f"// QASM Serialization Failed: {e}"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    mode="w",
                    suffix=".qasm",
                    encoding="utf-8",
                ) as f:
                    f.write(qasm_str)
                    tmp_qasm = f.name

                qasm_hash = self.tape.attach_artifact(
                    src_path=tmp_qasm,
                    kind=Kind.circuit_qasm,
                    encoding=Encoding.openqasm,
                )
                os.remove(tmp_qasm)

                # open the execution span
                #
                with self.tape.execution_span(
                    payload_ref=qasm_hash,
                    backend_id=self._backend_id(),
                ) as span:

                    # store expectation values as inline outcome
                    #
                    evs_str = json.dumps(
                        np.array(res[i].data.evs).tolist()
                    )
                    span.outcome_ref = span.attach_inline(
                        evs_str,
                        kind="execution_outcome",
                        encoding="json",
                        producer="qiskit",
                    )

                    # store parameter bindings as inline artifact
                    #
                    if (
                        isinstance(pub, tuple)
                        and len(pub) > 2
                        and pub[2] is not None
                    ):
                        try:
                            span.attach_inline(
                                json.dumps(np.array(pub[2]).tolist()),
                                kind="parameters",
                                encoding="json",
                                producer="qiskit",
                            )
                        except Exception:
                            pass

                    # store observable strings as inline artifact
                    #
                    if (
                        isinstance(pub, tuple)
                        and len(pub) > 1
                        and pub[1] is not None
                    ):
                        try:
                            obs_str = json.dumps([
                                [str(p), complex(c).real, complex(c).imag]
                                for p, c in pub[1].to_list()
                            ])
                            span.attach_inline(
                                obs_str,
                                kind="observables",
                                encoding="json",
                                producer="qiskit",
                            )
                        except Exception:
                            pass

                    # emit the completion event with the measurement
                    # budget the user requested (shots / precision)
                    #
                    attrs = {"batch_index": i}
                    attrs.update(_resolve_shot_evidence(
                        self.real_estimator, pub, kwargs,
                    ))
                    span.add_event("EXECUTION_COMPLETED", attrs)

            except Exception as e:
                print(
                    f"  [HilbertBench] Warning: Failed to record "
                    f"span on Batch {i}: {e}"
                )

        # exit gracefully
        #
        return job
    #
    # end of method
#
# end of class


class HilbertSamplerProxy(BaseSamplerV2):  # type: ignore[misc]
    """
    Class: HilbertSamplerProxy

    description:
     Transparent wrapper around any Qiskit V2 Sampler primitive.
     Captures bitstring measurement outcomes (counts), parameter
     bindings, and circuit structure for every PUB, covering shot-based
     experiments such as QAOA, Grover's search, randomised benchmarking,
     and Bell tests.

     Works with both local simulators and IBM Cloud hardware via
     dependency injection. Pass real_sampler=None (the default) to use
     StatevectorSampler locally, or pass any V2 Sampler for hardware.

     Usage:
      proxy = HilbertSamplerProxy(tape)                # local sim
      proxy = HilbertSamplerProxy(tape, real_sampler)  # hardware
    """

    def __init__(
        self,
        tape: HilbertTape,
        real_sampler: Optional[BaseSamplerV2] = None,
    ) -> None:
        """
        method: constructor

        arguments:
         tape:        the HilbertTape to record spans into
         real_sampler: optional V2 Sampler; defaults to
                       StatevectorSampler for local simulation

        return:
         none

        description:
         Stores the tape and the underlying sampler. Initialises the
         calibration capture state (rate limit + content hash).
        """
        super().__init__()

        # store the tape and underlying sampler
        #
        self.tape = tape
        self.real_sampler = (
            real_sampler
            if real_sampler is not None
            else StatevectorSampler()
        )

        # calibration capture state: queries are rate-limited and a
        # snapshot is attached only when the content hash changes
        #
        self.calibration_refresh_s = CALIBRATION_REFRESH_S
        self._cal_next_check = 0.0
        self._cal_last_hash: Optional[str] = None
    #
    # end of method

    @property
    def options(self) -> Any:
        """
        method: options

        arguments:
         none

        return:
         the options object from the underlying real sampler
        """
        return self.real_sampler.options
    #
    # end of method

    def __deepcopy__(self, memo: dict) -> "HilbertSamplerProxy":
        """
        method: __deepcopy__

        arguments:
         memo: the deepcopy memo dict

        return:
         a new HilbertSamplerProxy wrapping a deep-copied sampler

        description:
         Required by Qiskit optimizers that deepcopy primitives. The
         tape reference is shared (not copied) intentionally, and the
         calibration capture state is carried over so a deepcopy does
         not trigger a redundant backend query.
        """
        clone = HilbertSamplerProxy(
            tape=self.tape,
            real_sampler=copy.deepcopy(self.real_sampler, memo),
        )
        clone.calibration_refresh_s = self.calibration_refresh_s
        clone._cal_next_check = self._cal_next_check
        clone._cal_last_hash = self._cal_last_hash
        return clone
    #
    # end of method

    def _backend_id(self) -> str:
        """
        method: _backend_id

        arguments:
         none

        return:
         a human-readable backend identifier string

        description:
         Returns the backend name if available, otherwise falls back to
         the sampler class name.
        """

        # prefer the backend's name attribute
        #
        backend = getattr(self.real_sampler, "backend", None)
        if backend is not None:
            return getattr(backend, "name", str(backend))
        return type(self.real_sampler).__name__
    #
    # end of method

    def run(self, pubs: Any, **kwargs: Any) -> Any:
        """
        method: run

        arguments:
         pubs:    list of primitive unified blocs (PUBs) to execute
         **kwargs: forwarded to the real sampler's .run()

        return:
         the real sampler job object

        description:
         Intercepts each PUB after batch execution. For each PUB:
          - Circuit is serialised to QASM and stored in the file store
          - Bitstring counts are stored as inline outcome artifacts
          - Parameter bindings are stored as inline artifacts
         Failures per-PUB are printed as warnings and do not interrupt
         the batch (INV-007 is honoured at the batch level).
        """

        # capture/refresh the calibration snapshot (rate-limited;
        # handles property-, method-, and private-attr backend access)
        #
        _maybe_capture_calibration(self, self.tape, self.real_sampler)

        # execute the batch via the real sampler
        #
        job = self.real_sampler.run(pubs, **kwargs)
        res = job.result()

        # record one span per PUB
        #
        for i, pub in enumerate(pubs):
            try:
                if getattr(self.tape, "_closed", True):
                    continue

                # extract the circuit from the PUB
                #
                circuit = (
                    pub[0]
                    if isinstance(pub, tuple)
                    else getattr(pub, "circuit", pub)
                )

                # serialise the circuit to QASM for the file store
                #
                try:
                    qasm_str = qasm3.dumps(circuit)
                except Exception as e:
                    qasm_str = f"// QASM Serialization Failed: {e}"

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    mode="w",
                    suffix=".qasm",
                    encoding="utf-8",
                ) as f:
                    f.write(qasm_str)
                    tmp_qasm = f.name

                qasm_hash = self.tape.attach_artifact(
                    src_path=tmp_qasm,
                    kind=Kind.circuit_qasm,
                    encoding=Encoding.openqasm,
                )
                os.remove(tmp_qasm)

                # open the execution span
                #
                with self.tape.execution_span(
                    payload_ref=qasm_hash,
                    backend_id=self._backend_id(),
                ) as span:

                    # store bitstring counts as inline outcome;
                    # vars(pub_result.data) gives {register: BitArray, ...}
                    #
                    pub_result = res[i]
                    total_shots = 0
                    try:
                        counts_by_register = {}
                        for reg_name, bit_array in vars(
                            pub_result.data
                        ).items():
                            if hasattr(bit_array, "get_counts"):
                                counts_by_register[reg_name] = {
                                    "counts":    bit_array.get_counts(),
                                    "num_shots": bit_array.num_shots,
                                    "num_bits":  bit_array.num_bits,
                                }
                                total_shots = bit_array.num_shots
                        span.outcome_ref = span.attach_inline(
                            json.dumps(counts_by_register),
                            kind="execution_outcome",
                            encoding="json",
                            producer="qiskit",
                        )
                    except Exception:
                        pass

                    # store parameter bindings as inline artifact
                    #
                    if (
                        isinstance(pub, tuple)
                        and len(pub) > 1
                        and pub[1] is not None
                    ):
                        try:
                            span.attach_inline(
                                json.dumps(np.array(pub[1]).tolist()),
                                kind="parameters",
                                encoding="json",
                                producer="qiskit",
                            )
                        except Exception:
                            pass

                    # emit the completion event with shot count
                    #
                    span.add_event(
                        "EXECUTION_COMPLETED",
                        {"batch_index": i, "shots": total_shots},
                    )

            except Exception as e:
                print(
                    f"  [HilbertBench] Warning: Failed to record "
                    f"Sampler span on Batch {i}: {e}"
                )

        # exit gracefully
        #
        return job
    #
    # end of method
#
# end of class

#
# end of file
