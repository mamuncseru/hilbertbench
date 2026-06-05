#!/usr/bin/env python
#
# file: demo/02_qiskit_sampler.py
#
# revision history:
#  20260605 (am): initial version
#
# QAOA with Qiskit V2 Sampler + HilbertBench passive recording.
#
# Runs a parameter sweep over 10 (gamma, beta) configurations for a
# 2-qubit QAOA circuit targeting MaxCut on the edge (0, 1). Each
# configuration is one Sampler call — the proxy records a span
# containing the circuit QASM and bitstring counts inline.
#
# The demo prints the top-2 bitstrings per configuration and then
# inspects the recorded trace to show how outcome data is stored.
#
# Prerequisites:
#   pip install hilbertbench[qiskit,storage]
#
# Usage:
#   python demo/02_qiskit_sampler.py
#------------------------------------------------------------------------------

# import system modules
#
import json
import os

# import third-party modules
#
import numpy as np

# import qiskit modules
#
from qiskit.circuit import QuantumCircuit, ParameterVector

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.integrations.qiskit import HilbertSamplerProxy
from hilbertbench.recorder.tape import HilbertTape
from hilbertbench.recorder.storage.writer import convert_trace_to_parquet

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# output root — timestamped run directory is created inside
#
RUNS_DIR = "runs/sampler_qaoa"

# shots per configuration
#
N_SHOTS = 512

# number of (gamma, beta) parameter configurations to sweep
#
N_CONFIGS = 10

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def build_qaoa_circuit() -> tuple:
    """
    function: build_qaoa_circuit

    arguments:
     none

    return:
     (circuit, params) — parameterized QAOA circuit and ParameterVector

    description:
     Builds a 1-layer QAOA circuit for the MaxCut problem on the
     edge graph (0, 1).

       H ⊗ H
       Rzz(2γ, 0, 1)   — cost layer (ZZ entangler)
       Rx(2β, 0)        — mixer layer
       Rx(2β, 1)
       measure_all()

     MaxCut ground state is |01⟩ or |10⟩ (cuts the single edge).
     The optimal parameters are γ = π/4, β = π/8 for this graph.
    """

    # build the QAOA circuit
    #
    p = ParameterVector("p", 2)   # p[0] = gamma, p[1] = beta
    qc = QuantumCircuit(2)

    # equal superposition
    #
    qc.h([0, 1])

    # cost layer: Rzz = CNOT · Rz · CNOT
    #
    qc.cx(0, 1)
    qc.rz(2.0 * p[0], 1)
    qc.cx(0, 1)

    # mixer layer
    #
    qc.rx(2.0 * p[1], 0)
    qc.rx(2.0 * p[1], 1)

    qc.measure_all()

    # exit gracefully
    #
    return qc, p
#
# end of function


def main() -> None:
    """
    function: main

    arguments:
     none

    return:
     none

    description:
     Opens a HilbertTape and wraps StatevectorSampler in
     HilbertSamplerProxy. Runs N_CONFIGS Sampler calls — one per
     (gamma, beta) configuration. Each call is recorded as one span
     with the bitstring counts stored as an inline JSON artifact.

     After the tape closes the trace is converted to Parquet and
     inspected to show how outcome data is accessible from
     HilbertTrace.
    """

    # print run header
    #
    sep = "-" * 60
    print(f"\n[{__FILE__}]  HilbertBench — Qiskit Sampler (QAOA)")
    print(sep)

    # build the circuit
    #
    circuit, params = build_qaoa_circuit()
    print(f"  Circuit    : 2-qubit QAOA, 2 parameters (γ, β)")
    print(f"  Problem    : MaxCut on edge (0, 1)")
    print(f"  Shots/call : {N_SHOTS}")
    print(f"  Configs    : {N_CONFIGS}")
    print(sep)

    # generate a uniform sweep over [0, π/2] × [0, π/4]
    #
    rng = np.random.default_rng(42)
    gamma_vals = rng.uniform(0.0, np.pi / 2.0, N_CONFIGS)
    beta_vals = rng.uniform(0.0, np.pi / 4.0, N_CONFIGS)

    # open the tape and run the parameter sweep
    #
    with HilbertTape(
        RUNS_DIR,
        tags={"demo": "qiskit_sampler", "algorithm": "qaoa"},
    ) as tape:

        # wrap StatevectorSampler — no other code changes required
        #
        sampler = HilbertSamplerProxy(tape)

        for i, (gamma, beta) in enumerate(zip(gamma_vals, beta_vals)):

            # one call per configuration — proxy records the span
            #
            pv = np.array([[gamma, beta]])   # shape (1, 2)
            job = sampler.run([(circuit, pv, N_SHOTS)])
            result = job.result()

            # extract bitstring counts from the DataBin
            #
            counts = result[0].data.meas.get_counts()
            top2 = sorted(counts.items(), key=lambda kv: -kv[1])[:2]
            top_str = "  ".join(f"{k}:{v}" for k, v in top2)
            print(
                f"  config {i+1:2d}  γ={gamma:.3f}  β={beta:.3f}"
                f"  top-2: {top_str}"
            )

    # convert to Parquet
    #
    parquet_path = convert_trace_to_parquet(tape.dir_path)

    # inspect the trace — show inline outcome from span 0
    #
    trace = HilbertTrace(tape.dir_path)
    span0 = trace.completed()[0]
    raw_counts = span0.outcome    # resolves the inline JSON artifact
    if isinstance(raw_counts, dict):
        total_shots = sum(raw_counts.get("meas", {}).get("counts", {}).values()
                          if isinstance(raw_counts.get("meas"), dict)
                          else raw_counts.values())
    else:
        total_shots = "n/a"

    print(sep)
    print(f"  Trace dir     : {tape.dir_path}")
    print(f"  Parquet       : {parquet_path.name}")
    print(f"  Trace status  : {trace.status}")
    print(f"  Spans recorded: {len(trace)}")
    print(f"  Span[0] outcome type : {type(raw_counts).__name__}")
    print(sep)
#
# end of function

#------------------------------------------------------------------------------
#
# main entry point
#
#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#
# end of file
