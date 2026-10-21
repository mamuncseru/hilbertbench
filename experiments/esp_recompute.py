#!/usr/bin/env python
#
# file: experiments/esp_recompute.py
#
# revision history:
#  20261021 (pm): initial version
#  20261021 (pm): add the measured damping of the Study 1 hardware
#                 runs and the unanalysed ibm_kingston single points
#
# Recomputes the estimated success probability (ESP) of every logged
# device-noise circuit from the calibration snapshot stored in its log,
# and compares it with the value the frozen v1.0.0 noise analyzer
# reports.
#
# The v1.0.0 analyzer averages the gate_error of EVERY one-qubit entry
# in the snapshot's gate list and applies that mean to every one-qubit
# gate in the circuit; likewise for two-qubit entries. On IBM Heron
# snapshots the one-qubit list includes `measure` (error ~1e-2) and
# `rz` (error 0, a virtual gate), and the two-qubit list includes `rzz`
# (a fractional gate the circuits never use), so the mean single-qubit
# error is inflated about fivefold and is charged to every virtual RZ.
#
# The corrected ESP charges each physical gate its own reported error
# on its own qubits:
#   prod over sx, x gates      (1 - e_sx[q])
#   prod over cz / cx gates    (1 - e_2q[edge])
#   prod over measured qubits  (1 - readout[q])
# with rz error-free, and the measured qubits taken from the support of
# the logged (physical) observable. The registered results keep the
# v1.0.0 values; this file reports the correction.
#
#   python esp_recompute.py
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import json
import re
import sys
import warnings
from collections import Counter
from pathlib import Path

# import third-party modules
#
import numpy as np
from qiskit.quantum_info import Statevector

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.analysis.noise import noise_profile

# import experiment infrastructure
#
from common import (RESULTS_ROOT, TRACES_ROOT, build_ansatz,
                    pair_observable, save_result)

# silence framework deprecation chatter for a clean report
#
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

GATE_LINE = re.compile(r"^\s*([a-z_]\w*)\s*(\([^;]*\))?\s+(\$\d+(?:\s*,\s*\$\d+)*)\s*;")
ONE_Q_PHYSICAL = ("sx", "x")
TWO_Q_NATIVE = ("cz", "cx", "ecr")
VIRTUAL = ("rz", "barrier", "delay")
THRESHOLD = 0.5

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------


def _calibration_tables(cal: dict) -> tuple:
    """Per-gate, per-qubit-tuple errors and per-qubit readout errors."""
    errs = {}
    for g in cal.get("gates", []):
        name = g.get("gate") or g.get("name")
        p = {x["name"]: x["value"] for x in g.get("parameters", [])}
        if "gate_error" in p:
            errs.setdefault(name, {})[tuple(g["qubits"])] = p["gate_error"]
    ro = {}
    for i, q in enumerate(cal.get("qubits", [])):
        p = {x["name"]: x["value"] for x in q}
        if "readout_error" in p:
            ro[i] = p["readout_error"]
    return errs, ro


def _gates(qasm: str) -> list:
    """(name, physical-qubit tuple) for every gate line of ISA QASM."""
    out = []
    for line in qasm.splitlines():
        m = GATE_LINE.match(line)
        if not m:
            continue
        qubits = tuple(int(x.strip()[1:]) for x in m.group(3).split(","))
        out.append((m.group(1), qubits))
    return out


def _measured(trace: HilbertTrace) -> list:
    """Physical qubits in the support of the first logged observable."""
    for sp in trace.completed():
        obs = sp.observables
        if obs:
            label = obs[0][0]
            n = len(label)
            return sorted(n - 1 - i for i, c in enumerate(label) if c != "I")
    return []


def corrected_esp(qasm: str, errs: dict, ro: dict, measured: list) -> dict:
    """
    function: corrected_esp

    arguments:
     qasm:     ISA OpenQASM of one circuit (physical qubits $k)
     errs:     calibration errors by gate name and qubit tuple
     ro:       readout error by qubit
     measured: physical qubits that are read out

    return:
     the corrected ESP and its three factors, plus gate counts
    """

    gates = _gates(qasm)
    counts = Counter(g for g, _ in gates)
    s1 = s2 = 1.0
    unknown = Counter()
    for name, qs in gates:
        if name in VIRTUAL:
            continue
        if name in ONE_Q_PHYSICAL:
            e = errs.get(name, {}).get(qs)
            if e is None:
                e = errs.get("sx", {}).get(qs)
            if e is None:
                unknown[name] += 1
                continue
            s1 *= 1.0 - e
        elif name in TWO_Q_NATIVE:
            table = errs.get(name, {})
            e = table.get(qs, table.get(qs[::-1]))
            if e is None:
                unknown[name] += 1
                continue
            s2 *= 1.0 - e
        else:
            unknown[name] += 1
    sro = float(np.prod([1.0 - ro[q] for q in measured if q in ro]))
    return {
        "esp": s1 * s2 * sro,
        "single_qubit_factor": s1, "two_qubit_factor": s2,
        "readout_factor": sro,
        "n_sx_x": sum(counts[g] for g in ONE_Q_PHYSICAL),
        "n_rz": counts["rz"],
        "n_2q": sum(counts[g] for g in TWO_Q_NATIVE),
        "unpriced_gates": dict(unknown),
    }


def v100_fhat(prof: dict, qasm_struct: dict) -> float:
    """The v1.0.0 formula, applied to one circuit's structure."""
    n_meas = qasm_struct["num_measurements"] or qasm_struct["num_qubits"]
    return float((1 - prof["gate_error_1q_mean"]) ** qasm_struct["single_qubit_gates"]
                 * (1 - prof["gate_error_2q_mean"]) ** qasm_struct["entangling_gates"]
                 * (1 - prof["readout_error"]["mean"]) ** n_meas)


def analyse_log(run_dir: Path) -> list:
    """Every recorded circuit of one log, both estimates."""
    from hilbertbench.analysis.circuit import _parse_qasm
    t = HilbertTrace(str(run_dir))
    cal = t.calibration()
    if cal is None:
        return []
    errs, ro = _calibration_tables(cal)
    prof = noise_profile(t)
    measured = _measured(t)
    rows = []
    for ref, meta in t.catalog.items():
        if meta.get("kind") != "circuit_qasm":
            continue
        qasm = t._resolve_ref({}, ref)
        if not isinstance(qasm, str):
            continue
        st = _parse_qasm(qasm)
        c = corrected_esp(qasm, errs, ro, measured)
        rows.append({
            "backend": cal.get("backend_name"),
            "calibration_date": cal.get("last_update_date"),
            "num_qubits": st["num_qubits"],
            "entangling_gates": st["entangling_gates"],
            "fhat_v100": v100_fhat(prof, st),
            "measured_qubits": measured,
            **c,
        })
    return rows


def _primary(rows: list) -> dict:
    return max(rows, key=lambda r: r["entangling_gates"])


def study1_damping() -> dict:
    """
    function: study1_damping

    return:
     per hardware class of Study 1: the damping factor f pooled over
     the nine runs (slope of hardware on noiseless values, bootstrap
     interval), and per run the noiseless and hardware variances and
     the expected shot-noise variance

    description:
     The Study 1 hardware runs are random-parameter samples, so their
     noiseless values can be computed exactly and the damping measured
     directly, as in hw_reanalysis.py.
    """

    settings = {"healthy": (4, 2, 400), "shot_starved": (4, 2, 5),
                "noise_dominated": (4, 16, 100),
                "barren_plateau": (8, 8, 1112)}
    root = TRACES_ROOT / "corpus_test"
    man = json.loads((root / "manifest.json").read_text())
    rng = np.random.default_rng(20261021)
    out = {}
    for lab, (n, layers, shots) in settings.items():
        qc, _ = build_ansatz(n, layers, "linear")
        obs = pair_observable(n)
        xs, ys, runs = [], [], []
        for rel, v in man.items():
            if v.get("stratum") != "hardware" or v["label"] != lab:
                continue
            sps = list(HilbertTrace(str(root / rel)).completed())
            P = np.vstack([np.atleast_2d(np.asarray(sp.parameters, float))
                           for sp in sps])
            y = np.concatenate([np.asarray(sp.outcome, float).ravel()
                                for sp in sps])
            x = np.array([Statevector(qc.assign_parameters(q))
                          .expectation_value(obs).real for q in P])
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(x), len(x))
                bs.append(np.polyfit(x[i], y[i], 1)[0])
            runs.append({"run": rel,
                         "f": float(np.polyfit(x, y, 1)[0]),
                         "f_ci95": [float(q) for q in
                                    np.percentile(bs, [2.5, 97.5])],
                         "r_squared": float(np.corrcoef(x, y)[0, 1] ** 2),
                         "noiseless_variance": float(np.var(x)),
                         "hardware_variance": float(np.var(y)),
                         "shot_variance": float(np.mean((1 - y ** 2) / shots))})
            xs.append(x)
            ys.append(y)
        x, y = np.concatenate(xs), np.concatenate(ys)
        f = float(np.polyfit(x, y, 1)[0])
        boot = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            boot.append(np.polyfit(x[i], y[i], 1)[0])
        # noise-corrected variance estimate (Var_hw - shot) / f^2, once
        # with the condition's pooled f and once with the run's own f
        for r in runs:
            excess = r["hardware_variance"] - r["shot_variance"]
            r["noiseless_estimate"] = excess / f ** 2
            r["noiseless_estimate_own_f"] = excess / r["f"] ** 2
        out[lab] = {"pooled_f": f,
                    "pooled_f_ci95": [float(q) for q in
                                      np.percentile(boot, [2.5, 97.5])],
                    "runs": runs}
        print(f"  Study 1 hw {lab:<16} f = {f:.3f} "
              f"[{out[lab]['pooled_f_ci95'][0]:.3f}, "
              f"{out[lab]['pooled_f_ci95'][1]:.3f}]")
    return out


def kingston_points() -> list:
    """Single-point logs on ibm_kingston, not in the planned analysis."""
    m_to_depth = {build_ansatz(4, d, "linear")[1]: d for d in range(1, 25)}
    obs = pair_observable(4)
    rows = []
    for d in sorted((TRACES_ROOT / "hw_fidelity").iterdir()):
        t = HilbertTrace(str(d))
        cal = t.calibration() or {}
        if cal.get("backend_name") != "ibm_kingston":
            continue
        sp = next(iter(t.completed()))
        theta = np.asarray(sp.parameters, float).ravel()
        noisy = float(np.asarray(sp.outcome, float).ravel()[0])
        depth = m_to_depth[len(theta)]
        qc, _ = build_ansatz(4, depth, "linear")
        ideal = float(Statevector(qc.assign_parameters(theta))
                      .expectation_value(obs).real)
        p = _primary(analyse_log(d))
        rows.append({"run": d.name, "depth": depth, "ideal": ideal,
                     "noisy": noisy, "observed_attenuation": noisy / ideal,
                     "fhat_v100": p["fhat_v100"], "esp": p["esp"],
                     "calibration_date": p["calibration_date"]})
    return rows


def main() -> int:
    out = {}

    # hardware landscape logs, one entry per depth
    #
    hw = []
    for device, rel, date in [
            ("ibm_marrakesh", "hw_var_full/20260630_143721_587f61e5", "2026-06-30"),
            ("ibm_marrakesh", "hw_var_full/20260711_193615_1d303c23", "2026-07-11"),
            ("ibm_fez", "hw_onset_final/20260618_060633_a2a2731d", "2026-06-18")]:
        for r in analyse_log(TRACES_ROOT / rel):
            r.update({"device": device, "job_date": date,
                      "depth": r["entangling_gates"] // 3})
            hw.append(r)
    hw.sort(key=lambda r: (r["device"], r["job_date"], r["depth"]))
    out["hardware_landscape"] = hw
    print("== hardware landscape logs")
    for r in hw:
        print(f"  {r['device']:<14} {r['job_date']} d={r['depth']:>2}  "
              f"v1.0.0 {r['fhat_v100']:.3f}  corrected {r['esp']:.3f}  "
              f"(1q {r['single_qubit_factor']:.3f}, 2q "
              f"{r['two_qubit_factor']:.3f}, ro {r['readout_factor']:.3f}; "
              f"measured {r['measured_qubits']})")

    # Study 1 corpus: hardware stratum and simulated noise runs
    #
    root = TRACES_ROOT / "corpus_test"
    s1 = []
    for rel, v in json.loads((root / "manifest.json").read_text()).items():
        rows = analyse_log(root / rel)
        if not rows:
            continue
        p = _primary(rows)
        s1.append({"run": rel, "label": v["label"],
                   "stratum": v.get("stratum") or "simulated", **p})
    out["study1"] = s1

    # sweeps and Study 2: every log with a calibration snapshot
    #
    for key, corpus in [("sweeps", "corpus_hardmode"), ("study2", "corpus_g")]:
        root = TRACES_ROOT / corpus
        man = json.loads((root / "manifest.json").read_text())
        rows_out = []
        for rel, v in man.items():
            run = v.get("run", rel)
            rows = analyse_log(root / run)
            if not rows:
                continue
            rows_out.append({"run": run, "arm": v["arm"],
                             "level": v["level"], **_primary(rows)})
        out[key] = rows_out

    # single-point hardware checks (App. D) and the development corpus
    #
    sp = []
    for d in sorted((TRACES_ROOT / "hw_fidelity").iterdir()):
        rows = analyse_log(d)
        if rows:
            sp.append({"run": d.name, **_primary(rows)})
    smoke = analyse_log(TRACES_ROOT / "hw_smoke" / "20260610_232503_9c8a892e")
    out["single_points"] = sp
    out["smoke"] = smoke
    root = TRACES_ROOT / "corpus_dev"
    out["development"] = [
        {"run": rel, "label": v["label"], **_primary(analyse_log(root / rel))}
        for rel, v in json.loads((root / "manifest.json").read_text()).items()
        if analyse_log(root / rel)]

    # summaries: does the corrected value change any registered flag?
    #
    def flips(rows):
        return sum((r["fhat_v100"] < THRESHOLD) != (r["esp"] < THRESHOLD)
                   for r in rows)

    summ = {}
    for key in ("study1", "sweeps", "study2", "development"):
        rows = out[key]
        summ[key] = {
            "n_with_calibration": len(rows),
            "flagged_v100": sum(r["fhat_v100"] < THRESHOLD for r in rows),
            "flagged_corrected": sum(r["esp"] < THRESHOLD for r in rows),
            "flag_changes": flips(rows),
        }
    out["summary"] = summ
    print(json.dumps(summ, indent=1))

    print("== Study 1 hardware stratum")
    for lab in ("healthy", "barren_plateau", "shot_starved", "noise_dominated"):
        rr = [r for r in s1 if r["stratum"] == "hardware" and r["label"] == lab]
        if rr:
            print(f"  {lab:<16} v1.0.0 {np.mean([r['fhat_v100'] for r in rr]):.3f}"
                  f"  corrected {np.mean([r['esp'] for r in rr]):.3f}")
    print("== simulated noise conditions (Study 2, by depth)")
    for lv in sorted({r["level"] for r in out["study2"]
                      if r["arm"] == "noise_dominated"}):
        rr = [r for r in out["study2"]
              if r["arm"] == "noise_dominated" and r["level"] == lv]
        print(f"  depth {lv:>2}: v1.0.0 {np.mean([r['fhat_v100'] for r in rr]):.3f}"
              f"  corrected {np.mean([r['esp'] for r in rr]):.3f}")
    print("== simulated noise conditions (sweeps, by depth)")
    for lv in sorted({r["level"] for r in out["sweeps"]
                      if r["arm"] == "noise_dominated"}):
        rr = [r for r in out["sweeps"]
              if r["arm"] == "noise_dominated" and r["level"] == lv]
        print(f"  depth {lv:>2}: v1.0.0 {np.mean([r['fhat_v100'] for r in rr]):.3f}"
              f"  corrected {np.mean([r['esp'] for r in rr]):.3f}")

    print("== Study 1 hardware runs: measured damping")
    out["study1_hardware_damping"] = study1_damping()
    out["kingston_points"] = kingston_points()
    k = out["kingston_points"]
    o = np.array([r["observed_attenuation"] for r in k])
    out["kingston_summary"] = {
        "n": len(k),
        "rmse_v100": float(np.sqrt(np.mean((o - [r["fhat_v100"] for r in k]) ** 2))),
        "rmse_esp": float(np.sqrt(np.mean((o - [r["esp"] for r in k]) ** 2))),
        "rmse_predict_mean": float(np.std(o)),
    }
    print("  kingston:", json.dumps(out["kingston_summary"]))

    path = save_result("esp_recompute", out)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#
# end of file
