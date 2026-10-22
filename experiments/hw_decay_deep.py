#!/usr/bin/env python
#
# file: experiments/hw_decay_deep.py
#
# revision history:
#  20261022 (pm): initial version
#
# Exploratory deep follow-up to hw_decay.py (see hw_decay_deep_plan.md):
# one job at depths 64, 96 and 128 on the same two chains, to separate
# the exponential from the straight line where they disagree. Reuses the
# registered script's helpers unchanged.
#
# Usage:
#   python hw_decay_deep.py prepare      build and check, no QPU
#   python hw_decay_deep.py run          submit the one job
#   python hw_decay_deep.py analyze      exploratory analysis
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import json
import os
import sys
import warnings

# import third-party modules
#
import numpy as np
from qiskit import qasm3
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)

# import hilbertbench modules
#
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape

# import experiment infrastructure
#
import hw_decay as reg
from common import RESULTS_ROOT, TRACES_ROOT, build_ansatz, pair_observable
from esp_recompute import _calibration_tables, corrected_esp

# silence framework deprecation chatter for a clean report
#
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

DEPTHS = [64, 96, 128]
SEED = 20261032
OUT = RESULTS_ROOT / "hw_decay_deep"
TRACES = TRACES_ROOT / "hw_decay_deep"

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def design() -> dict:
    """Random and reference rows for the deep depths."""
    rng = np.random.default_rng(SEED)
    rng_ref = np.random.default_rng(SEED + 1)
    out = {}
    for d in DEPTHS:
        m = build_ansatz(reg.N_QUBITS, d, "linear")[1]
        rand = rng.uniform(0, 2 * np.pi, (reg.N_POINTS, m))
        refs, vals = [], []
        while len(refs) < reg.N_REF:
            row = rng_ref.integers(0, 4, m) * (np.pi / 2)
            v = reg._ideal(d, row[None, :])[0]
            if abs(v) > reg.REF_TOL:
                refs.append(row)
                vals.append(float(np.round(v)))
        out[d] = {"random": rand, "ref": np.array(refs),
                  "ideal_random": reg._ideal(d, rand),
                  "ideal_ref": np.array(vals)}
    return out


def build_pubs(backend, rows: dict, layouts: dict) -> tuple:
    """Same transpilation and routing checks as the registered script."""
    obs = pair_observable(reg.N_QUBITS)
    pubs, meta = [], []
    for d in DEPTHS:
        qc, _ = build_ansatz(reg.N_QUBITS, d, "linear")
        for name, layout in layouts.items():
            pm = generate_preset_pass_manager(
                backend=backend, optimization_level=1,
                initial_layout=layout, seed_transpiler=reg.SEED,
            )
            isa = pm.run(qc)
            placed = isa.layout.final_index_layout(filter_ancillas=True)
            ops = isa.count_ops()
            n2 = sum(ops.get(g, 0) for g in ("cz", "ecr", "cx"))
            if placed != layout or n2 != 3 * d:
                raise RuntimeError(f"routing changed chain {name} d={d}")
            P = np.vstack([rows[d]["random"], rows[d]["ref"]])
            pubs.append((isa, obs.apply_layout(isa.layout), P,
                         reg.PRECISION))
            meta.append({"chain": name, "layout": layout, "depth": d,
                         "two_qubit_gates": n2,
                         "count_ops": {k: int(v) for k, v in ops.items()},
                         "measured": [layout[2], layout[3]],
                         "qasm": qasm3.dumps(isa)})
    return pubs, meta


def _setup():
    lay = json.loads((RESULTS_ROOT / "hw_decay" / "layout.json")
                     .read_text())
    backend = reg.get_backend(False, lay["device"])
    rows = design()
    pubs, meta = build_pubs(backend, rows,
                            {"A": reg.LAYOUT_A, "B": lay["chain_b"]})
    return backend, rows, pubs, meta


def prepare() -> int:
    """Build and check every PUB; report the shot count."""
    backend, rows, pubs, _ = _setup()
    n = sum(p[2].shape[0] for p in pubs) * 512
    for d in DEPTHS:
        print(f"  d={d:>3}  ideal var {np.var(rows[d]['ideal_random']):.4f}"
              f"  refs {rows[d]['ideal_ref'].tolist()}")
    print(f"  {len(pubs)} PUBs on {backend.name}, {n} shots")
    return 0


def run() -> int:
    """Submit the job under the registered budget rule and record it."""
    from qiskit_ibm_runtime import EstimatorV2
    import qiskit
    import qiskit_ibm_runtime

    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "job_1.json").exists():
        print("  already run")
        return 0
    backend, rows, pubs, meta = _setup()
    n_shots = sum(p[2].shape[0] for p in pubs) * 512

    # budget: actual usage of the registered jobs, scaled estimate
    #
    prev = [json.loads(f.read_text())
            for f in sorted((RESULTS_ROOT / "hw_decay").glob("job_*.json"))]
    spent = float(sum(j.get("quantum_seconds") or 0.0 for j in prev))
    rate = max((j["quantum_seconds"] or 0.0) / j["n_shots"] for j in prev)

    cal_submit = reg._props_dict(backend, refresh=True)
    est = EstimatorV2(mode=backend)
    est.options.resilience_level = 0
    est.options.twirling.enable_gates = False
    est.options.twirling.enable_measure = False
    submitted = reg._now()
    job = est.run(pubs)
    job_id = job.job_id()
    print(f"  job {job_id} on {backend.name}, {len(pubs)} PUBs, "
          f"{n_shots} shots", flush=True)
    qs_est, src = reg._estimate(job, n_shots, [])
    if src != "ibm":
        qs_est, src = 1.3 * rate * n_shots, "1.3 x registered rate"
    print(f"  estimated {qs_est:.0f} s ({src}); spent {spent:.0f} s",
          flush=True)
    if qs_est > reg.JOB_CAP_S or spent + qs_est > reg.BUDGET_S:
        job.cancel()
        print("  cancelled: budget rule", flush=True)
        return 2

    with HilbertTape(
        TRACES, mode=Mode.active,
        tags={"experiment": "hw_decay_deep", "backend": backend.name,
              "job_id": job_id},
    ) as tape:
        proxy = HilbertEstimatorProxy(
            tape, real_estimator=reg._Submitted(job, backend, est.options))
        proxy.run(pubs)
    res = job.result()
    metrics = json.loads(json.dumps(job.metrics(), default=str))
    pub_out = []
    for i, m in enumerate(meta):
        rec = dict(m)
        rec["evs"] = np.asarray(res[i].data.evs, float).ravel().tolist()
        rec["stds"] = np.asarray(res[i].data.stds, float).ravel().tolist()
        rec["shots"] = 512
        pub_out.append(rec)
    record = {
        "job_id": job_id, "backend": backend.name,
        "client": {"qiskit": qiskit.__version__,
                   "qiskit_ibm_runtime": qiskit_ibm_runtime.__version__},
        "submitted": submitted, "finished": reg._now(),
        "quantum_seconds": (metrics.get("usage") or {}).get(
            "quantum_seconds"),
        "metrics": metrics, "n_shots": n_shots,
        "run_dir": str(tape.dir_path), "cal_submit": cal_submit,
        "cal_done": reg._props_dict(backend, refresh=True),
        "pubs": pub_out,
    }
    (OUT / "job_1.json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"  done: {record['quantum_seconds']} QPU s", flush=True)
    return 0


def analyze() -> int:
    """Exploratory analysis of hw_decay_deep_plan.md."""
    reg_res = json.loads((RESULTS_ROOT / "hw_decay" / "results.json")
                         .read_text())
    job = json.loads((OUT / "job_1.json").read_text())
    rows = design()
    cal, which = reg._snapshot(job)
    errs, ro = _calibration_tables(cal)
    rng = np.random.default_rng(SEED + 2)
    recs = []
    for p in job["pubs"]:
        d = p["depth"]
        y_all = np.asarray(p["evs"])
        x, y = rows[d]["ideal_random"], y_all[:reg.N_POINTS]
        y_ref = y_all[reg.N_POINTS:]
        f = float(np.polyfit(x, y, 1)[0])
        se, ci = reg._boot_slope(x, y, rng)
        fits = reg_res["fits"][p["chain"]]["pooled"]["models"]
        ex, li = fits["exponential"]["params"], fits["line"]["params"]
        pe = ex["A"] * ex["q"] ** d
        pl = li["a"] + li["b"] * d
        recs.append({
            "chain": p["chain"], "depth": d, "slope_f": f,
            "slope_f_se": se, "slope_f_ci95": ci,
            "ref_damping": float(np.mean(y_ref / rows[d]["ideal_ref"])),
            "ref_values": y_ref.tolist(),
            "esp": corrected_esp(p["qasm"], errs, ro,
                                 p["measured"])["esp"],
            "esp_snapshot": which,
            "pred_exponential": pe, "pred_line": pl,
            "z_exponential": (f - pe) / se, "z_line": (f - pl) / se,
        })

    # refit on all depths: registered records plus the deep points
    #
    refits = {}
    for chain in ("A", "B"):
        pts = [(r["depth"], r["slope_f"], r["slope_f_se"])
               for r in reg_res["records"] if r["chain"] == chain]
        pts += [(r["depth"], r["slope_f"], r["slope_f_se"])
                for r in recs if r["chain"] == chain]
        L, f, se = zip(*pts)
        refits[chain] = reg._fits(L, f, se)

    payload = {"protocol": "hw_decay_deep_v1 (exploratory)",
               "plan": "experiments/hw_decay_deep_plan.md",
               "job_id": job["job_id"],
               "quantum_seconds": job["quantum_seconds"],
               "records": recs, "refits_all_depths": refits}
    (OUT / "results.json").write_text(
        json.dumps(payload, indent=1, default=str) + "\n")
    for r in recs:
        print(f"  {r['chain']} d={r['depth']:>3}  f={r['slope_f']:.3f} "
              f"[{r['slope_f_ci95'][0]:.3f},{r['slope_f_ci95'][1]:.3f}]  "
              f"ref={r['ref_damping']:.3f}  esp={r['esp']:.3f}  "
              f"exp pred {r['pred_exponential']:.3f} "
              f"(z {r['z_exponential']:+.1f})  line pred "
              f"{r['pred_line']:.3f} (z {r['z_line']:+.1f})")
    for chain, rf in refits.items():
        m = rf["models"]
        print(f"  {chain} all depths: AIC exp {m['exponential']['aic']:.1f}"
              f"  line {m['line']['aic']:.1f}  const "
              f"{m['constant']['aic']:.1f}; q "
              f"{m['exponential']['params']['q']:.4f} "
              f"{[round(v, 4) for v in m['exponential']['q_ci95']]}")
    return 0


def main() -> int:
    """Command-line entry point."""
    cmd = sys.argv[1] if len(sys.argv) > 1 else "prepare"
    return {"prepare": prepare, "run": run, "analyze": analyze}[cmd]()
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
