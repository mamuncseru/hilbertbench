#!/usr/bin/env python
#
# file: experiments/hw_decay.py
#
# revision history:
#  20261022 (pm): initial version
#
# Depth decay of the noise damping on one IBM Heron processor: the
# measurement planned in experiments/hw_decay_plan.md and sent to the
# advisor before any data existed. Everything that decides the result
# (depths, parameter draws, reference circuits, the second qubit chain
# rule, the budget rule, and the analysis) is fixed in this file.
#
# One job holds 16 PUBs: two fixed 4-qubit chains x eight depths. Each
# PUB binds 40 random parameter rows plus 4 reference rows (angles in
# multiples of pi/2, noiseless cost exactly +1 or -1) to the same ISA
# circuit, so the reference circuits run the identical physical gates.
# The job is repeated three times about an hour apart. The job is
# submitted first, IBM's usage estimate is checked against the budget
# (and the job cancelled while still queued if it is too high), and
# only then is it handed to the HilbertBench proxy, which records the
# calibration, circuits, parameters and outcomes in a sealed log.
#
# Usage:
#   python hw_decay.py prepare [--fake]     build and check, no QPU
#   python hw_decay.py session [--fake]     three jobs, an hour apart
#   python hw_decay.py run --repeat K       one job (resume a session)
#   python hw_decay.py analyze [--fake]     the pre-specified analysis
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import hashlib
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

# import third-party modules
#
import numpy as np
from qiskit import qasm3
from qiskit.quantum_info import Statevector
from qiskit.transpiler.preset_passmanagers import (
    generate_preset_pass_manager,
)
from scipy.optimize import curve_fit

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.analysis.circuit import _parse_qasm
from hilbertbench.analysis.noise import noise_profile
from hilbertbench.integrations.qiskit import HilbertEstimatorProxy
from hilbertbench.models import Mode
from hilbertbench.recorder.tape import HilbertTape

# import experiment infrastructure
#
from common import (RESULTS_ROOT, TRACES_ROOT, build_ansatz,
                    pair_observable)
from esp_recompute import _calibration_tables, corrected_esp, v100_fhat

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

# private token file (outside the repo; never hardcode credentials)
#
TOKEN_FILE = Path.home() / ".qiskit" / "hb_ibm_token_decay"

# measurement design (fixed before any data; see hw_decay_plan.md)
#
N_QUBITS = 4
DEPTHS = [1, 4, 8, 12, 16, 24, 32, 48]
N_POINTS = 40
N_REF = 4
PRECISION = 0.0442           # 512 shots per row, as in the earlier jobs
SEED = 20261022
DEVICE = "ibm_marrakesh"
FALLBACK_DEVICE = "ibm_fez"
LAYOUT_A = [0, 1, 2, 3]
N_REPEATS = 3
GAP_S = 3600                 # submissions about an hour apart

# budget rule: the account holds 600 s of QPU time
#
BUDGET_S = 570.0
JOB_CAP_S = 180.0
SECONDS_PER_SHOT = 3.0e-4    # fallback when IBM gives no estimate

# analysis
#
N_BOOT = 2000
REF_TOL = 0.999

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def _dirs(fake: bool) -> tuple:
    """Result and trace directories (fake runs kept apart)."""
    name = "hw_decay_fake" if fake else "hw_decay"
    out = RESULTS_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    return out, TRACES_ROOT / name


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ideal(depth: int, rows: np.ndarray) -> np.ndarray:
    """Exact noiseless cost for each parameter row."""
    qc, _ = build_ansatz(N_QUBITS, depth, "linear")
    obs = pair_observable(N_QUBITS)
    return np.array([
        float(Statevector(qc.assign_parameters(p))
              .expectation_value(obs).real)
        for p in rows
    ])


def design() -> dict:
    """
    function: design

    return:
     {depth: {"random": rows, "ref": rows, "ideal_random": values,
     "ideal_ref": values}} plus a digest of all rows

    description:
     Draws the 40 random rows per depth uniformly in [0, 2 pi), and the
     reference rows by rejection: angles k pi / 2 drawn uniformly until
     N_REF rows have a noiseless cost of exactly +1 or -1. The same rows
     are used in every job and on both chains.
    """

    rng = np.random.default_rng(SEED)
    rng_ref = np.random.default_rng(SEED + 1)
    out, h = {}, hashlib.sha256()
    for d in DEPTHS:
        m = build_ansatz(N_QUBITS, d, "linear")[1]
        rand = rng.uniform(0, 2 * np.pi, (N_POINTS, m))
        refs, ref_vals, tries = [], [], 0
        while len(refs) < N_REF:
            tries += 1
            row = rng_ref.integers(0, 4, m) * (np.pi / 2)
            v = _ideal(d, row[None, :])[0]
            if abs(v) > REF_TOL:
                refs.append(row)
                ref_vals.append(float(np.round(v)))
        out[d] = {"random": rand, "ref": np.array(refs),
                  "ideal_random": _ideal(d, rand),
                  "ideal_ref": np.array(ref_vals),
                  "ref_draws": tries}
        h.update(np.round(rand, 12).tobytes())
        h.update(np.array(refs).tobytes())
    return {"rows": out, "digest": h.hexdigest()}


def _service():
    from qiskit_ibm_runtime import QiskitRuntimeService
    token = TOKEN_FILE.read_text().strip()
    return QiskitRuntimeService(channel="ibm_cloud", token=token)


def get_backend(fake: bool, name: str | None = None):
    """The pinned device (or the local fake copy of it)."""
    if fake:
        from qiskit_ibm_runtime.fake_provider import FakeMarrakesh
        return FakeMarrakesh()
    service = _service()
    if name:
        return service.backend(name)
    b = service.backend(DEVICE)
    if b.status().operational:
        return b
    return service.backend(FALLBACK_DEVICE)


def _props(backend, refresh: bool = False):
    try:
        return backend.properties(refresh=refresh)
    except TypeError:
        return backend.properties()


def _props_dict(backend, refresh: bool = False) -> dict:
    return json.loads(json.dumps(_props(backend, refresh).to_dict(),
                                 default=str))


def choose_chain(backend) -> dict:
    """
    function: choose_chain

    arguments:
     backend: the pinned device

    return:
     the second chain and its score

    description:
     Pre-specified rule: among all chains of four connected qubits
     disjoint from LAYOUT_A, take the one with the lowest sum of its
     three two-qubit gate errors, four readout errors and four sx
     errors in the current calibration. Chains with a missing value,
     an error >= 0.5, or a non-operational qubit are excluded. Of the
     two orientations of a chain the lexicographically smaller is used
     (circuit qubits 2 and 3, the measured pair, are its last two).
    """

    props = _props(backend)
    g2 = "cz" if "cz" in backend.operation_names else "ecr"
    adj = {}
    for a, b in backend.coupling_map.get_edges():
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)

    def err(fn, *args):
        try:
            v = fn(*args)
        except Exception:
            return None
        return None if v is None or v >= 0.5 else float(v)

    def ok(q):
        try:
            return props.is_qubit_operational(q)
        except Exception:
            return True

    best = None
    for a in adj:
        for b in adj[a]:
            for c in adj[b] - {a}:
                for d in adj[c] - {a, b}:
                    path = [a, b, c, d]
                    if path > path[::-1] or set(path) & set(LAYOUT_A):
                        continue
                    if not all(ok(q) for q in path):
                        continue
                    parts = [err(props.gate_error, g2, [p, q])
                             for p, q in zip(path, path[1:])]
                    parts += [err(props.readout_error, q) for q in path]
                    parts += [err(props.gate_error, "sx", [q])
                              for q in path]
                    if any(v is None for v in parts):
                        continue
                    s = float(sum(parts))
                    if best is None or (s, path) < (best[0], best[1]):
                        best = (s, path)
    return {"chain": best[1], "score": best[0], "two_qubit_gate": g2,
            "calibration_date": str(props.last_update_date)}


def build_pubs(backend, rows: dict, layouts: dict) -> tuple:
    """
    function: build_pubs

    return:
     (pubs, meta): one PUB per (chain, depth) and its description

    description:
     Transpiles each depth once per chain with the chain fixed as the
     initial layout, checks that no routing was added (3 two-qubit
     gates per layer, chain qubits as placed), and binds the random
     rows followed by the reference rows.
    """

    obs = pair_observable(N_QUBITS)
    pubs, meta = [], []
    for d in DEPTHS:
        qc, _ = build_ansatz(N_QUBITS, d, "linear")
        for name, layout in layouts.items():
            pm = generate_preset_pass_manager(
                backend=backend, optimization_level=1,
                initial_layout=layout, seed_transpiler=SEED,
            )
            isa = pm.run(qc)
            placed = isa.layout.final_index_layout(filter_ancillas=True)
            ops = isa.count_ops()
            n2 = sum(ops.get(g, 0) for g in ("cz", "ecr", "cx"))
            if placed != layout or n2 != 3 * d:
                raise RuntimeError(f"routing changed chain {name} d={d}: "
                                   f"{placed} {n2} two-qubit gates")
            P = np.vstack([rows[d]["random"], rows[d]["ref"]])
            pubs.append((isa, obs.apply_layout(isa.layout), P,
                         PRECISION))
            meta.append({"chain": name, "layout": layout, "depth": d,
                         "n_params": int(P.shape[1]),
                         "two_qubit_gates": n2,
                         "count_ops": {k: int(v) for k, v in ops.items()},
                         "measured": [layout[2], layout[3]],
                         "qasm": qasm3.dumps(isa)})
    return pubs, meta


def _estimate(job, n_shots: int) -> tuple:
    """IBM's usage estimate for a queued job, or the shot fallback."""
    t0 = time.time()
    while time.time() - t0 < 120:
        try:
            est = job.usage_estimation or {}
            qs = est.get("quantum_seconds")
            if qs is not None:
                return float(qs), "ibm"
        except Exception:
            break
        time.sleep(5)
    return n_shots * SECONDS_PER_SHOT, "shots x 0.3 ms"


class _Submitted:
    """Estimator stand-in that hands the proxy an already-run job."""

    def __init__(self, job, backend, options) -> None:
        self._job = job
        self.backend = backend
        self.options = options

    def run(self, pubs, **kwargs):
        return self._job


def run_job(k: int, fake: bool) -> dict:
    """
    function: run_job

    arguments:
     k:    repeat number (1..N_REPEATS)
     fake: run locally on the FakeMarrakesh noise model

    return:
     the saved job record

    description:
     Submits one job, applies the budget rule before it runs, records
     it through the HilbertBench proxy, and saves outcomes, timing,
     QPU usage and the calibration before and after.
    """

    from qiskit_ibm_runtime import EstimatorV2

    out, troot = _dirs(fake)
    lay = json.loads((out / "layout.json").read_text())
    backend = get_backend(fake, None if fake else lay["device"])
    rows = design()["rows"]
    pubs, meta = build_pubs(backend, rows,
                            {"A": LAYOUT_A, "B": lay["chain_b"]})
    n_shots = sum(p[2].shape[0] for p in pubs) * int(
        np.ceil(1 / PRECISION ** 2))

    # budget so far: actual usage of the earlier jobs of this session
    #
    spent = 0.0
    for f in sorted(out.glob("job_*.json")):
        spent += json.loads(f.read_text()).get("quantum_seconds") or 0.0

    cal_submit = _props_dict(backend, refresh=not fake)
    est = EstimatorV2(mode=backend)
    est.options.resilience_level = 0
    try:
        est.options.twirling.enable_gates = False
        est.options.twirling.enable_measure = False
    except Exception:
        pass

    # submit, then cancel while queued if the estimate breaks the rule
    #
    submitted = _now()
    job = est.run(pubs)
    job_id = job.job_id()
    print(f"  job {k}: {job_id} on {backend.name}, {len(pubs)} PUBs, "
          f"{n_shots} shots", flush=True)
    if not fake:
        qs_est, src = _estimate(job, n_shots)
        print(f"  estimated {qs_est:.0f} s ({src}); spent {spent:.0f} s",
              flush=True)
        if qs_est > JOB_CAP_S or spent + qs_est > BUDGET_S:
            job.cancel()
            print("  cancelled: budget rule", flush=True)
            (out / f"cancelled_{k}.json").write_text(json.dumps({
                "job_id": job_id, "estimate_s": qs_est, "spent_s": spent,
                "time": _now()}, indent=2))
            raise SystemExit(2)
    else:
        qs_est, src = None, "fake"

    # record through the tool (the proxy blocks until the job is done)
    #
    with HilbertTape(
        troot, mode=Mode.active,
        tags={"experiment": "hw_decay", "backend": backend.name,
              "repeat": str(k), "job_id": job_id},
    ) as tape:
        proxy = HilbertEstimatorProxy(
            tape, real_estimator=_Submitted(job, backend, est.options))
        proxy.run(pubs)
    res = job.result()

    # timing, usage, calibration after
    #
    metrics = {}
    try:
        metrics = json.loads(json.dumps(job.metrics(), default=str))
    except Exception:
        pass
    qs = (metrics.get("usage") or {}).get("quantum_seconds")
    cal_done = _props_dict(backend, refresh=not fake)

    pub_out = []
    for i, m in enumerate(meta):
        r = res[i]
        rec = dict(m)
        rec["evs"] = np.asarray(r.data.evs, float).ravel().tolist()
        rec["stds"] = np.asarray(r.data.stds, float).ravel().tolist()
        rec["shots"] = int((r.metadata or {}).get(
            "shots", int(np.ceil(1 / PRECISION ** 2))))
        pub_out.append(rec)
    record = {
        "repeat": k, "job_id": job_id, "backend": backend.name,
        "submitted": submitted, "finished": _now(),
        "quantum_seconds": qs, "usage_estimate_s": qs_est,
        "usage_estimate_source": src, "metrics": metrics,
        "n_shots": n_shots, "run_dir": str(tape.dir_path),
        "design_digest": design()["digest"],
        "cal_submit": cal_submit, "cal_done": cal_done, "pubs": pub_out,
    }
    path = out / f"job_{k}.json"
    path.write_text(json.dumps(record, indent=1) + "\n")
    print(f"  job {k} done: {qs} QPU s; wrote {path}", flush=True)
    return record


def prepare(fake: bool) -> int:
    """Pick the device and second chain, build and check every PUB."""
    out, _ = _dirs(fake)
    backend = get_backend(fake)
    lay_path = out / "layout.json"
    if lay_path.exists():
        lay = json.loads(lay_path.read_text())
    else:
        ch = choose_chain(backend)
        lay = {"device": backend.name, "chain_b": ch["chain"],
               "chain_b_score": ch["score"],
               "chain_b_calibration": ch["calibration_date"],
               "chosen": _now()}
        lay_path.write_text(json.dumps(lay, indent=2) + "\n")
    dz = design()
    pubs, meta = build_pubs(backend, dz["rows"],
                            {"A": LAYOUT_A, "B": lay["chain_b"]})
    shots = int(np.ceil(1 / PRECISION ** 2))
    n = sum(p[2].shape[0] for p in pubs) * shots
    print(f"  device {lay['device']}; chain A {LAYOUT_A}; "
          f"chain B {lay['chain_b']} (score {lay['chain_b_score']:.4f})")
    print(f"  design digest {dz['digest'][:16]}")
    for d in DEPTHS:
        r = dz["rows"][d]
        print(f"  d={d:>2}  ideal var {np.var(r['ideal_random']):.4f}  "
              f"refs {r['ideal_ref'].tolist()} ({r['ref_draws']} draws)")
    print(f"  {len(pubs)} PUBs, {n} shots per job, about "
          f"{n * SECONDS_PER_SHOT:.0f} s QPU by the fallback rate")
    return 0


def session(fake: bool) -> int:
    """Run the remaining repeats about GAP_S apart."""
    out, _ = _dirs(fake)
    prepare(fake)
    for k in range(1, N_REPEATS + 1):
        if (out / f"job_{k}.json").exists():
            continue
        prev = out / f"job_{k - 1}.json"
        if prev.exists() and not fake:
            t = datetime.fromisoformat(
                json.loads(prev.read_text())["submitted"])
            wait = GAP_S - (datetime.now(timezone.utc) - t).total_seconds()
            if wait > 0:
                print(f"  waiting {wait / 60:.0f} min before job {k}",
                      flush=True)
                time.sleep(wait)
        run_job(k, fake)
    return 0


#------------------------------------------------------------------------------
#
# analysis (pre-specified; see hw_decay_plan.md)
#
#------------------------------------------------------------------------------

def _boot_slope(x, y, rng) -> tuple:
    n = len(x)
    bs = []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        bs.append(np.polyfit(x[i], y[i], 1)[0])
    bs = np.asarray(bs)
    return float(np.std(bs, ddof=1)), [float(v) for v in
                                       np.percentile(bs, [2.5, 97.5])]


def _fits(L, f, se) -> dict:
    """Exponential, straight line, constant: weighted fits and AIC."""
    L, f, se = map(np.asarray, (L, f, se))
    w = 1.0 / se ** 2
    out = {}

    # constant
    #
    c = float(np.sum(w * f) / np.sum(w))
    chi = float(np.sum(w * (f - c) ** 2))
    out["constant"] = {"params": {"c": c}, "chi2": chi, "k": 1}

    # straight line
    #
    p, cov = curve_fit(lambda x, a, b: a + b * x, L, f, sigma=se,
                       absolute_sigma=True)
    chi = float(np.sum(w * (f - (p[0] + p[1] * L)) ** 2))
    out["line"] = {"params": {"a": float(p[0]), "b": float(p[1])},
                   "chi2": chi, "k": 2}

    # exponential f = A q^L
    #
    p, cov = curve_fit(lambda x, A, q: A * q ** x, L, f, sigma=se,
                       absolute_sigma=True, p0=(f[0], 0.98),
                       bounds=([0, 0], [1.5, 1.2]))
    chi = float(np.sum(w * (f - p[0] * p[1] ** L) ** 2))
    dof = len(L) - 2
    scale = max(1.0, chi / dof) if dof > 0 else 1.0
    sq = float(np.sqrt(cov[1, 1] * scale))
    out["exponential"] = {"params": {"A": float(p[0]), "q": float(p[1])},
                          "q_ci95": [float(p[1] - 1.96 * sq),
                                     float(p[1] + 1.96 * sq)],
                          "chi2": chi, "k": 2}

    for m in out.values():
        m["aic"] = m["chi2"] + 2 * m["k"]
        m["reduced_chi2"] = m["chi2"] / max(1, len(L) - m["k"])
    best = min(out, key=lambda m: out[m]["aic"])
    return {"models": out, "best": best}


def _snapshot(job: dict) -> tuple:
    """
    The calibration in effect when the job ran: the after-completion
    snapshot if it was published before execution started, else the
    one captured at submission.
    """
    def _dt(s):
        t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)

    start = ((job.get("metrics") or {}).get("timestamps") or {}).get(
        "running")
    done = job["cal_done"]
    if start and done.get("last_update_date"):
        try:
            if _dt(done["last_update_date"]) <= _dt(start):
                return done, "after"
        except ValueError:
            pass
    return job["cal_submit"], "submit"


def analyze(fake: bool) -> int:
    """
    function: analyze

    description:
     For every job, chain and depth: damping f as the slope of the
     hardware cost on the noiseless cost over the 40 random rows, with
     a paired bootstrap interval; the reference damping (measured over
     noiseless value, averaged over the reference rows); the
     shot-corrected variance ratio against f^2; and the correctly
     priced ESP and the v1.0.0 value. Then, per chain, the three model
     fits of f against depth pooled over the jobs (primary) and per job
     (secondary), the calibration-predicted per-layer rate, and the
     spread between jobs.
    """

    out, _ = _dirs(fake)
    rows = design()["rows"]
    jobs = [json.loads(p.read_text())
            for p in sorted(out.glob("job_*.json"))]
    rng = np.random.default_rng(SEED + 2)
    recs = []
    for job in jobs:
        cal, which = _snapshot(job)
        errs, ro = _calibration_tables(cal)
        prof = None
        try:
            prof = noise_profile(HilbertTrace(job["run_dir"]))
        except Exception:
            pass
        for p in job["pubs"]:
            d = p["depth"]
            y_all = np.asarray(p["evs"])
            x = rows[d]["ideal_random"]
            y = y_all[:N_POINTS]
            y_ref = y_all[N_POINTS:]
            f = float(np.polyfit(x, y, 1)[0])
            se, ci = _boot_slope(x, y, rng)
            shot_var = float(np.mean((1 - y ** 2) / p["shots"]))
            ratio = float((np.var(y) - shot_var) / np.var(x))
            esp = corrected_esp(p["qasm"], errs, ro, p["measured"])
            rec = {
                "repeat": job["repeat"], "chain": p["chain"],
                "layout": p["layout"], "depth": d, "slope_f": f,
                "slope_f_se": se, "slope_f_ci95": ci,
                "r_squared": float(np.corrcoef(x, y)[0, 1] ** 2),
                "ref_damping": float(np.mean(y_ref / rows[d]["ideal_ref"])),
                "ref_values": y_ref.tolist(),
                "ideal_variance": float(np.var(x)),
                "hw_variance": float(np.var(y)),
                "shot_variance": shot_var,
                "variance_ratio_shot_corrected": ratio,
                "f_squared": f * f,
                "esp": esp["esp"], "esp_unpriced": esp["unpriced_gates"],
                "esp_snapshot": which,
                "fhat_v100": (v100_fhat(prof, _parse_qasm(p["qasm"]))
                              if prof else None),
            }
            recs.append(rec)

    # model fits and the calibration-predicted rate
    #
    fits = {}
    for chain in ("A", "B"):
        sub = [r for r in recs if r["chain"] == chain]
        if not sub:
            continue
        L = [r["depth"] for r in sub]
        fits[chain] = {"pooled": _fits(L, [r["slope_f"] for r in sub],
                                       [r["slope_f_se"] for r in sub])}
        per_job, pred = {}, {}
        for k in sorted({r["repeat"] for r in sub}):
            s = [r for r in sub if r["repeat"] == k]
            per_job[k] = _fits([r["depth"] for r in s],
                               [r["slope_f"] for r in s],
                               [r["slope_f_se"] for r in s])
            b = np.polyfit([r["depth"] for r in s],
                           np.log([r["esp"] for r in s]), 1)
            pred[k] = {"q_pred": float(np.exp(b[0])),
                       "A_pred": float(np.exp(b[1]))}
        fits[chain]["per_job"] = per_job
        fits[chain]["calibration_prediction"] = pred
        spread = {}
        for d in DEPTHS:
            v = [r["slope_f"] for r in sub if r["depth"] == d]
            spread[d] = {"min": min(v), "max": max(v),
                         "sd": float(np.std(v, ddof=1)) if len(v) > 1
                         else None}
        fits[chain]["spread_between_jobs"] = spread

    payload = {
        "protocol": "hw_decay_v1", "plan": "experiments/hw_decay_plan.md",
        "seed": SEED, "design_digest": design()["digest"],
        "depths": DEPTHS, "n_points": N_POINTS, "n_ref": N_REF,
        "jobs": [{k: j[k] for k in ("repeat", "job_id", "backend",
                                    "submitted", "finished",
                                    "quantum_seconds", "run_dir")}
                 | {"layouts": sorted({str(p["layout"])
                                       for p in j["pubs"]})}
                 for j in jobs],
        "records": recs, "fits": fits,
    }
    path = out / "results.json"
    path.write_text(json.dumps(payload, indent=1, default=str) + "\n")

    # report
    #
    for chain, fc in fits.items():
        print(f"\n  chain {chain}")
        for r in [r for r in recs if r["chain"] == chain]:
            print(f"   job {r['repeat']} d={r['depth']:>2}  "
                  f"f={r['slope_f']:.3f} [{r['slope_f_ci95'][0]:.3f},"
                  f"{r['slope_f_ci95'][1]:.3f}]  ref={r['ref_damping']:.3f}"
                  f"  var/f^2={r['variance_ratio_shot_corrected'] / max(r['f_squared'], 1e-9):.2f}"
                  f"  esp={r['esp']:.3f}")
        pm = fc["pooled"]
        for name, m in pm["models"].items():
            print(f"   pooled {name:<11} AIC {m['aic']:8.1f}  "
                  f"chi2/dof {m['reduced_chi2']:.2f}  {m['params']}")
        print(f"   best: {pm['best']}; q = "
              f"{pm['models']['exponential']['params']['q']:.4f} "
              f"{pm['models']['exponential']['q_ci95']}")
        for k, p in fc["calibration_prediction"].items():
            print(f"   job {k}: q_pred {p['q_pred']:.4f}; per-job best "
                  f"{fc['per_job'][k]['best']}, q "
                  f"{fc['per_job'][k]['models']['exponential']['params']['q']:.4f}")
    print(f"\n  wrote {path}")
    return 0


def main() -> int:
    """Command-line entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("command",
                    choices=["prepare", "session", "run", "analyze"])
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--fake", action="store_true")
    a = ap.parse_args()
    if a.command == "prepare":
        return prepare(a.fake)
    if a.command == "session":
        return session(a.fake)
    if a.command == "run":
        run_job(a.repeat, a.fake)
        return 0
    return analyze(a.fake)
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
