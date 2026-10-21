#!/usr/bin/env python
#
# file: experiments/hw_reanalysis.py
#
# revision history:
#  20261021 (pm): initial version
#  20261021 (pm): add residual diagnostics (drift, curvature)
#
# Re-analysis of the archived hardware data, requested in review. No
# new device time: every number is computed from the sealed traces
# and the recorded parameter draws.
#
#  1. paired:   for each depth, regress the hardware estimates
#               C_hw(theta_i) on the exact noiseless values
#               C_ideal(theta_i) over the shared draws. The slope f is
#               the expectation-value damping factor; f^2 is compared
#               with the hardware-to-ideal variance ratio, and f with
#               the instrument's calibration-based fidelity estimate
#               F_hat for the same transpiled circuit.
#  2. expr:     shot-matched noiseless control for the expressibility
#               shift. The exact pair fidelities are resampled as
#               Binomial(shots, F)/shots, with the hardware shot count,
#               and pushed through the same binned KL estimator, so
#               hardware and control share sampling noise and
#               estimator bias.
#  3. fidelity: the single-point hardware fidelity check against a
#               predict-the-mean baseline.
#
#   python hw_reanalysis.py
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import json
import sys
import warnings

# import third-party modules
#
import numpy as np
from qiskit.quantum_info import Statevector
from scipy import stats

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.analysis.circuit import circuit_structure
from hilbertbench.analysis.noise import noise_profile

# import experiment infrastructure
#
from common import (RESULTS_ROOT, TRACES_ROOT, build_ansatz,
                    pair_observable, save_result)
from hw_frontier import kl_vs_haar

# silence framework deprecation chatter for a clean report
#
warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

N_QUBITS = 4
SEED = 20260615
N_BOOT = 2000

# landscape traces: (device, trace, job date, shots per point)
#
VARIANCE_TRACES = [
    ("ibm_marrakesh", "hw_var_full/20260630_143721_587f61e5",
     "2026-06-30", 1024),
    ("ibm_marrakesh", "hw_var_full/20260711_193615_1d303c23",
     "2026-07-11", 1024),
    ("ibm_fez", "hw_onset_final/20260618_060633_a2a2731d",
     "2026-06-18", 512),
]

# expressibility runs: reproduce the recorded draws exactly
#
EXPR_RUNS = [
    {"device": "ibm_marrakesh", "result": "hw_expr_full",
     "rng_seed": SEED + 7, "depths": [1, 2, 4, 6], "pairs": 500,
     "shots": 256, "bins": 75},
    {"device": "ibm_fez", "result": "hw_expr_final",
     "rng_seed": SEED, "depths": [1, 2, 4, 6, 8], "pairs": 180,
     "shots": 256, "bins": 30},
]
N_SHOT_RESAMPLES = 500

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------


def _ideal_values(depth: int, params: np.ndarray) -> np.ndarray:
    """Exact <Z0 Z1> for each parameter row."""
    qc, _ = build_ansatz(N_QUBITS, depth, "linear")
    obs = pair_observable(N_QUBITS)
    return np.array([
        float(Statevector(qc.assign_parameters(p))
              .expectation_value(obs).real)
        for p in params
    ])


def _fhat_by_entanglers(trace: HilbertTrace) -> dict:
    """
    function: _fhat_by_entanglers

    arguments:
     trace: a sealed hardware trace

    return:
     {entangling-gate count: F_hat} for every recorded circuit

    description:
     Applies the frozen noise analyzer's product formula, with its
     own scoped mean error rates, to each recorded circuit (the
     analyzer itself reports only the largest circuit).
    """

    prof = noise_profile(trace)
    e1 = prof["gate_error_1q_mean"]
    e2 = prof["gate_error_2q_mean"]
    ro = prof["readout_error"]["mean"]
    out = {}
    for c in circuit_structure(trace)["circuits"]:
        n_meas = c["num_measurements"] or c["num_qubits"]
        out[c["entangling_gates"]] = float(
            (1 - e1) ** c["single_qubit_gates"]
            * (1 - e2) ** c["entangling_gates"]
            * (1 - ro) ** n_meas)
    return out


def paired() -> list:
    """
    function: paired

    return:
     one record per (device, depth) landscape PUB

    description:
     Paired regression of hardware on ideal over the shared draws,
     with paired bootstrap intervals for the slope, the variance
     ratio, and the ideal variance.
    """

    m_to_depth = {build_ansatz(N_QUBITS, d, "linear")[1]: d
                  for d in range(1, 25)}
    rng = np.random.default_rng(SEED)
    records = []
    for device, rel, date, shots in VARIANCE_TRACES:
        trace = HilbertTrace(str(TRACES_ROOT / rel))
        fhat = _fhat_by_entanglers(trace)
        for sp in trace.completed():
            P = np.atleast_2d(np.asarray(sp.parameters, dtype=float))
            y = np.asarray(sp.outcome, dtype=float).ravel()
            if P.shape[0] < 20 or P.shape[1] not in m_to_depth:
                continue
            d = m_to_depth[P.shape[1]]
            x = _ideal_values(d, P)
            n = len(x)

            # point estimates
            #
            f = float(np.polyfit(x, y, 1)[0])
            ratio = float(np.var(y) / np.var(x))
            shot_var = float(np.mean((1.0 - y ** 2) / shots))
            ratio_corr = float((np.var(y) - shot_var) / np.var(x))
            r2 = float(np.corrcoef(x, y)[0, 1] ** 2)

            # residual diagnostics: scatter against shot noise, drift
            # along the record order, and curvature (quadratic F-test)
            #
            b1, b0 = np.polyfit(x, y, 1)
            res = y - (b0 + b1 * x)
            res2 = y - np.polyval(np.polyfit(x, y, 2), x)
            f_quad = ((res @ res - res2 @ res2)
                      / (res2 @ res2 / (n - 3)))
            order = stats.spearmanr(np.arange(n), res)

            # paired bootstrap over parameter draws
            #
            bs_f, bs_ratio, bs_vx = [], [], []
            for _ in range(N_BOOT):
                i = rng.integers(0, n, n)
                bs_f.append(np.polyfit(x[i], y[i], 1)[0])
                bs_ratio.append(np.var(y[i]) / np.var(x[i]))
                bs_vx.append(np.var(x[i]))
            ci = lambda v: [float(q) for q in np.percentile(v, [2.5, 97.5])]

            rec = {
                "device": device, "job_date": date, "depth": d,
                "n_points": n, "shots_per_point": shots,
                "slope_f": f, "slope_f_ci95": ci(bs_f),
                "f_squared": f * f,
                "variance_ratio": ratio, "variance_ratio_ci95":
                    ci(bs_ratio),
                "variance_ratio_shot_corrected": ratio_corr,
                "ideal_variance": float(np.var(x)),
                "ideal_variance_ci95": ci(bs_vx),
                "hw_variance": float(np.var(y)),
                "r_squared": r2,
                "residual_variance": float(np.var(res)),
                "shot_variance": shot_var,
                "residual_order_spearman": float(order.statistic),
                "residual_order_p": float(order.pvalue),
                "quadratic_term_p": float(1 - stats.f.cdf(f_quad, 1, n - 3)),
                "fhat": fhat.get(3 * d),
            }
            records.append(rec)
            print(f"  {device:<14} {date} d={d:>2} n={n:>3}  "
                  f"f={f:.3f} [{rec['slope_f_ci95'][0]:.3f},"
                  f"{rec['slope_f_ci95'][1]:.3f}]  f^2={f*f:.3f}  "
                  f"ratio={ratio:.3f} [{rec['variance_ratio_ci95'][0]:.3f},"
                  f"{rec['variance_ratio_ci95'][1]:.3f}]  "
                  f"Fhat={rec['fhat']:.3f}")

    # exit gracefully
    #
    return records
#
# end of function


def expr() -> list:
    """
    function: expr

    return:
     one record per (device, depth) expressibility PUB

    description:
     Reproduces the recorded pair draws, checks that the exact KL
     matches the archived value, then builds the shot-matched
     noiseless control distribution of the KL estimate.
    """

    records = []
    for run in EXPR_RUNS:
        archived = {r["depth"]: r for r in json.loads(
            (RESULTS_ROOT / run["result"] / "results.json")
            .read_text())["records"]}
        rng = np.random.default_rng(run["rng_seed"])
        srng = np.random.default_rng(SEED + 99)
        for d in run["depths"]:
            qc, n_params = build_ansatz(N_QUBITS, d, "linear")
            a = rng.uniform(0, 2 * np.pi, (run["pairs"], n_params))
            b = rng.uniform(0, 2 * np.pi, (run["pairs"], n_params))
            fids = np.array([
                float(np.abs(Statevector(qc.assign_parameters(bb)).inner(
                    Statevector(qc.assign_parameters(aa)))) ** 2)
                for aa, bb in zip(a, b)
            ])
            exact = kl_vs_haar(fids, N_QUBITS, run["bins"], SEED + d)
            arch = archived[d]
            assert abs(exact["kl"] - arch["kl_sim"]) < 1e-9, \
                f"draw reproduction failed at depth {d}"

            # shot-matched control: Binomial(shots, F)/shots
            #
            kls = []
            for _ in range(N_SHOT_RESAMPLES):
                sampled = srng.binomial(run["shots"], fids) / run["shots"]
                kls.append(kl_vs_haar(sampled, N_QUBITS, run["bins"],
                                      SEED + d)["kl"])
            kls = np.array(kls)
            lo, hi = np.percentile(kls, [2.5, 97.5])
            p_ge = float(np.mean(kls >= arch["kl_hw"]))
            rec = {
                "device": run["device"], "depth": d,
                "pairs": run["pairs"], "shots": run["shots"],
                "bins": run["bins"],
                "kl_hw": arch["kl_hw"],
                "kl_exact_ideal": exact["kl"],
                "kl_shot_matched_mean": float(kls.mean()),
                "kl_shot_matched_95": [float(lo), float(hi)],
                "frac_control_ge_hw": p_ge,
            }
            records.append(rec)
            print(f"  {run['device']:<14} d={d}  KL hw={arch['kl_hw']:.3f}"
                  f"  exact={exact['kl']:.3f}  shot-matched="
                  f"{kls.mean():.3f} [{lo:.3f},{hi:.3f}]  "
                  f"P(control>=hw)={p_ge:.3f}")

    # exit gracefully
    #
    return records
#
# end of function


def fidelity() -> dict:
    """
    function: fidelity

    return:
     RMSE of the F_hat prediction and of a predict-the-mean baseline
     over the seven single-point hardware fidelity checks

    description:
     The six hw_fidelity points plus the smoke-test point, exactly
     as plotted in the fidelity figure. The smoke point is the
     depth-7 PUB (21 entangling gates); its prediction is the frozen
     analyzer's F_hat for that circuit, recomputed from the trace.
    """

    recs = json.loads((RESULTS_ROOT / "hw_fidelity" / "results.json")
                      .read_text())["records"]
    smoke = json.loads((RESULTS_ROOT / "hw_smoke" / "results.json")
                       .read_text())
    pred = [r["predicted_fidelity"] for r in recs]
    obs = [r["observed_attenuation"] for r in recs]
    depths = [r["depth"] for r in recs]
    point = next(p for p in smoke["fidelity_points"] if p["depth"] == 7)
    fhat = _fhat_by_entanglers(HilbertTrace(smoke["run_dir"]))[21]
    pred.append(fhat)
    obs.append(point["observed_attenuation"])
    depths.append(7)
    pred, obs = np.array(pred), np.array(obs)
    out = {
        "n": int(len(pred)),
        "depths": depths,
        "rmse_prediction": float(np.sqrt(np.mean((obs - pred) ** 2))),
        "rmse_predict_mean": float(np.sqrt(np.mean(
            (obs - obs.mean()) ** 2))),
        "pearson_r": float(np.corrcoef(pred, obs)[0, 1]),
        "pearson_r_without_smoke_point": float(
            np.corrcoef(pred[:-1], obs[:-1])[0, 1]),
        "predicted": pred.tolist(), "observed": obs.tolist(),
    }
    print(f"  fidelity: n={out['n']} RMSE pred={out['rmse_prediction']:.3f}"
          f" vs predict-the-mean={out['rmse_predict_mean']:.3f}; "
          f"r={out['pearson_r']:.2f} "
          f"(without smoke point {out['pearson_r_without_smoke_point']:.2f})")
    return out
#
# end of function


def main() -> int:
    print("== paired regression")
    p = paired()
    print("== shot-matched expressibility control")
    e = expr()
    print("== single-point fidelity check")
    f = fidelity()
    path = save_result("hw_reanalysis", {
        "protocol": "hw_reanalysis_v1",
        "status": ("post-hoc re-analysis of archived hardware traces; "
                   "no new device time"),
        "paired": p, "expressibility": e, "fidelity": f,
    })
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
#
# end of file
