#!/usr/bin/env python
#
# file: experiments/hw_decay_posthoc.py
#
# revision history:
#  20261022 (pm): initial version
#
# Post hoc analysis of the depth-decay jobs, added after review. The
# three jobs reuse the same 40 parameter rows per depth, so the error
# from sampling those rows is shared between jobs and pooling the jobs
# counts it three times. This script resamples the rows jointly across
# jobs (the same row indices in every job) to give honest intervals for
# the pooled and per-job per-layer factor q and for the AIC margin of the
# exponential over the straight line, and it reports the job-level
# spread of q. Not part of the planned analysis.
#
# Usage:
#   python hw_decay_posthoc.py
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
from scipy import stats

# import experiment infrastructure
#
import hw_decay as reg
from common import RESULTS_ROOT

warnings.filterwarnings("ignore")

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

N_BOOT = 2000
SEED = 20261023

#------------------------------------------------------------------------------
#
# functions are listed here
#
#------------------------------------------------------------------------------

def main() -> int:
    """Joint-row bootstrap, per-job and job-level q, per-depth residuals."""
    out = RESULTS_ROOT / "hw_decay"
    res = json.loads((out / "results.json").read_text())
    jobs = [json.loads(p.read_text()) for p in sorted(out.glob("job_*.json"))]
    rows = reg.design()["rows"]
    rng = np.random.default_rng(SEED)
    n = reg.N_POINTS
    report = {"note": "post hoc, after review; not in the planned analysis"}

    for chain in ("A", "B"):
        # hardware values y[job][depth] and noiseless x[depth]
        #
        y = {j["repeat"]: {p["depth"]: np.asarray(p["evs"])[:n]
                           for p in j["pubs"] if p["chain"] == chain}
             for j in jobs}
        x = {d: rows[d]["ideal_random"] for d in reg.DEPTHS}
        se = {(r["repeat"], r["depth"]): r["slope_f_se"]
              for r in res["records"] if r["chain"] == chain}
        ks = sorted(y)

        def var_rate(idx):
            # log-linear fit of the shot-corrected variance ratio against
            # depth over the saturated depths (8 to 48), all jobs
            L, lr = [], []
            for k in ks:
                for d in reg.DEPTHS:
                    if d < 8:
                        continue
                    yy, xx = y[k][d][idx[d]], x[d][idx[d]]
                    sv = np.mean((1 - yy ** 2) / 511)
                    L.append(d)
                    lr.append(np.log((np.var(yy) - sv) / np.var(xx)))
            return float(np.exp(np.polyfit(L, lr, 1)[0]))

        def fit_all(idx):
            f = {(k, d): float(np.polyfit(x[d][idx[d]], y[k][d][idx[d]],
                                          1)[0])
                 for k in ks for d in reg.DEPTHS}
            L = [d for k in ks for d in reg.DEPTHS]
            pooled = reg._fits(L, [f[(k, d)] for k in ks
                                   for d in reg.DEPTHS],
                               [se[(k, d)] for k in ks for d in reg.DEPTHS])
            per = {k: reg._fits(reg.DEPTHS, [f[(k, d)] for d in reg.DEPTHS],
                                [se[(k, d)] for d in reg.DEPTHS])
                   for k in ks}
            return f, pooled, per

        ident = {d: np.arange(n) for d in reg.DEPTHS}
        f0, pooled0, per0 = fit_all(ident)
        s0 = var_rate(ident)
        bq, bd, bs, bper = [], [], [], {k: [] for k in ks}
        for _ in range(N_BOOT):
            idx = {d: rng.integers(0, n, n) for d in reg.DEPTHS}
            _, pooled, per = fit_all(idx)
            bs.append(var_rate(idx))
            m = pooled["models"]
            bq.append(m["exponential"]["params"]["q"])
            bd.append(m["line"]["aic"] - m["exponential"]["aic"])
            for k in ks:
                bper[k].append(per[k]["models"]["exponential"]["params"]["q"])
        pct = lambda v: [float(t) for t in np.percentile(v, [2.5, 97.5])]

        # job-level spread: mean and t interval over the per-job q
        #
        qj = np.array([per0[k]["models"]["exponential"]["params"]["q"]
                       for k in ks])
        tq = stats.t.ppf(0.975, len(qj) - 1) * qj.std(ddof=1) / np.sqrt(
            len(qj))

        # per-depth residual of the pooled exponential, averaged over jobs
        #
        ex = pooled0["models"]["exponential"]["params"]
        resid = {d: float(np.mean([f0[(k, d)] for k in ks])
                          - ex["A"] * ex["q"] ** d) for d in reg.DEPTHS}

        # quasi-AIC: chi^2 scaled by the exponential fit's reduced chi^2
        #
        m0 = pooled0["models"]
        chat = max(1.0, m0["exponential"]["reduced_chi2"])
        qaic = {name: m["chi2"] / chat + 2 * m["k"]
                for name, m in m0.items()}
        report[chain] = {
            "variance_ratio_rate_d8_48": s0,
            "variance_ratio_rate_sqrt": float(np.sqrt(s0)),
            "variance_ratio_rate_ci95_joint_row_bootstrap": pct(bs),
            "qaic": qaic,
            "qaic_line_minus_exp": qaic["line"] - qaic["exponential"],
            "overdispersion_chat": chat,
            "pooled_q": ex["q"],
            "pooled_q_ci95_joint_row_bootstrap": pct(bq),
            "aic_line_minus_exp": m0["line"]["aic"] - m0["exponential"]["aic"],
            "aic_line_minus_exp_bootstrap_ci95": pct(bd),
            "aic_line_minus_exp_bootstrap_frac_positive":
                float(np.mean(np.asarray(bd) > 0)),
            "per_job_q": {k: per0[k]["models"]["exponential"]["params"]["q"]
                          for k in ks},
            "per_job_q_ci95_joint_row_bootstrap": {k: pct(bper[k])
                                                   for k in ks},
            "job_level_mean_q": float(qj.mean()),
            "job_level_q_ci95_t": [float(qj.mean() - tq),
                                   float(qj.mean() + tq)],
            "mean_residual_by_depth": resid,
        }
        r = report[chain]
        print(f"  chain {chain}: pooled q {r['pooled_q']:.4f} joint-row "
              f"[{r['pooled_q_ci95_joint_row_bootstrap'][0]:.4f}, "
              f"{r['pooled_q_ci95_joint_row_bootstrap'][1]:.4f}]; "
              f"AIC(line)-AIC(exp) {r['aic_line_minus_exp']:.1f} "
              f"[{r['aic_line_minus_exp_bootstrap_ci95'][0]:.1f}, "
              f"{r['aic_line_minus_exp_bootstrap_ci95'][1]:.1f}], "
              f"> 0 in {100 * r['aic_line_minus_exp_bootstrap_frac_positive']:.0f}%")
        for k in ks:
            c = r["per_job_q_ci95_joint_row_bootstrap"][k]
            print(f"    job {k}: q {r['per_job_q'][k]:.4f} "
                  f"[{c[0]:.4f}, {c[1]:.4f}]")
        print(f"    job level: {r['job_level_mean_q']:.4f} "
              f"[{r['job_level_q_ci95_t'][0]:.4f}, "
              f"{r['job_level_q_ci95_t'][1]:.4f}]")
        print(f"    variance-ratio rate (8-48) {s0:.4f} sqrt {np.sqrt(s0):.4f} "
              f"[{pct(bs)[0]:.4f}, {pct(bs)[1]:.4f}]; quasi-AIC margin "
              f"{r['qaic_line_minus_exp']:.1f} (c-hat {chat:.2f})")
        print("    mean residual by depth: " + ", ".join(
            f"{d}:{v:+.3f}" for d, v in resid.items()))

    (out / "posthoc.json").write_text(json.dumps(report, indent=1) + "\n")
    print(f"  wrote {out / 'posthoc.json'}")
    return 0
#
# end of function


# begin gracefully
#
if __name__ == "__main__":
    sys.exit(main())
#
# end of file
