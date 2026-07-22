#!/usr/bin/env python
#
# file: experiments/study_g_score.py
#
# revision history:
#  20260715 (am): initial version
#
# Study G scorer -- committed BEFORE the corpus was generated, so no
# scoring choice postdates the data.
#
# Two subcommands:
#
#   rule   compute the frozen Study F rule's outputs on the BLINDED
#          corpus and write study_g_rule_outputs.json + its SHA-256;
#          the hash is published before the diagnostician receives
#          the corpus, so the comparator cannot react to her sheet
#
#   score  after the diagnostician's sheet is hash-committed and the
#          answer key unsealed: verify every commitment, join key +
#          sheet + rule outputs, and compute the registered outcomes
#          (exact McNemar primary; psychometric overlay, H2 failure-
#          class recovery, mixture flag recall, healthy false
#          positives, H3 confidence calibration)
#
#   python study_g_score.py rule  --blinded traces/blinded_g
#   python study_g_score.py score --blinded traces/blinded_g \
#       --key answer_key.json --commitment answer_key.sha256 \
#       --diagnosis diagnosis_sheet_filled.json
#------------------------------------------------------------------------------

# future imports must come first
#
from __future__ import annotations

# import system modules
#
import argparse
import hashlib
import json
import math
import sys
import warnings
from pathlib import Path

# import hilbertbench modules
#
from hilbertbench import HilbertTrace
from hilbertbench.analysis import summary

# the comparator is the frozen Study F rule, imported unchanged
#
from study_f_hardmode import apply_rule, _verify_commitment

# silence framework deprecation chatter for a clean report
#
warnings.filterwarnings("ignore")

HERE = Path(__file__).parent
RULE_OUT_PATH = HERE / "study_g_rule_outputs.json"
RULE_SHA_PATH = HERE / "study_g_rule_outputs.sha256"

# transition regions measured in Study F (levels whose evidence is
# genuinely borderline); used only for the H3 calibration report
#
TRANSITION_LEVELS = {
    "barren_plateau": {7, 8},
    "shot_starved": {0.12, 0.18, 0.28},
    "noise_dominated": {12, 14, 16},
}
ANCHOR_LEVELS = {
    "barren_plateau": {11},
    "shot_starved": {0.45},
    "noise_dominated": {20},
}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wilson(successes: int, n: int) -> tuple:
    """95% Wilson score interval (same formula as blind_corpus.py)."""
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(
        p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _blind_run_dirs(blinded: Path) -> list:
    """Blinded run directories (8-hex-char ids) under the corpus."""
    return sorted(
        d for d in blinded.iterdir()
        if d.is_dir() and (d / "trace.json").is_file()
    )


def cmd_rule(args: argparse.Namespace) -> int:
    """
    function: cmd_rule

    description:
     Applies the frozen rule to every blinded run and writes the
     outputs plus their SHA-256. The Study F protocol commitment is
     verified first, so the rule provably matches commit 5c25350.
    """

    digest = _verify_commitment()
    print(f"study F rule commitment verified: {digest}")

    blinded = Path(args.blinded)
    outputs = {}
    for run_dir in _blind_run_dirs(blinded):
        report = summary(HilbertTrace(str(run_dir)))
        flags, primary = apply_rule(report)
        mea_status = (report.get("measurement") or {}).get("status", "")
        opt_status = (report.get("optimization") or {}).get("status", "")
        outputs[run_dir.name] = {
            "flags": flags, "primary": primary,
            "measurement_status": mea_status,
            "optimization_status": opt_status,
        }
        print(f"  {run_dir.name}: primary={primary} flags={flags}")

    text = json.dumps(
        {"protocol": "study_g_rule_outputs_v1",
         "rule_commitment": digest, "outputs": outputs},
        indent=2, sort_keys=True) + "\n"
    RULE_OUT_PATH.write_text(text)
    out_digest = hashlib.sha256(text.encode()).hexdigest()
    RULE_SHA_PATH.write_text(out_digest + "\n")
    print(f"\nwrote {RULE_OUT_PATH}")
    print(f"rule-outputs commitment (publish before diagnosis): "
          f"{out_digest}")
    return 0


def _correct(primary, truth_flags) -> bool:
    """Registered primary-correctness: healthy iff no mechanism was
    planted; otherwise any planted mechanism counts as correct."""
    if not truth_flags:
        return primary == "healthy"
    return primary in truth_flags


def cmd_score(args: argparse.Namespace) -> int:
    """
    function: cmd_score

    description:
     Verifies all commitments, joins the answer key, the
     diagnostician's sheet, and the pre-committed rule outputs, and
     computes the registered Study G outcomes.
    """

    # verify the answer key and the rule outputs against commitments
    #
    key_text = Path(args.key).read_text()
    expected = Path(args.commitment).read_text().strip()
    if hashlib.sha256(key_text.encode()).hexdigest() != expected:
        print("error: answer key does NOT match its commitment")
        return 1
    print(f"answer-key commitment verified: {expected}")

    rule_text = RULE_OUT_PATH.read_text()
    rule_expected = RULE_SHA_PATH.read_text().strip()
    if hashlib.sha256(rule_text.encode()).hexdigest() != rule_expected:
        print("error: rule outputs do NOT match their commitment")
        return 1
    print(f"rule-outputs commitment verified: {rule_expected}")

    key = json.loads(key_text)["corpus"]
    sheet = json.loads(Path(args.diagnosis).read_text())
    rule = json.loads(rule_text)["outputs"]

    missing = sorted((set(key) - set(sheet)) | (set(key) - set(rule)))
    if missing:
        print(f"error: runs missing from sheet or rule: {missing}")
        return 1

    # join per run
    #
    rows = []
    for bid, truth in sorted(key.items()):
        d, r = sheet[bid], rule[bid]
        rows.append({
            "id": bid,
            "arm": truth["arm"], "level": truth["level"],
            "truth_flags": truth["truth_flags"],
            "h_primary": d.get("primary"),
            "h_secondary": d.get("secondary"),
            "h_conf": d.get("confidence"),
            "r_primary": r["primary"], "r_flags": r["flags"],
            "r_measurement": r["measurement_status"],
            "r_optimization": r["optimization_status"],
            "h_ok": _correct(d.get("primary"), truth["truth_flags"]),
            "r_ok": _correct(r["primary"], truth["truth_flags"]),
        })

    # PRIMARY: exact one-sided McNemar on discordant pairs
    #
    b = sum(1 for x in rows if x["h_ok"] and not x["r_ok"])
    c = sum(1 for x in rows if x["r_ok"] and not x["h_ok"])
    n_disc = b + c
    p_mcnemar = (sum(math.comb(n_disc, k) * 0.5 ** n_disc
                     for k in range(b, n_disc + 1))
                 if n_disc else 1.0)
    h_acc = sum(x["h_ok"] for x in rows) / len(rows)
    r_acc = sum(x["r_ok"] for x in rows) / len(rows)

    result = {
        "protocol": "study_g_score_v1",
        "n_runs": len(rows),
        "primary_mcnemar": {
            "human_correct_rule_wrong_b": b,
            "rule_correct_human_wrong_c": c,
            "one_sided_exact_p": p_mcnemar,
            "human_accuracy": h_acc, "rule_accuracy": r_acc,
        },
    }
    print(f"\n== PRIMARY: McNemar ==")
    print(f"  human {h_acc:.3f} vs rule {r_acc:.3f}; discordant "
          f"b={b} c={c}; one-sided exact p = {p_mcnemar:.4g}")

    # SECONDARY 1: psychometric overlay (detection of the planted
    # mechanism per arm and level, human {primary,secondary} vs rule
    # flags)
    #
    overlay = {}
    sweep_arms = ("barren_plateau", "shot_starved", "noise_dominated")
    for arm in sweep_arms:
        sub = [x for x in rows if x["arm"] == arm]
        levels = sorted({x["level"] for x in sub})
        overlay[arm] = []
        for lv in levels:
            ss = [x for x in sub if x["level"] == lv]
            h = sum(1 for x in ss
                    if arm in (x["h_primary"], x["h_secondary"]))
            r = sum(1 for x in ss if arm in x["r_flags"])
            overlay[arm].append({
                "level": lv, "n": len(ss),
                "human_detect": h, "rule_detect": r,
                "human_ci95": list(_wilson(h, len(ss))),
                "rule_ci95": list(_wilson(r, len(ss))),
            })
    result["psychometric_overlay"] = overlay

    # SECONDARY 2 (H2): the Study F rule-failure classes
    #
    marginal = [x for x in rows
                if x["r_measurement"].startswith("Marginal")
                and "shot_starved" in x["truth_flags"]]
    traps = [x for x in rows
             if x["arm"] == "healthy" and x["level"] == 1]
    result["h2_failure_classes"] = {
        "marginal_snr_runs": {
            "n": len(marginal),
            "human_correct": sum(x["h_ok"] for x in marginal),
            "rule_correct": sum(x["r_ok"] for x in marginal),
        },
        "converged_trap_runs": {
            "n": len(traps),
            "human_correct": sum(x["h_ok"] for x in traps),
            "rule_correct": sum(x["r_ok"] for x in traps),
        },
    }

    # SECONDARY 3: two-mechanism completeness (flag-set recall)
    #
    mixes = [x for x in rows if len(x["truth_flags"]) == 2]
    mix_recall = {"n": len(mixes), "human": {}, "rule": {}}
    for mech in ("barren_plateau", "shot_starved", "noise_dominated"):
        planted = [x for x in mixes if mech in x["truth_flags"]]
        if not planted:
            continue
        mix_recall["human"][mech] = sum(
            1 for x in planted
            if mech in (x["h_primary"], x["h_secondary"])
        ) / len(planted)
        mix_recall["rule"][mech] = sum(
            1 for x in planted if mech in x["r_flags"]
        ) / len(planted)
    result["mixture_flag_recall"] = mix_recall

    # SECONDARY 4: false positives on mechanism-free runs
    #
    frees = [x for x in rows if not x["truth_flags"]]
    result["healthy_false_positives"] = {
        "n": len(frees),
        "human": sum(1 for x in frees if x["h_primary"] != "healthy"),
        "rule": sum(1 for x in frees if x["r_primary"] != "healthy"),
    }

    # SECONDARY 5 (H3): confidence calibration
    #
    # ADDENDUM (disclosed): as committed in b35ee2b this block recorded
    # only the group means and omitted the one-sided Mann-Whitney U
    # named in the OSF registration. The test was specified in the
    # registration BEFORE any data existed; only the implementation was
    # incomplete, so adding it here changes no analysis decision. See
    # study_g_unsealing_log.md, Disclosures item 1.
    #
    def _conf(xs):
        vals = [x["h_conf"] for x in xs if x["h_conf"] is not None]
        return (sum(vals) / len(vals)) if vals else None

    trans = [x for x in rows
             if x["level"] in TRANSITION_LEVELS.get(x["arm"], set())]
    anchor = [x for x in rows
              if x["level"] in ANCHOR_LEVELS.get(x["arm"], set())]
    t_vals = [x["h_conf"] for x in trans if x["h_conf"] is not None]
    a_vals = [x["h_conf"] for x in anchor if x["h_conf"] is not None]

    # one-sided (transition < anchor), as registered
    u_stat = p_value = None
    if t_vals and a_vals:
        from scipy.stats import mannwhitneyu
        u_stat, p_value = mannwhitneyu(
            t_vals, a_vals, alternative="less")
        u_stat, p_value = float(u_stat), float(p_value)

    result["h3_confidence"] = {
        "transition_mean": _conf(trans),
        "anchor_mean": _conf(anchor),
        "n_transition": len(trans), "n_anchor": len(anchor),
        "mannwhitney_u": u_stat,
        "mannwhitney_p_one_sided": p_value,
    }

    result["rows"] = rows
    out = Path(args.out) if args.out else HERE / "results" / \
        "study_g" / "results.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="study_g_score.py",
        description="Study G: rule commitment + registered scoring",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rule = sub.add_parser("rule", help="commit rule outputs")
    p_rule.add_argument("--blinded", required=True)
    p_rule.set_defaults(func=cmd_rule)

    p_score = sub.add_parser("score", help="registered scoring")
    p_score.add_argument("--blinded", required=True)
    p_score.add_argument("--key", required=True)
    p_score.add_argument("--commitment", required=True)
    p_score.add_argument("--diagnosis", required=True)
    p_score.add_argument("--out", default=None)
    p_score.set_defaults(func=cmd_score)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
#
# end of file
