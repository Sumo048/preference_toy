#!/usr/bin/env python3
"""Render results/a3lf/report.json (scripts/a3lf_report.py) into the tables of docs/A3LF.md.
Prints markdown to stdout; the prose around it is written by hand."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
rep = json.loads((REPO / "results" / "a3lf" / "report.json").read_text())


def f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


out = []
out.append("### Per setting (mean ± SE over cells; 'lower' = A3-LF D lower than the comparator, cells with both)\n")
out.append("| setting | arm | cells | D | Removed | Collateral | A3-LF − arm (D) | A3-LF lower in |")
out.append("|---|---|---|---|---|---|---|---|")
for s, e in rep["per_setting"].items():
    for arm in ("A3-LF", "B1a", "B1b", "A3", "B6"):
        v = e.get(arm)
        if not v:
            continue
        diff = v.get("diff")
        out.append(f"| {s} | {arm} | {v['n']} | {f(v['D'][0])} ± {f(v['D'][1])} | {f(v['removed'][0])} | "
                   f"{f(v['collateral'][0])} | {('%+.3f ± %.3f' % tuple(diff)) if diff else ''} | {v.get('lower_in', '')} |")
out.append("")
out.append("### Pre-registered tests (main cells: k4 and alt, npm 4 and 16, seeds 5-9; unit = cell)\n")
out.append("| test | comparison | n | mean diff (A3-LF − comp) | 95% CI | t | p | p (Holm) | lower in | Δ Collateral | result |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|")
names = {"L1": "A3-LF vs B1a (one-sided)", "L2": "A3-LF vs B1b (one-sided)", "L3": "A3-LF vs A3 (two-sided)"}
for k in ("L1", "L2", "L3"):
    t = rep["tests"][k]
    if t.get("n", 0) < 2:
        out.append(f"| {k} | {names[k]} | {t.get('n', 0)} | | | | | | | | too few pairs |")
        continue
    res = ("supported" if t.get("supported") else ("rejects" if t.get("reject") else "not rejected")) \
        if k != "L3" else ("rejects (two-sided)" if t.get("reject") else "not rejected")
    out.append(f"| {k} | {names[k]} | {t['n']} | {t['mean_diff']:+.3f} | [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | "
               f"{t['t']:.2f} | {t['p']:.4f} | {f(t.get('p_holm'), 4)} | {t['lower_in']} of {t['n']} | "
               f"{t['mean_collateral_diff']:+.3f} | {res} |")
out.append("")
out.append("### Robustness (not tested for claims)\n")
out.append("| comparison | unit | scope | n | mean diff | 95% CI | p | lower in |")
out.append("|---|---|---|---|---|---|---|---|")
for k, t in rep["robustness"].items():
    c, u, sc = k.split("_")
    if t.get("n", 0) < 2:
        continue
    out.append(f"| A3-LF − {c} | {u} | {sc} | {t['n']} | {t['mean_diff']:+.3f} | [{t['ci95'][0]:+.3f}, {t['ci95'][1]:+.3f}] | "
               f"{t['p']:.4f} | {t['lower_in']} of {t['n']} |")
out.append("")
a = rep["agreement"]
out.append("### Gate agreement (gate_lf vs oracle gate on the same batch, every gate_lf check)\n")
if a["n"]:
    out.append(f"Agreement {a['agree']} of {a['n']} checks ({100 * a['agree'] / a['n']:.1f}%).\n")
    out.append("| | oracle accept | oracle reject |")
    out.append("|---|---|---|")
    out.append(f"| gate_lf accept | {a['lf_acc_or_acc']} | {a['lf_acc_or_rej']} |")
    out.append(f"| gate_lf reject | {a['lf_rej_or_acc']} | {a['lf_rej_or_rej']} |")
    out.append("")
    out.append("| test | agree | of | % |")
    out.append("|---|---|---|---|")
    for t, (g, n) in a["per_test"].items():
        out.append(f"| {t} | {g} | {n} | {100 * g / n:.1f} |")
out.append("")
c = rep["cost"]
out.append("### Cost (surgeon + triage subagents, `transcripts` price table)\n")
out.append(f"Runs {c['runs']}; mean ${c['mean_cost_usd']:.2f} per run (sd {c['sd_cost_usd']:.2f}); total ${c['total_cost_usd']:.2f}; "
           f"mean {c['mean_sessions']:.1f} surgeon sessions, {c['mean_accepted']:.1f} accepted edits, "
           f"{c['mean_gate_checks']:.1f} gate checks per run. Stop rules: {c['stops']}.\n")
out.append("### Per cell\n")
out.append("| cell | A3-LF D | Removed | Coll. | accepted | stop | cost | B1a | B1b | A3 | B6 | shadow agree |")
out.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
for r in rep["rows"]:
    x = r["A3-LF"]
    if not x or "D" not in x:
        continue
    g = lambda arm: f(r[arm]["D"]) if r.get(arm) and "D" in r[arm] else "-"  # noqa: E731
    sh = x.get("shadow") or {}
    out.append(f"| {r['setting']} s{r['seed']} n{r['npm']} | {f(x['D'])} | {f(x['removed'], 2)} | {f(x['collateral'])} | "
               f"{x['accepted']} | {x['stop']} | {x['cost_usd']:.2f} | {g('B1a')} | {g('B1b')} | {g('A3')} | {g('B6')} | "
               f"{sh.get('agree', '-')}/{sh.get('n', '-')} |")
print("\n".join(out))
