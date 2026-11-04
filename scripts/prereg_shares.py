#!/usr/bin/env python3
"""Print the PREREG.md grid shares (W2b, Phase 5): every cell for every arm per share, in run order.

    python scripts/prereg_shares.py --cost_per_run 2.8 > /tmp/shares.md

Priority items (PLAN 9.1, Amendment 1.3). Seeds 5-9. Projected cost = LLM runs x Phase 4 mean A3 cost per
run (A1, A2 and the exploratory arms are projected at the same per-run cost; code arms cost CPU only).
"""
import argparse

SEEDS = (5, 6, 7, 8, 9)
ITEMS = [
    (1, [("k4", (4, 16))]),
    (2, [("alt", (4, 16))]),
    (3, [("k1", (4, 16))]),
    (4, [("contig", (4, 16))]),
    (5, [("pairs", (4, 16)), ("b3p1", (4, 16))]),
    (6, [("k4", (1, 64)), ("alt", (1, 64))]),
    (7, [("k2", (4, 16))]),
]
CODE = ("B4", "R0", "D1", "B5")


def cells(item):
    out = []
    for setting, npms in dict(ITEMS)[item]:
        for seed in SEEDS:
            for npm in npms:
                out.append(f"{setting} s{seed} n{npm}")
    return out


def rows_for(share):
    """[(row label, llm arms, cells, code arms run here)] in run order."""
    if share == 1:
        return [("item 1: k4 npm 4,16", ("A3", "A2", "A1"), cells(1), CODE),
                ("item 2: alt npm 4,16", ("A3", "A2", "A1"), cells(2), CODE)]
    if share == 2:
        return [("item 3: k1 npm 4,16", ("A3", "A2", "A1"), cells(3), CODE),
                ("item 4: contig npm 4,16", ("A3",), cells(4), CODE),
                ("item 5a: pairs npm 4,16", ("A3",), [c for c in cells(5) if c.startswith("pairs")], CODE)]
    return [("exploratory: k4 npm 16", ("A3-nobranch", "A3-sonnet", "A3-nogate"),
             [c for c in cells(1) if c.endswith("n16")], ()),
            ("item 5b: b3p1 npm 4,16", ("A3",), [c for c in cells(5) if c.startswith("b3p1")], CODE),
            ("item 6: k4 and alt npm 1,64", ("A3",), cells(6), CODE),
            ("item 7: k2 npm 4,16", ("A3",), cells(7), CODE)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost_per_run", type=float, required=True)
    a = ap.parse_args()
    out = []
    tot = {}
    for share in (1, 2, 3):
        rs = rows_for(share)
        n_llm = sum(len(arms) * len(cs) for _, arms, cs, _ in rs)
        tot[share] = n_llm
        out.append(f"### Share {share} (W3-{share}): {n_llm} LLM runs, projected ${n_llm * a.cost_per_run:.0f}")
        out.append("")
        out.append("| order | row | LLM arms (in this order per cell) | code arms | cells (in order) |")
        out.append("|---|---|---|---|---|")
        for i, (label, arms, cs, code) in enumerate(rs, 1):
            code_s = ", ".join(code) if code else "none here (share 1 runs them on these cells)"
            out.append(f"| {i} | {label} | {', '.join(arms)} | {code_s} | {'; '.join(cs)} |")
        a3 = [c for _, arms, cs, _ in rs if "A3" in arms for c in cs]
        out.append("")
        out.append(f"Replicate order (`A3_rep1`, after the share is done): {'; '.join(a3)}.")
        out.append("")
    out.append(f"Totals: {tot[1]} / {tot[2]} / {tot[3]} LLM runs; projected ${sum(tot.values()) * a.cost_per_run:.0f} "
               f"at ${a.cost_per_run:.2f} per run (Phase 4 mean A3 run, surgeon + triage subagents; main-agent cost "
               f"not included).")
    print("\n".join(out))


if __name__ == "__main__":
    main()
