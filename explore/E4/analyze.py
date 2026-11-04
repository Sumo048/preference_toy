"""E4 analysis (frozen in docs/E4_PREREG.md). Reads explore/E4/results, writes the E4 tables to stdout.

Primary test per setting: d_i = D(method) - D(B1a) over cells i = (seed 5-9, npm 1/4/16/64), N = 20.
PASS iff mean(d) < -2 * SE, SE = std(d, ddof=1) / sqrt(N) (complete pairs only; N reported).
"""
import glob, json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.path.join(ROOT, "explore", "E4", "results")
METHOD = "a=track60k-w=1-st=20000"
SETTINGS = ["k1", "k2", "k4", "contig", "alt", "pairs", "b3p1"]
SEEDS, NPMS = [5, 6, 7, 8, 9], [1, 4, 16, 64]
ARMS = ["B0", "B1a", METHOD]


def load():
    out = {}
    for st in SETTINGS:
        for s in SEEDS:
            for n in NPMS:
                for a in ARMS:
                    f = os.path.join(RES, st, f"s{s}_n{n}", f"{a}.json")
                    if os.path.exists(f):
                        out[(st, s, n, a)] = json.load(open(f))
    return out


def paired(R, st, a, b="B1a", npms=NPMS):
    cells = [(s, n) for s in SEEDS for n in npms if (st, s, n, a) in R and (st, s, n, b) in R]
    d = np.array([R[(st, s, n, a)]["D"] - R[(st, s, n, b)]["D"] for s, n in cells])
    if len(d) < 2:
        return len(d), float("nan"), float("nan"), 0, float("nan")
    se = d.std(ddof=1) / math.sqrt(len(d))
    # secondary: seed-clustered SE (mean over npm within seed, n = #seeds)
    ps = [np.mean([R[(st, s, n, a)]["D"] - R[(st, s, n, b)]["D"] for s2, n in cells if s2 == s])
          for s in SEEDS if any(s2 == s for s2, _ in cells)]
    sec = np.std(ps, ddof=1) / math.sqrt(len(ps)) if len(ps) > 1 else float("nan")
    return len(d), float(d.mean()), float(se), int((d < 0).sum()), float(sec)


def mean(R, st, a, key, npms=NPMS):
    v = [R[(st, s, n, a)][key] for s in SEEDS for n in npms if (st, s, n, a) in R]
    return (float(np.mean(v)) if v else float("nan")), len(v)


def table(R, npms=NPMS, title="all npm"):
    L = [f"### Per setting, {title} (seeds 5-9)", "",
         "| Setting | Method | cells | D | Removed | Collateral | mass_off | ΔD vs B1a ± SE | seed-clustered SE | lower than B1a | test |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    verdicts = {}
    for st in SETTINGS:
        for a in ARMS:
            D, nc = mean(R, st, a, "D", npms)
            if nc == 0:
                continue
            Rm, _ = mean(R, st, a, "Removed", npms); C, _ = mean(R, st, a, "Collateral", npms)
            off, _ = mean(R, st, a, "mass_off", npms)
            name = {"B1a": "B1a (R_hopB_bsup_neg_AMdist)", METHOD: f"track-masked BC `{METHOD}`"}.get(a, a)
            if a == "B1a":
                dd, lo, test, sec = "0", "—", "", "—"
            else:
                N, md, se, low, sc = paired(R, st, a, npms=npms)
                dd, lo, sec = f"{md:+.3f} ± {se:.3f}", f"{low}/{N}", f"{sc:.3f}"
                test = ""
                if a == METHOD and npms == NPMS:
                    ok = N == len(SEEDS) * len(NPMS)
                    v = "PASS" if md < -2 * se else "FAIL"
                    test = v + ("" if ok else f" (incomplete, N={N})")
                    verdicts[st] = (v, N, md, se)
            L.append(f"| {st} | {name} | {nc} | {D:.3f} | {Rm:.3f} | {C:.3f} | {off:.3f} | {dd} | {sec} | {lo} | {test} |")
    return "\n".join(L), verdicts


if __name__ == "__main__":
    R = load()
    t, v = table(R)
    print(t)
    print()
    for npm in NPMS:
        print(table(R, [npm], f"npm {npm}")[0]); print()
    print("VERDICTS", json.dumps(v))
