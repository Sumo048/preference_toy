"""Collect GPROP cells and print per-setting tables and the paired tests.

Usage: python explore/GPROP/analyze.py [--stage 1|2] [--json out.json]
Sources:
  GP1/GP2/GP3/E1 fits:     explore/GPROP/results/<ARM>-<cfg>/<cell>.json
  E1 stored-graft re-evals: explore/GPROP/results/E1-stored/<cell>.json  (E1 seeds 0-4, E5 seeds 5-9)
  B6 / B1a: dev seeds 0-4 k4/alt from explore/E3/results, contig from explore/GPROP/results/E4code (E4 code);
            seeds 5-9 from explore/GPROP/ref/E4 (snapshot of origin/art/confirm-E4 280d2a9).
"""
import argparse, glob, json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
B6 = "a=track60k-w=1-st=20000"
SETTINGS = ["k1", "k2", "k4", "contig", "alt", "pairs", "b3p1"]


def _load(p):
    return json.load(open(p))


def norm(rec, arm):
    if "flips" not in rec:   # e3 / e4 record
        rec = {**rec, "flips": rec.get("retain_noise_flip"), "off": rec.get("mass_off"),
               "gate_retained": 1.0, "params_added": None}
    return {"arm": arm, "D": rec["D"], "Removed": rec["Removed"], "Collateral": rec["Collateral"],
            "off": rec["off"], "flips": rec["flips"], "gate": rec.get("gate_retained"),
            "params": rec.get("params_added"), "frac_changed": rec.get("frac_changed")}


def collect():
    rows = {}   # (arm, setting, seed) -> row
    for d in glob.glob(os.path.join(HERE, "results", "*-*")):
        tag = os.path.basename(d)
        if tag == "E4code":
            continue
        for p in glob.glob(os.path.join(d, "*_n4.json")):
            r = _load(p)
            rows[(tag, r["setting"], r["seed"])] = norm(r, tag)
            if tag.startswith("GP1-") and len(r.get("rounds", [])) == 2:
                # round 1 of a 2-round fit is bit-identical to the 1-round config (same seeds, same targets)
                r1 = {**r, **r["rounds"][0], "params_added": r["params_added"] // 2}
                rows.setdefault((tag + "1", r["setting"], r["seed"]), norm(r1, tag + "1"))
    for setting in SETTINGS:
        for seed in range(10):
            for src in (os.path.join(ROOT, "explore", "E3", "results", setting, f"s{seed}_n4"),
                        os.path.join(HERE, "results", "E4code", setting, f"s{seed}_n4"),
                        os.path.join(HERE, "results", "E4rerun", setting, f"s{seed}_n4"),
                        os.path.join(HERE, "ref", "E4", setting, f"s{seed}_n4")):
                for m, arm in ((B6, "B6"), ("B1a", "B1a")):
                    p = os.path.join(src, f"{m}.json")
                    if os.path.exists(p) and (arm, setting, seed) not in rows:
                        rows[(arm, setting, seed)] = norm(_load(p), arm)
    return rows


def paired(rows, a, b, setting, seeds, key="D"):
    d = [rows[(a, setting, s)][key] - rows[(b, setting, s)][key] for s in seeds
         if (a, setting, s) in rows and (b, setting, s) in rows]
    if len(d) < 2:
        return None
    d = np.array(d)
    return {"n": len(d), "mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d))),
            "lower": int((d < 0).sum())}


def table(rows, arms, settings, seeds):
    out = []
    hdr = f"| setting | arm | n | D | Removed | Collateral | off | flips | gate(ret) | params |"
    out += [hdr, "|" + "---|" * 10]
    for st in settings:
        for a in arms:
            rr = [rows[(a, st, s)] for s in seeds if (a, st, s) in rows]
            if not rr:
                continue
            m = lambda k: np.nanmean([r[k] if r[k] is not None else np.nan for r in rr])
            pa = rr[0]["params"]
            out.append(f"| {st} | {a} | {len(rr)} | {m('D'):.4f} | {m('Removed'):.3f} | {m('Collateral'):.4f} | "
                       f"{m('off'):.4f} | {m('flips'):.4f} | {m('gate'):.3f} | {pa if pa else 'all'} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=1)
    a = ap.parse_args()
    rows = collect()
    seeds = [0, 1, 2] if a.stage == 1 else [5, 6, 7, 8, 9]
    settings = ["k4", "alt", "contig"] if a.stage == 1 else SETTINGS
    arms = sorted({k[0] for k in rows}, key=lambda x: (x[:2] not in ("B6", "B1", "E1"), x))
    print(table(rows, arms, settings, seeds))
    print()
    for st in settings:
        for g in [x for x in arms if x.startswith("GP")]:
            for ref in ("E1-stored" if a.stage == 2 or st != "contig" else "E1-e1", "B6"):
                r = paired(rows, g, ref, st, seeds)
                c = paired(rows, g, ref, st, seeds, key="Collateral")
                f = paired(rows, g, ref, st, seeds, key="flips")
                if r:
                    print(f"{st:7s} {g:10s} - {ref:9s}: dD {r['mean']:+.4f} se {r['se']:.4f} lower {r['lower']}/{r['n']} | "
                          f"dCol {c['mean']:+.4f} | dflips {f['mean']:+.4f}")
