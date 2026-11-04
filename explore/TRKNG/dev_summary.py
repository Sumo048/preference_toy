"""Stage-1 summary of mask quality over the development cells (seeds 0-2, npm 4)."""
import glob, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
rows = []
for f in glob.glob(os.path.join(HERE, "results", "*", "s[0-4]_n*", "masks_*.json")):
    m = json.load(open(f))
    for name, r in m["masks"].items():
        rows.append({"setting": m["setting"], "seed": m["seed"], "npm": m["npm"], "name": name.rsplit("-N", 1)[0],
                     "N": r["N"], "prec": r["prec"], "rec": r["rec"], "kfb": r["keep_frac_bad"],
                     "keep": r["keep"], "un": r["unassigned"], "ari": r.get("stab_ari"),
                     "fj": r.get("stab_flag_jacc"), "bal": r["balance"], "agree": r["agree"] if "agree" in r else np.nan,
                     "ncl": r["info"]["n_clusters"], "nbad": r["info"]["n_bad"]})
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
g = df.groupby(["name", "N"]).agg(n=("prec", "size"), prec=("prec", "mean"), prec_min=("prec", "min"),
                                  rec=("rec", "mean"), kfb=("kfb", "mean"), keep=("keep", "mean"), un=("un", "mean"),
                                  ari=("ari", "mean"), fj=("fj", "mean"), bal=("bal", "mean"))
print(g.round(3).to_string())
if len(sys.argv) > 1:
    print(df.pivot_table(index=["name", "N"], columns="setting", values="prec", aggfunc="mean").round(3).to_string())
