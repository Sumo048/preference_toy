"""Evaluate a graft fitted on the npm-4 cell on the npm-16 cell of the same seed and layout.

The fit uses only pi_ref, E1's own random streams and the tracker's bad tracks; the negatives enter
only through which tracks are bad. If both cells name the same bad tracks, the npm-16 fit is
identical to the npm-4 fit (confirmed bit-for-bit by the K10 sweep, which fitted both), so we
reuse it and record that here."""
import json, sys, torch
from dataclasses import asdict
from common import *
from graft import GCfg, Graft, combine_hooks
tag = sys.argv[1]
for seed in range(5):
    for layout in ("random", "alt"):
        c4 = load_cell(seed, layout, 4)
        src = f"results/graft_{tag}_{c4.name}.json"
        if not os.path.exists(src):
            continue
        c16 = load_cell(seed, layout, 16)
        dst = f"results/graft_{tag}_{c16.name}.json"
        if os.path.exists(dst):
            continue
        b4 = build_tracker(c4.ref, seed, c4.ed).bad
        T16 = build_tracker(c16.ref, seed, c16.ed)
        rec = json.load(open(src)); cfg = GCfg(**rec["cfg"])
        if T16.bad != b4:
            print(c16.name, "bad tracks differ, needs its own fit", b4, T16.bad); continue
        grs = []
        for sd in torch.load(f"cache/grafts_{tag}_{c4.name}.pt", weights_only=False):
            g = Graft(cfg, sd["mu_d"].shape[0], sd["mu_f"].shape[0], sd["det_w"], sd["det_b"], sd["mu_d"], sd["sd_d"],
                      sd["mu_f"], sd["sd_f"], sd["w_out"].shape[0]); g.load_state_dict(sd); grs.append(g)
        ev = oracle_eval(c16, Policy(c16.ref.net, combine_hooks(c16.ref.net, grs)))
        out = {**rec, "cell": c16.name, "npm": 16, "tracker": tracker_diag(T16, c16),
               "note": f"graft fitted on {c4.name}; same bad tracks {b4}, so the npm-16 fit is identical", **ev}
        json.dump(out, open(dst, "w"), indent=1)
        print(f"[{tag}] {c16.name} D {ev['D']:.3f} Rem {ev['Removed']:.3f} Col {ev['Collateral']:.3f} (n4: D {rec['D']:.3f})", flush=True)
