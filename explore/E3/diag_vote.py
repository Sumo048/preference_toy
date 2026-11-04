"""Diagnostic (oracle used for scoring only): vote-mask vs track-mask quality on the anchor pool."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e3, tracks
from toypref.cond import experiment as E
torch.set_num_threads(1)
out = {}
for st, s, n in [("k4", 0, 4), ("k4", 0, 16), ("alt", 0, 4), ("alt", 0, 16)]:
    cell = e3.Cell(st, s, n); ctx = cell.ctx
    bank, keep, diag = E.get_vote(ctx)
    P = ctx.anchor; N = E.N_VOTE_POOL
    bad = np.isin(P.mode[:N], cell.ed.forget); flag = ~keep.numpy()
    c = P.c[:N].numpy()
    bins = np.digitize(c, [0.25, 0.5, 0.75])
    per = [{"prec": float(bad[flag & (bins == b)].mean()) if (flag & (bins == b)).any() else None,
            "rec": float(flag[bad & (bins == b)].mean())} for b in range(4)]
    tr = cell.tracks(); bt, _ = tracks.bad_tracks(tr, cell.ed.c, cell.ed.a)
    t = tr.assign(P.a[:N], P.c[:N]); tf = np.isin(t, bt)
    # retained-mode share of the kept anchor set (balance): oracle masses of kept samples
    kept_modes = np.bincount(P.mode[:N][keep.numpy() & (P.mode[:N] >= 0)], minlength=8) / max(1, keep.sum())
    rec = {"vote_prec": float(bad[flag].mean()), "vote_rec": float(flag[bad].mean()), "vote_flag_frac": float(flag.mean()),
           "vote_by_c_quarter": per, "track_prec": float(bad[tf].mean()), "track_rec": float(tf[bad].mean()),
           "vote_kept_mode_shares": np.round(kept_modes, 3).tolist(), "forget": cell.ed.forget}
    out[f"{st}_s{s}_n{n}"] = rec; print(st, s, n, json.dumps(rec), flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "diag_vote.json"), "w"), indent=1)
