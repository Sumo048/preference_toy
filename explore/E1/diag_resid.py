"""Where does the residual forgotten mass of a fitted graft come from? (oracle used for diagnosis only)
Splits samples that pi_ref sends to forgotten modes and that the grafted policy still sends there into
'gate never fired' (detector miss) vs 'gate fired, redirect failed'; and attributes new off-mode mass."""
import sys, json, torch, numpy as np
from common import *
from graft import GCfg, Graft, combine_hooks
tag, seed, layout, npm = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
cell = load_cell(seed, layout, npm); net = cell.ref.net
rec = json.load(open(f"results/graft_{tag}_{cell.name}.json")); cfg = GCfg(**rec["cfg"])
sds = torch.load(f"cache/grafts_{tag}_{cell.name}.pt", weights_only=False)
grs = []
for sd in sds:
    g = Graft(cfg, sd["mu_d"].shape[0], sd["mu_f"].shape[0], sd["det_w"], sd["det_b"], sd["mu_d"], sd["sd_d"], sd["mu_f"], sd["sd_f"], sd["w_out"].shape[0])
    g.load_state_dict(sd); grs.append(g)
fired = torch.zeros(20000, dtype=torch.bool)
b = cell.bank; xi, c = b.xi, b.c
cf = net.c_feat(c)
# decode with grafts, recording whether any gate fired
x = xi.clone()
hooks = combine_hooks(net, grs)
with torch.no_grad():
    for k in range(K_STEPS):
        t = torch.full((x.shape[0],), 1.0 - k * DT)
        v, acts = net_forward(net, x, t, c, capture=True, hooks=hooks)
        for g in grs:
            fired |= g.gate(acts, x, t, cf)
        x = x - DT * v
lab1 = E.MIX.assign(x, c); lab0 = b.lab_ref; F_ = cell.ed.forget
bad0, bad1 = np.isin(lab0, F_), np.isin(lab1, F_)
f = fired.numpy()
out = {"ref_forget_mass": float(bad0.mean()), "still_forget": float((bad0 & bad1).mean()),
       "still_forget_gate_never_fired": float((bad0 & bad1 & ~f).mean()),
       "still_forget_gate_fired": float((bad0 & bad1 & f).mean()),
       "new_forget_from_retained": float((~bad0 & bad1).mean()),
       "off_ref": float((lab0 == -1).mean()), "off_new": float((lab1 == -1).mean()),
       "off_new_from_forget": float((bad0 & (lab1 == -1)).mean()),
       "off_new_from_retained_fired": float((~bad0 & (lab0 != -1) & (lab1 == -1) & f).mean()),
       "retained_fired": float(f[np.isin(lab0, cell.ed.retain)].mean()),
       "retained_flipped": float((np.isin(lab0, cell.ed.retain) & (lab1 != lab0)).mean())}
print(cell.name, tag, json.dumps({k: round(v, 4) for k, v in out.items()}))
json.dump(out, open(f"results/resid_{tag}_{cell.name}.json", "w"), indent=1)
