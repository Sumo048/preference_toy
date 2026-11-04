import sys, numpy as np, torch
torch.set_num_threads(2)
sys.path.insert(0, "scripts")
from run_arm import cached_cell_for
from toyrepair import tracks, gate_lf, pools, policy, oracle, miner
for (s, seed, n) in [("k4", 0, 4), ("alt", 0, 16)]:
    cell = cached_cell_for("cond", s, seed, n)
    tr = tracks.get(cell)
    base = policy.base_version(cell)
    at = miner.mine(cell, base)
    cl = at.clusters[0]
    bk = gate_lf.buckets_for(tr, cell, list(map(int, cl.uids)))
    c, xi = pools.draw(cell, pools.GATE, 16384, index=0)
    a = policy.decode(base, xi, c).action
    b = bk.of(tr.assign(a, c))
    lab = np.asarray(oracle.label(cell, a, c)); lab = np.where(lab == oracle.OFF, 8, lab)
    F = oracle.forget_modes(cell); T = oracle.target_modes(cell, list(map(int, cl.uids)))
    print(s, seed, n, "tracks", tr.n_tracks, "buckets T,O,G", bk.target, bk.other, bk.good, "F", F, "T", T)
    names = ["T", "O"] + [f"G{g}" for g in bk.good] + ["U"]
    tab = np.zeros((bk.nb, 9), int)
    np.add.at(tab, (b, lab), 1)
    print("rows=lf bucket, cols=oracle mode 0-7,OFF")
    for i, r in enumerate(tab):
        print(f"{names[i]:>4}", " ".join(f"{x:5d}" for x in r))
