"""Check that e5's chunked, resumable fit equals E1's graft.fit_graft (unchunked) bit for bit.

Short config (K10c with iters 60) on a dev cell (k4 seed 0 npm 4), chunks of 20 iterations with a
checkpoint/exit after every chunk, vs E1's own fit loop in one process.
Usage: OMP_NUM_THREADS=1 python explore/E5/test_chunk.py <scratch dir>
"""
import os, sys, tempfile
from dataclasses import replace
work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
os.makedirs(work, exist_ok=True)
os.environ["E5_CHUNK_S"] = "0"; os.environ["E5_STEP"] = "20"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e5, torch, graft, common
torch.set_num_threads(1)
cfg = replace(e5.K10C, iters=60)
e5.OUT, e5.CACHE, e5.GRAFTS = (os.path.join(work, d) for d in ("results", "cache", "grafts"))
n = 0
while not e5.unit_fit("k4", 0, cfg=cfg, tag="test"):
    n += 1
print("chunked: units", n + 1)
chunked = torch.load(os.path.join(e5.GRAFTS, "grafts_test_k4_s0_n4.pt"), weights_only=False)

# reference: E1's fit_graft, unchanged (run from a separate dir so its relative cache/ is fresh)
os.makedirs(os.path.join(work, "ref", "cache"), exist_ok=True); os.chdir(os.path.join(work, "ref"))
cell = e5.load_cell("k4", 0, 4)
T = common.build_tracker(cell.ref, 0, cell.ed)
grs = []
for r in range(cfg.rounds):
    gr, hooks, info = graft.fit_graft(cell, T, cfg, prev=grs)
    grs.append(gr)
ref = [g.state_dict() for g in grs]
ok = all(torch.equal(a[k], b[k]) for a, b in zip(chunked, ref) for k in a) and len(chunked) == len(ref)
print("bit-identical:", ok)
sys.exit(0 if ok else 1)
