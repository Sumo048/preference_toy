# STATE E3

Branch: `claude/session-e3-7loi2v` (harness-designated), mirrored to `art/explore-E3`.

Done:
- Setup (torch 2.14.0 from PyPI; pandas unpinned: 2.2.2 does not build on Python 3.13).
- `explore/E3/e3.py` reproduces B1a bit-for-bit on k4 s0 n4 (D 0.259).
- Batch A (12 cells): param subsets / steps / anchor on-off barely move D (all within +0.01 of B1a); oracle BC floor D 0.037 (5k) / 0.025 (20k).
- Found: B1a in alt destroys retained modes; vote mask is noise-only, ring rotates with c.
- `explore/E3/tracks.py`: label-free behaviour tracks across c; 8 pure tracks, flag precision 1.0 on 3 test cells.

Running: batch C (track-masked BC, track anchor + reroute, oracle-mask decompositions, PSEUDO_vote_dist).

Next: batch B leftovers (subsets b0/b1/fc2/cond/lora16, anchors map/local/dist20k, weights, lr, mined) if time; best method on seeds 3-4; write docs/explore/E3.md and docs/DONE_E3.md.
