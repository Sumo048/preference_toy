# STATE E3

Branch: `claude/session-e3-7loi2v` (harness-designated; SESSION_PROMPTS says `art/explore-E3`, also pushed there if allowed).

Done:
- Setup (torch 2.14.0 from PyPI; pandas unpinned because 2.2.2 does not build on Python 3.13).
- `explore/E3/e3.py` reproduces B1a bit-for-bit on k4 s0 n4 (D 0.259).

Running: batch A (B0, B1a, oracle BC, out/b2/lora4 subsets, no/full anchor, 20k steps) on 12 cells, `explore/E3/runall.sh`.

Next: batch B (more subsets, anchors, weights), then best method on seeds 3-4, write `docs/explore/E3.md` and `docs/DONE_E3.md`.
