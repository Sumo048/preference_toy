# Phase 4 final patch stacks (W2b)

One directory per run (A3, B4, D1, R0; cond k4, npm 4/16, seeds 0-4): `stack.json` (cell, final version id, patch
ids in order, state sha256, final D) and `patches/<id>.json` + `.npz` (no caches). An empty stack is pi_ref (v0).
Exported and rebuild-checked by `scripts/export_patches.py`.

Rebuild a version from pi_ref (each patch's sha256 is checked; the result equals the live version bit for bit):

    python -c "import json,toyrepair; from toyrepair.cell import load_cell; from toyrepair.patches import rebuild; d='results/phase4/patches/p4_cond_k4_s0_n4_A3'; s=json.load(open(d+'/stack.json')); v=rebuild(load_cell(*s['cell']), d, s['patches'], s['version']); print(v.id)"

R0 accepted no edit in any run; its versions would be sampling-time controllers (code), not weight patches.
