# toypref

Every robopref preference-learning arm (rerouting, its ablations and versions, and the
baselines), run on a 2-D, 8-mode unconditional flow policy that must forget some modes
from a handful of logged samples. Sweep: forgotten modes x negatives per mode x 5 seeds.

See `docs/DESIGN.md` for the setup, the arm list, the fairness/seeding rules and every
metric; `docs/RESULTS.md` for findings.

```
src/toypref/   mixture, flow policy, edit sets, rerouting index, objectives, arms, metrics
scripts/       train_base, run_cell, build_manifest/run_task, aggregate, plot
slurm/         sbatch scripts (CPU only)
tests/         fairness / RNG-plumbing tests
```
Environment: `~/miniconda3/envs/pref_pillars` (torch CPU, numpy, pandas, matplotlib).
