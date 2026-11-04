#!/bin/bash
# Ablations on k4 seed 0 npm 4 (after the c_feat cache fix).
cd "$(dirname "$0")"
run() { tag=$1; shift; OMP_NUM_THREADS=1 python graft.py --seed 0 --layout random --npm 4 --iters 1500 --tag $tag "$@" > logs/ab_$tag.log 2>&1; }
export -f run
cat <<'L' | xargs -P ${P:-4} -L 1 bash -c 'run "$@"' _
ab_v1
ab_bal --balance 1
ab_A --balance 1 --n_units 128 --k_gate 6
ab_K4 --balance 1 --n_units 128 --k_gate 4 --det_steps 4 --per_track 1
ab_b2 --balance 1 --n_units 128 --k_gate 8 --det_steps 6 --per_track 1 --out_layer b2
ab_U32 --balance 1 --n_units 32 --k_gate 10 --det_steps 8 --per_track 1
L
