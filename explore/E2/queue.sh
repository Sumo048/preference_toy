#!/bin/bash
# usage: queue.sh <P> <file with lines "seed layout npm method"> ; runs run_edit.py in parallel
cd "$(dirname "$0")/../.."
export TOYPREF_RUNS=$PWD/runs OMP_NUM_THREADS=1
mkdir -p explore/E2/logs
P=$1; F=$2
cat "$F" | xargs -P "$P" -L 1 sh -c 'python explore/E2/run_edit.py --seed $0 --layout $1 --npm $2 --method $3 >> explore/E2/logs/edits_$1_s$0_n$2.log 2>&1; echo "$0 $1 $2 $3 rc=$?" >> explore/E2/logs/queue_done.log'
