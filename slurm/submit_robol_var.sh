#!/bin/bash
#SBATCH --job-name=robol_var
#SBATCH --output=logs/%A_%a_robol_var.log
#SBATCH --error=logs/%A_%a_robol_var.log
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=12:00:00
#SBATCH --partition=all,shubhamlong
#SBATCH --exclude=grogu-0-25,grogu-1-20
# MANIFEST rows: mode,seed,labels,ns,arm,ks (ks ';'-separated). Never the shubhamlong A6000 nodes.
set -eo pipefail
cd /home/soumojib/repos/toypref-robo
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MKL_DYNAMIC=FALSE MKL_CBWR=AVX2 ATEN_CPU_CAPABILITY=avx2 PYTHONUNBUFFERED=1
PY=$HOME/miniconda3/envs/pref_pillars/bin/python
ROW=$($PY -c "import csv;r=list(csv.DictReader(open('$MANIFEST')))[$SLURM_ARRAY_TASK_ID];print(r['mode'],r['seed'],r['labels'],r['ns'],r['arm'],r['ks'].replace(';',','),(r.get('shifts') or '2;4;8').replace(';',','))")
read MO S L N A K SH <<< "$ROW"
$PY scripts/robol_sdpo_variants.py --mode $MO --seed $S --labels $L --ns $N --arm $A --ks $K --shifts $SH
echo "=== ROBOL VAR TASK $SLURM_ARRAY_TASK_ID DONE ==="
