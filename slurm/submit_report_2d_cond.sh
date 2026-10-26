#!/bin/bash
#SBATCH --job-name=r2d_cond
#SBATCH --output=logs/%A_%a_r2d_cond.log
#SBATCH --error=logs/%A_%a_r2d_cond.log
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH --partition=shubhamlong
# One task per arm (+ "common"); conditional cell seed 0, random layout, n_forget 4, n_neg 16.
set -eo pipefail
cd /home/soumojib/repos/toypref-report
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TOYPREF_RUNS=/grogu/datasets/rollout/soumojib/toypref/report_2d
ARMS=(common B0_frozen NPO_b100_neg KTO_b10_neg R_hop_bsup_neg R_hopB_bsup_neg_AMdist SDPO_fresh_b25 SDPO_pin_b25 SDPO_hop_pin_b25)
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/report/twod_cond_run.py --arm ${ARMS[$SLURM_ARRAY_TASK_ID]}
