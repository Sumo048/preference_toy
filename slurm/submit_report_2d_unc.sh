#!/bin/bash
#SBATCH --job-name=r2d_unc
#SBATCH --output=logs/%A_%a_r2d_unc.log
#SBATCH --error=logs/%A_%a_r2d_unc.log
#SBATCH --array=0-10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --time=03:00:00
#SBATCH --partition=shubhamlong
# One task per arm (+ "common"), unconditional cell seed 0, n_forget 2, n_neg 8.
set -eo pipefail
cd /home/soumojib/repos/toypref-report
export OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
export TOYPREF_RUNS=/grogu/datasets/rollout/soumojib/toypref/report_2d
ARMS=(common B0_frozen B1_filtered B3_npo_b100_negonly B5_kto_b100_negonly R_ours_neg R_near_nf_nhood_neg SDPO_fresh_b1 SDPO_fresh_b25 SDPO_pin_b1 SDPO_pin_b25)
$HOME/miniconda3/envs/pref_pillars/bin/python scripts/report/twod_unc_run.py --arm ${ARMS[$SLURM_ARRAY_TASK_ID]}
