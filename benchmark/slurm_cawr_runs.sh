#!/bin/bash
# 3-arm relaxations: one array task per structure of ONE system.
# Submit once per system (MaxArraySize=1001 forbids a single 1200 array):
#   for s in B28 MgO TiO2 Si; do
#     sbatch --export=ALL,SYS=$s --job-name=cawr_$s --array=0-299%25 slurm_cawr_runs.sh
#   done
#SBATCH --partition=main
#SBATCH --array=0-299%25
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/lz432/cawr_campaign/logs/run_%x_%A_%a.out
set -e
ulimit -l unlimited 2>/dev/null || true          # let unused MPI auto-init succeed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate msim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE=/scratch/lz432/cawr_campaign
export PYTHONPATH=$BASE/ReformPy
sys=${SYS:?must pass the system via --export=ALL,SYS=<name>}
python $BASE/ReformPy/benchmark/cawr_csp_campaign.py run \
    --system "$sys" --idx "$SLURM_ARRAY_TASK_ID" \
    --pool $BASE/pools --refs $BASE/refs --out $BASE/results
