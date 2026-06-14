#!/bin/bash
# 3-arm relaxations: one array task per (system, structure). 4 systems x 300.
# Array index -> system = idx//300, structure = idx%300.
#SBATCH --job-name=cawr_csp
#SBATCH --partition=main
#SBATCH --array=0-1199%60
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/lz432/cawr_campaign/logs/run_%A_%a.out
set -e
ulimit -l unlimited 2>/dev/null || true          # let unused MPI auto-init succeed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate msim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE=/scratch/lz432/cawr_campaign
export PYTHONPATH=$BASE/ReformPy
SYSTEMS=(B28 MgO TiO2 Si)
N=300
sys=${SYSTEMS[$(( SLURM_ARRAY_TASK_ID / N ))]}
idx=$(( SLURM_ARRAY_TASK_ID % N ))
python $BASE/ReformPy/benchmark/cawr_csp_campaign.py run \
    --system "$sys" --idx "$idx" \
    --pool $BASE/pools --refs $BASE/refs --out $BASE/results
