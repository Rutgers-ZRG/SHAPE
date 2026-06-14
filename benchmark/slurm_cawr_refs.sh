#!/bin/bash
# Reference relaxations: one array task per system. Run before the run array.
#SBATCH --job-name=cawr_ref
#SBATCH --partition=main
#SBATCH --array=0-3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/lz432/cawr_campaign/logs/ref_%A_%a.out
set -e
ulimit -l unlimited 2>/dev/null || true          # let unused MPI auto-init succeed
source ~/miniconda3/etc/profile.d/conda.sh
conda activate msim
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
BASE=/scratch/lz432/cawr_campaign
export PYTHONPATH=$BASE/ReformPy
SYSTEMS=(B28 MgO TiO2 Si)
sys=${SYSTEMS[$SLURM_ARRAY_TASK_ID]}
python $BASE/ReformPy/benchmark/cawr_csp_campaign.py reference \
    --system "$sys" --refs $BASE/refs
