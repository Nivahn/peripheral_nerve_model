#!/bin/bash
#SBATCH -J on_5p7_m05
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches_10jobs_no_branches/logs/on_5p7_m05_%j.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches_10jobs_no_branches/logs/on_5p7_m05_%j.err
set -euo pipefail
SCENARIO="no_branch_reference"
STIM_PROTOCOL="sync"   # sync or delay
STIM_B_DELAY_MS="0.5"  # used only when STIM_PROTOCOL="delay"
cd /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/final_run_10jobs_no_branches
mkdir -p /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches_10jobs_no_branches/logs
module load anaconda/anaconda3/4.7.5
export PATH="/home/dubrovin/.conda/envs/dorsal_column_env/bin:$PATH"
export PS1="${PS1:-}"
source activate dorsal_column_env
export MPLBACKEND=Agg
python final_results_worker.py one_node_branching 5.7 "$SCENARIO" misaligned_0.5 "$STIM_PROTOCOL" "$STIM_B_DELAY_MS"
