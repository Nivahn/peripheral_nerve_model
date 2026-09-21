#!/bin/bash
#SBATCH -J 7a5_ed0p5
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19
#SBATCH --time=11-00:00:00
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_ed0p5_%A_%a.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_ed0p5_%A_%a.err

export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load anaconda/anaconda3/4.7.5
export PATH="/home/dubrovin/.conda/envs/dorsal_column_env/bin:$PATH"
export PS1="${PS1:-}"
source activate dorsal_column_env

WORK_DIR="/home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons"
cd "$WORK_DIR" || exit 1

# 7 axons (1 center + 6 around), only the central one branches.
# SLURM array task index -> frequency: 50, 100, ..., 1000 Hz
FREQ=$((50 + 50 * SLURM_ARRAY_TASK_ID))
echo "=== 7-axon edge=0.5 um  freq=${FREQ} Hz  (task ${SLURM_ARRAY_TASK_ID}) ==="

python -u run_batches/run_20axon_sweep.py \
    --fiber-diameter-um 4.5 \
    --edge-dist-um 0.5 \
    --n-axons 7 \
    --amp-na -5.0 \
    --freq-single "$FREQ" \
    --t-start-ms 10.0 \
    --t-end-ms 1000.0 \
    --dt-ms 0.05 \
    --branch-center-only \
    --skip-existing \
    --out-dir /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/data/prescott_7axon_amp5_sweep

echo "=== DONE task ${SLURM_ARRAY_TASK_ID} ==="
