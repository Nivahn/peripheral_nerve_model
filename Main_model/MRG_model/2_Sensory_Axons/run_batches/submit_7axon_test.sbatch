#!/bin/bash
#SBATCH -J 7a_test
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1
#SBATCH --time=06:00:00
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_test_%A_%a.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_test_%A_%a.err

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

EDGE=${EDGE:-0.1}
OUT_DIR=${OUT_DIR:-$WORK_DIR/data/prescott_7axon_ec_test}
FREQ=$((50 + 350 * SLURM_ARRAY_TASK_ID))
echo "=== EC test: edge=${EDGE} um  freq=${FREQ} Hz  task=${SLURM_ARRAY_TASK_ID} ==="

python -u run_batches/run_20axon_sweep.py \
    --fiber-diameter-um 4.5 \
    --edge-dist-um "$EDGE" \
    --n-axons 7 \
    --amp-na -3.0 \
    --freq-single "$FREQ" \
    --t-start-ms 10.0 \
    --t-end-ms 1000.0 \
    --dt-ms 0.05 \
    --branch-center-only \
    --skip-existing \
    --out-dir "$OUT_DIR"

echo "=== DONE ==="
