#!/bin/bash
#SBATCH -J 7a_ampcal
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-5
#SBATCH --time=06:00:00
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_ampcal_%A_%a.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_ampcal_%A_%a.err

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

# Свип амплитуд (нА, отрицательные): 3,4,5,6,8,10
AMPS=(-3.0 -4.0 -5.0 -6.0 -8.0 -10.0)
AMP=${AMPS[$SLURM_ARRAY_TASK_ID]}
TAG=$(echo "$AMP" | tr -d '-' | tr '.' 'p')

echo "=== AMP CALIBRATION: amp=${AMP} nA, isolated center (--no-ephaptic), 100 Hz, 110 ms ==="

python -u run_batches/run_20axon_sweep.py \
    --fiber-diameter-um 4.5 \
    --edge-dist-um 0.1 \
    --n-axons 7 \
    --amp-na "$AMP" \
    --freq-single 100 \
    --t-start-ms 10.0 \
    --t-end-ms 110.0 \
    --dt-ms 0.05 \
    --branch-center-only \
    --no-ephaptic \
    --skip-existing \
    --out-dir "$WORK_DIR/data/prescott_7axon_amp_cal/$TAG"

echo "=== DONE amp=${AMP} ==="
