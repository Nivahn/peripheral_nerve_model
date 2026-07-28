#!/bin/bash
#SBATCH -J prescott_20axon
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/20axon_%j.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/20axon_%j.err

# ---- Environment ----
export MPLBACKEND=Agg
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load anaconda/anaconda3/4.7.5
export PATH="/home/dubrovin/.conda/envs/dorsal_column_env/bin:$PATH"
export PS1="${PS1:-}"
source activate dorsal_column_env

# ---- Parameters (override via env vars before sbatch) ----
FIBER_DIAMETER_UM=${FIBER_DIAMETER_UM:-4.5}
EDGE_DIST_UM=${EDGE_DIST_UM:-0.1}
N_AXONS=${N_AXONS:-20}
AMP_NA=${AMP_NA:--3.0}
FREQ_START=${FREQ_START:-50}
FREQ_END=${FREQ_END:-101}
FREQ_STEP=${FREQ_STEP:-50}
T_START_MS=${T_START_MS:-10.0}
T_END_MS=${T_END_MS:-60.0}
DT_MS=${DT_MS:-0.01}
STIMULATE_ALL=${STIMULATE_ALL:-""}
OUT_DIR=${OUT_DIR:-/home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/data/prescott_20axon_sweep}

# ---- Run ----
WORK_DIR="/home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons"
cd "$WORK_DIR" || exit 1

EXTRA_ARGS=""
if [ -n "$STIMULATE_ALL" ]; then
    EXTRA_ARGS="--stimulate-all"
fi

echo "=== SLURM JOB $SLURM_JOB_ID ==="
echo "  FIBER_DIAMETER_UM=$FIBER_DIAMETER_UM"
echo "  EDGE_DIST_UM=$EDGE_DIST_UM"
echo "  N_AXONS=$N_AXONS"
echo "  AMP_NA=$AMP_NA"
echo "  FREQ=$FREQ_START..$FREQ_END step $FREQ_STEP"
echo "  DT_MS=$DT_MS"
echo "  T_END_MS=$T_END_MS"
echo "  OUT_DIR=$OUT_DIR"
echo "================================="

python run_batches/run_20axon_sweep.py \
    --fiber-diameter-um "$FIBER_DIAMETER_UM" \
    --edge-dist-um "$EDGE_DIST_UM" \
    --n-axons "$N_AXONS" \
    --amp-na "$AMP_NA" \
    --freq-start "$FREQ_START" \
    --freq-end "$FREQ_END" \
    --freq-step "$FREQ_STEP" \
    --t-start-ms "$T_START_MS" \
    --t-end-ms "$T_END_MS" \
    --dt-ms "$DT_MS" \
    --out-dir "$OUT_DIR" \
    $EXTRA_ARGS

echo "=== JOB $SLURM_JOB_ID DONE ==="
