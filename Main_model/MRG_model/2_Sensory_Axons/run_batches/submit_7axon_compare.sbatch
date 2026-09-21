#!/bin/bash
#SBATCH -J 7a_compare
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_compare_%j.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_compare_%j.err

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

EC_SUM=${EC_SUM:-data/prescott_7axon_sweep/analysis/summary.csv}
NOEC_SUM=${NOEC_SUM:-data/prescott_7axon_noec_sweep/analysis/summary.csv}
OUT_DIR=${OUT_DIR:-data/prescott_7axon_sweep/analysis/compare}

echo "=== EC vs no-EC compare ==="
echo "  EC   = $EC_SUM"
echo "  noEC = $NOEC_SUM"
echo "  out  = $OUT_DIR"

python -u run_batches/compare_ec_noec.py \
    --ec "$EC_SUM" \
    --noec "$NOEC_SUM" \
    --out-dir "$OUT_DIR"

echo "=== DONE ==="
