#!/bin/bash
#SBATCH -J 7a_analyze
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --time=04:00:00
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_analyze_%j.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_analyze_%j.err

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

# ROOT можно переопределить при сабмите (например, для валидации на другом наборе):
#   sbatch --export=ALL,ROOT=data/prescott_20axon_sweep submit_7axon_analyze.sbatch
ROOT=${ROOT:-data/prescott_7axon_sweep}
OUT_DIR=${OUT_DIR:-$ROOT/analysis}

echo "=== 7-axon analysis: ROOT=${ROOT} OUT=${OUT_DIR} ==="

python -u run_batches/analyze_7axon_sweep.py \
    --root "$ROOT" \
    --out-dir "$OUT_DIR"

echo "=== DONE ==="
