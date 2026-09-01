#!/bin/bash
#SBATCH -J 20a_ed0p1
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/20a_ed0p1_%j.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/20a_ed0p1_%j.err

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

echo "=== 20-axon full run: edge_dist=0.1 um ==="

python run_batches/run_20axon_sweep.py \
    --fiber-diameter-um 4.5 \
    --edge-dist-um 0.1 \
    --n-axons 20 \
    --amp-na -3.0 \
    --freq-start 50 \
    --freq-end 1001 \
    --freq-step 50 \
    --t-start-ms 10.0 \
    --t-end-ms 1000.0 \
    --dt-ms 0.05 \
    --out-dir /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/data/prescott_20axon_sweep

echo "=== DONE ==="
