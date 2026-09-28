#!/bin/bash
#SBATCH -J 7a_plots
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH -o /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_plots_%j.out
#SBATCH -e /home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons/run_batches/logs/7a_plots_%j.err

export MPLBACKEND=Agg
module load anaconda/anaconda3/4.7.5
export PATH="/home/dubrovin/.conda/envs/dorsal_column_env/bin:$PATH"
export PS1="${PS1:-}"
source activate dorsal_column_env

WORK_DIR="/home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons"
cd "$WORK_DIR" || exit 1

EC=${EC:-data/prescott_7axon_amp5_allstim_sweep/analysis/summary.csv}
NOEC=${NOEC:-data/prescott_7axon_amp5_allstim_noec_sweep/analysis/summary.csv}
MIS=${MIS:-data/prescott_7axon_amp5_allstim_mis_sweep/analysis/summary.csv}
OUT_DIR=${OUT_DIR:-data/prescott_7axon_amp5_allstim_sweep/analysis/diploma}

echo "=== diploma plots ==="
echo "  EC=$EC"
echo "  NOEC=$NOEC"
echo "  MIS=$MIS"

MIS_ARG=""
if [ -f "$MIS" ]; then MIS_ARG="--mis $MIS"; fi

python -u run_batches/plot_diploma_multifiber.py \
    --ec "$EC" \
    --noec "$NOEC" \
    $MIS_ARG \
    --out-dir "$OUT_DIR"

echo "=== DONE ==="
