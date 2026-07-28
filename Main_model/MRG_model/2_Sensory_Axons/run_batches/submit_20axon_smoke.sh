#!/bin/bash
# Quick smoke test: 20 axons, 1 frequency, 60 ms.
# Run from the 2_Sensory_Axons directory on the cluster.
#
# Usage:
#   bash run_batches/submit_20axon_smoke.sh

set -e

WORK_DIR="/home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons"
cd "$WORK_DIR" || exit 1

mkdir -p "$WORK_DIR/run_batches/logs"

echo "Submitting smoke test: 20 axons, 50 Hz, 60 ms"

FIBER_DIAMETER_UM=4.5 \
EDGE_DIST_UM=0.1 \
N_AXONS=20 \
AMP_NA=-3.0 \
FREQ_START=50 \
FREQ_END=101 \
FREQ_STEP=50 \
T_START_MS=10.0 \
T_END_MS=60.0 \
DT_MS=0.01 \
OUT_DIR="$WORK_DIR/data/prescott_20axon_smoke" \
sbatch --job-name="20axon_smoke" \
       --output="$WORK_DIR/run_batches/logs/20axon_smoke_%j.out" \
       --error="$WORK_DIR/run_batches/logs/20axon_smoke_%j.err" \
       run_batches/submit_20axon.sbatch

echo "Check with: squeue -u \$USER"
