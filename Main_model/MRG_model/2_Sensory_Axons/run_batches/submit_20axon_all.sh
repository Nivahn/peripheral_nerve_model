#!/bin/bash
# Submit multiple 20-axon jobs for different edge distances.
# Run from the 2_Sensory_Axons directory on the cluster.
#
# Usage:
#   bash run_batches/submit_20axon_all.sh
#
# Before running, adjust OUT_DIR and other variables below as needed.

set -e

WORK_DIR="/home/sagalajev_lab/mathematical_models/dorsal_column/2_Sensory_Axons"
OUT_BASE="$WORK_DIR/data/prescott_20axon_sweep"

# Fiber diameters and corresponding amplitudes (nA)
declare -A DIAM_AMP
DIAM_AMP[2.5]=-1.0
DIAM_AMP[4.5]=-3.0
DIAM_AMP[5.7]=-5.0

# Edge distances to sweep
EDGE_DISTANCES="0.1 0.5 1.0"

# Frequencies: 50-1000 Hz, step 50
FREQ_START=50
FREQ_END=1001
FREQ_STEP=50

# Simulation time
T_START_MS=10.0
T_END_MS=60.0
DT_MS=0.01

N_AXONS=20

mkdir -p "$WORK_DIR/run_batches/logs"

for DIAM in 4.5; do
    AMP=${DIAM_AMP[$DIAM]}
    for ED in $EDGE_DISTANCES; do
        DiamTag=$(echo $DIAM | tr '.' 'p')
        EdTag=$(echo $ED | tr '.' 'p')
        JobName="20a_fd${DiamTag}_ed${EdTag}"

        echo "Submitting: $JobName  (diam=$DIAM, edge=$ED, amp=$AMP)"

        FIBER_DIAMETER_UM=$DIAM \
        EDGE_DIST_UM=$ED \
        N_AXONS=$N_AXONS \
        AMP_NA=$AMP \
        FREQ_START=$FREQ_START \
        FREQ_END=$FREQ_END \
        FREQ_STEP=$FREQ_STEP \
        T_START_MS=$T_START_MS \
        T_END_MS=$T_END_MS \
        DT_MS=$DT_MS \
        OUT_DIR="$OUT_BASE" \
        sbatch --job-name="$JobName" \
               --output="$WORK_DIR/run_batches/logs/${JobName}_%j.out" \
               --error="$WORK_DIR/run_batches/logs/${JobName}_%j.err" \
               run_batches/submit_20axon.sbatch
    done
done

echo ""
echo "All jobs submitted. Check with: squeue -u \$USER"
