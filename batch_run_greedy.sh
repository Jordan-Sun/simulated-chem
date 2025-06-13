#!/bin/zsh
# batch_run_greedy.sh
MAX_MEM_KB=$((16 * 1024 * 1024))  # 16 GB
ulimit -v $MAX_MEM_KB

# Configuration section: set your values here
NUM_INTERVALS=504
BATCH_SIZE=12
RES=48
HOSTS=1
PTILE=576
SWAP_ALG_NAME="greedy"
WORKLOAD_BASE="test/workloads/"
ORIGINAL_ASSIGNMENT_BASE="test/og_assignments/"
MOD=""  # Set to empty string if not using a mod
GROUP=0  # Set a seed or file for reproducibility

MAX_PARALLEL=$(($(nproc) / 2))

# Export variables for GNU parallel
export BATCH_SIZE RES HOSTS PTILE SWAP_ALG_NAME WORKLOAD_BASE ORIGINAL_ASSIGNMENT_BASE GROUP MOD

# Generate start indices for each batch and run with batch size
seq 0 $BATCH_SIZE $((NUM_INTERVALS-1)) | parallel -j $MAX_PARALLEL --eta \
  'python3 run_greedy_interval.py {} $BATCH_SIZE $RES $HOSTS $PTILE $SWAP_ALG_NAME $WORKLOAD_BASE $ORIGINAL_ASSIGNMENT_BASE $GROUP $MOD'

# Combine all assignments
python3 combine_assignments.py $NUM_INTERVALS $RES $HOSTS $PTILE $SWAP_ALG_NAME $GROUP $MOD
