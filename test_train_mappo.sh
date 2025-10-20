#!/bin/bash
# Test script for train_mappo_microgrids.py
# This script runs a quick training test with minimal iterations

set -e  # Exit on error

echo "=========================================="
echo "Testing MAPPO Training Script"
echo "=========================================="

# Activate virtual environment if needed
if [ ! -z "$VIRTUAL_ENV" ]; then
    echo "Using virtual environment: $VIRTUAL_ENV"
else
    echo "No virtual environment detected"
fi

# Test 1: Basic MAPPO training (shared policy)
echo ""
echo "Test 1: MAPPO with shared policy (5 iterations)"
echo "------------------------------------------"
python examples/train_mappo_microgrids.py \
    --iterations 5 \
    --num-workers 2 \
    --train-batch-size 1000 \
    --checkpoint-freq 5 \
    --experiment-name test_mappo_shared \
    --no-cuda

# Test 2: IPPO training (independent policies)
echo ""
echo "Test 2: IPPO with independent policies (5 iterations)"
echo "------------------------------------------"
python examples/train_mappo_microgrids.py \
    --iterations 5 \
    --independent-policies \
    --num-workers 2 \
    --train-batch-size 1000 \
    --checkpoint-freq 5 \
    --experiment-name test_ippo_independent \
    --no-cuda

# Test 3: Different environment configurations
echo ""
echo "Test 3: MAPPO with no shared reward (5 iterations)"
echo "------------------------------------------"
python examples/train_mappo_microgrids.py \
    --iterations 5 \
    --no-share-reward \
    --penalty 20 \
    --num-workers 2 \
    --train-batch-size 1000 \
    --checkpoint-freq 5 \
    --experiment-name test_mappo_no_shared_reward \
    --no-cuda

# Test 4: Custom hyperparameters
echo ""
echo "Test 4: MAPPO with custom hyperparameters (3 iterations)"
echo "------------------------------------------"
python examples/train_mappo_microgrids.py \
    --iterations 3 \
    --lr 1e-4 \
    --gamma 0.95 \
    --lambda 0.9 \
    --hidden-dim 128 \
    --num-workers 2 \
    --train-batch-size 1000 \
    --sgd-minibatch-size 64 \
    --num-sgd-iter 5 \
    --checkpoint-freq 3 \
    --experiment-name test_mappo_custom_params \
    --no-cuda

echo ""
echo "=========================================="
echo "All tests completed successfully!"
echo "=========================================="
echo ""
echo "Checkpoints saved in:"
echo "  - ./checkpoints/test_mappo_shared/"
echo "  - ./checkpoints/test_ippo_independent/"
echo "  - ./checkpoints/test_mappo_no_shared_reward/"
echo "  - ./checkpoints/test_mappo_custom_params/"
