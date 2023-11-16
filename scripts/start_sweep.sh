#!/bin/bash

# Set Mamba to use the correct environment
# eval "$(micromamba shell hook --shell=)"
# micromamba activate researcha # replace with your environment name

# Set your WandB API Key
# export WANDB_API_KEY="your-wandb-api-key" # should be in .netrc

# Capture the Sweep ID from the printed output
export SWEEP_PATH=$(python ./init_sweep.py | grep SWEEP_PATH | cut -d'=' -f2)

# Run the sweep agent with the training script and hyperparameters
wandb agent --count 1 $SWEEP_PATH
