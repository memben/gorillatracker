import wandb

# Define your sweep configuration
sweep_config = {
    "program": "./train.py",  # Note: not the sweep file, but the training script
    "name": "test2",
    "method": "grid",  # Specify the search method (random search in this case)
    "metric": {"goal": "minimize", "name": "train/loss"},  # Specify the metric to optimize
    "parameters": {
        # "param1": {"min": 1, "max": 10},  # Define parameter search space
        "loss_mode": {"values": ["offline", "online/soft", "online/hard", "online/semi-hard"]},
        # Add other parameters as needed
    },
}
# Initialize the sweep
project_name = "test_losses"
entity = "gorillas"
sweep_id = wandb.sweep(sweep=sweep_config, project=project_name, entity=entity)
# Print the sweep ID directly
print(f"SWEEP_PATH={entity}/{project_name}/{sweep_id}")
