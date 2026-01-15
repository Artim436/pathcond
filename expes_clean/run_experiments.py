import mlflow
from train_script import train_loop
import numpy as np

def run_grid_search():
    # Define the search space
    architectures = [
        [8, 16, 8],
        [16, 32, 32, 16],
        [16, 32, 64, 32, 16],
        [32, 64, 128, 64, 32],
    ]
    learning_rates = list(np.logspace(-4, 0, num=5))  # 1e-4 to 1e0
    methods = ["baseline", "pathcond", "enorm"]
    seeds = list(range(10))  # Seeds 0 to 9
    
    data = "moons"
    epochs = 10000
    optimizer = "sgd"

    total_runs = len(architectures) * len(learning_rates) * len(methods) * len(seeds)
    current_run = 1

    for arch in architectures:
        for lr in learning_rates:
            for method in methods:
                for seed in seeds:
                    print(f"[{current_run}/{total_runs}] Training: {method} | Arch: {arch} | LR: {lr} | Seed: {seed}")
                    
                    try:
                        train_loop(
                            architecture=arch,
                            method=method,
                            learning_rate=lr,
                            epochs=epochs,
                            seed=seed,
                            data=data,
                            optimizer=optimizer
                        )
                    except Exception as e:
                        print(f"Run failed: {e}")
                    
                    current_run += 1

if __name__ == "__main__":
    run_grid_search()