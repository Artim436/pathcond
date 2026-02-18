import mlflow
from train_script import train_loop
import numpy as np
import argparse
import socket

def run_grid_search(architectures=[[8, 8]], experiment_name="default_experiment", epochs=1000, seed_nb=3, nb_lr=0, selected_methods=None, data="moons", lrs=None, init_kai_normal=False):
    
    learning_rates = list(np.logspace(-4, 0, num=nb_lr)) if nb_lr > 1 else lrs
    all_methods = ["baseline", "pathcond", "enorm", "bn_baseline", "bn_pathcond", "bn_enorm", "pathcond_telep_schedule", 'bn_pathcond_telep_schedule']
    methods = selected_methods if selected_methods else all_methods
    seeds = list(range(seed_nb))
    optimizer = "sgd"
    
    total_runs = len(architectures) * len(learning_rates) * len(methods) * len(seeds)
    current_run = 1

    hostname = socket.gethostname()

    for seed in seeds:
        for lr in learning_rates:
            for method in methods:
                for arch in architectures:
                    print(f"[{hostname}] [{current_run}/{total_runs}] Training: {method} | LR: {lr}")
                    try:
                        train_loop(
                            architecture=arch,
                            method=method,
                            learning_rate=lr,
                            epochs=epochs,
                            seed=seed,
                            data=data,
                            optimizer=optimizer,
                            experiment_name=experiment_name,
                            init_kai_normal=init_kai_normal
                        )
                    except Exception as e:
                        print(f"Run failed on {hostname}: {e}")
                    
                    current_run += 1

if __name__ == "__main__":
    my_archs = [[500]*k for k in range(3, 9)]
    
    run_grid_search(
        architectures=my_archs, 
        experiment_name='FC_varying_depth',
        selected_methods=["bn_baseline", "bn_pathcond", "bn_enorm"],
        epochs=100,
        data="cifar10",
        seed_nb=3,
        lrs=[0.001],
        init_kai_normal=False
    )