import mlflow
from train_script import train_loop
import numpy as np
import argparse
import socket

def run_grid_search(architectures=[[8, 8]], experiment_name="default_experiment", epochs=1000, seed_nb=3, nb_lr=0, selected_methods=None, data="moons", lrs=None):
    
    learning_rates = list(np.logspace(-4, 0, num=nb_lr)) if nb_lr > 1 else lrs
    all_methods = ["baseline", "pathcond", "enorm", "bn_baseline", "bn_pathcond", "bn_enorm", "pathcond_telep_schedule", 'bn_pathcond_telep_schedule']
    
    # Filtrage des méthodes pour cette machine
    methods = selected_methods if selected_methods else all_methods
    seeds = list(range(seed_nb))
    optimizer = "sgd"
    
    total_runs = len(architectures) * len(learning_rates) * len(methods) * len(seeds)
    current_run = 1

    # On récupère le nom de la machine pour le loguer
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
                            experiment_name=experiment_name
                        )
                    except Exception as e:
                        print(f"Run failed on {hostname}: {e}")
                    
                    current_run += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", nargs="+", help="Liste des méthodes à tester (ex: baseline enorm)")
    parser.add_argument("--exp", type=str, default="grid_search_shared", help="Nom de l'expérience MLflow")
    args = parser.parse_args()

    # Exemple de config d'architecture
      # Architectures avec 500 neurones et profondeur de 2 à 8 mais ne ppas prendre 5 
    my_archs = [[784//2, 784//4, 784//2], [784//4, 784//16, 784//4], [784//7, 784//49, 784//7], [784, 784, 784]]
    
    run_grid_search(
        architectures=my_archs, 
        experiment_name='Encoder_decoder',
        selected_methods=args.methods, 
        epochs=500,
        data="mnist_enc_dec",
        seed_nb=3,
        lrs=[0.001, 0.01, 0.1]
    )