import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_experiment_data(experiment_name, target_acc=0.90):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    
    if not experiment:
        print(f"Expérience '{experiment_name}' non trouvée.")
        return pd.DataFrame()

    runs = client.search_runs(experiment.experiment_id)
    data_list = []

    for run in runs:
        params = run.data.params
        metrics_history = client.get_metric_history(run.info.run_id, "test_acc")
        
        # Récupération du LR (on gère les deux clés possibles)
        lr = params.get("learning_rate") or params.get("lr")
        if lr is None: continue
        
        # Convergence
        epoch_reached = next((m.step for m in metrics_history if m.value >= target_acc), np.nan)
        
        # Nombre de paramètres
        n_params = params.get("n_params")


        data_list.append({
            "method": params.get("method", "unknown"),
            "seed": int(params.get("seed", 0)),
            "lr": float(lr),
            "n_params": n_params,
            "epoch_to_target": epoch_reached
        })

    return pd.DataFrame(data_list)

def plot_efficiency_per_lr(df, target_name="40% Accuracy"):
    if df.empty: return

    # Style ICML
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3
    })

    # On itère sur chaque learning rate unique
    unique_lrs = sorted(df["lr"].unique())
    
    for lr_val in unique_lrs:
        df_lr = df[df["lr"] == lr_val]
        
        # Calcul stats pour les barres d'erreur
        df_grouped = df_lr.groupby(["method", "n_params"])["epoch_to_target"].agg(["mean", "std"]).reset_index()

        fig, ax = plt.subplots(figsize=(6, 4))
        
        methods = df_grouped["method"].unique()
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        markers = ['o', 's', '^', 'D', 'x']

        for i, method in enumerate(methods):
            subset = df_grouped[df_grouped["method"] == method].sort_values("n_params")
            
            # Utilisation de plt.errorbar avec fmt pour faire un scatter propre avec barres d'erreur
            ax.errorbar(
                subset["n_params"], 
                subset["mean"], 
                yerr=subset["std"],
                fmt=markers[i % len(markers)],
                color=colors[i % len(colors)],
                label=method.replace('_', ' ').title(),
                capsize=3,
                markersize=8,
                alpha=0.8,
                markeredgecolor='black', # Meilleur contraste pour ICML
                markeredgewidth=0.5
            )

        ax.set_title(f"Efficiency at Learning Rate: {lr_val}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Number of Parameters ($N$)", fontweight='bold')
        ax.set_ylabel(f"Epochs to {target_name}", fontweight='bold')
        
        ax.legend(loc='best', frameon=True, edgecolor='lightgrey')
        
        plt.tight_layout()
        filename = f"efficiency_lr_{lr_val}.pdf".replace('.', '_', 1) # évite double point
        plt.savefig(filename, bbox_inches='tight')
        plt.close() # Important pour libérer la mémoire
        print(f"Figure générée : {filename}")

if __name__ == "__main__":
    EXP_NAME = "PathCond_moons_experiments" 
    target = 0.90
    df = get_experiment_data(EXP_NAME, target_acc=target)
    
    if not df.empty:
        plot_efficiency_per_lr(df, target_name=f"{int(target*100)}% Acc")