import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathcond.utils import _ensure_outdir
from concurrent.futures import ThreadPoolExecutor
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import ast
import os
import seaborn as sns


def fetch_run_metrics(run, metrics_list, client):
    """Fonction helper optimisée pour récupérer les métriques d'un seul run."""
    params = run.data.params
    
    lr = params.get("learning_rate") or params.get("lr")
    if lr is None:
        return []
    
    method = params.get("method", "unknown")
    seed = int(params.get("seed", 0))
    arch = params.get("architecture", None)
    n_params = int(params.get("n_params", -1))
    max_rescale = float(params.get("max_rescaling_on_hidden_neurons_init", np.nan))
    
    l2_norm = np.nan
    rescaling_str = params.get("rescaling_on_hidden_neurons_init")
    if rescaling_str:
        try:
            rescaling_list = ast.literal_eval(rescaling_str)
            l2_norm = float(np.linalg.norm(rescaling_list))
        except (ValueError, SyntaxError):
            l2_norm = np.nan

    run_results = []

    for metric_name in metrics_list:
        try:
            history = client.get_metric_history(run.info.run_id, metric_name)
            
            run_results.extend([
                {
                    "method": method,
                    "lr": float(lr),
                    "seed": seed,
                    "metric": metric_name,
                    "step": m.step,
                    "value": m.value,
                    "architecture": arch,
                    "n_params": n_params,
                    "max_rescaling": max_rescale
                }
                for m in history
            ])
        except Exception as e:
            print(f"Erreur sur la métrique {metric_name} pour le run {run.info.run_id}: {e}")
            continue
            
    return run_results

def get_multiple_metrics_history_fast(experiment_name, metrics_list=["train_loss", "test_acc"], max_workers=10):
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        return pd.DataFrame()

    runs = client.search_runs(experiment.experiment_id)
    
    all_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(lambda r: fetch_run_metrics(r, metrics_list, client), runs))
    
    for run_data in results:
        all_data.extend(run_data)

    return pd.DataFrame(all_data)


def plot_multi_metrics_per_lr(df, metrics_to_plot, lrs_to_plot=None, experiment_name=None, 
                               show_std=True, use_sem=False, smoothing=0.0):
    """
    Affiche les métriques avec une option de lissage (EMA).
    smoothing: float entre 0 (pas de lissage) et 1 (lissage max).
    """
    if df.empty: 
        print("DataFrame vide, rien à tracer.")
        return
    n_metrics = len(metrics_to_plot)
    fig_width = 6.75 
    fig_height = 2.2 
    
    plt.rcParams.update({
        'font.size': 9,
        'axes.labelsize': 10,
        'legend.fontsize': 8,
        'font.family': 'serif',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

    name_map = {"train_acc": "train_accuracy", "test_acc": "test_accuracy", "acc": "test_accuracy", "accuracy": "test_accuracy"}
    df = df.copy()
    df['metric'] = df['metric'].replace(name_map)
    normalized_metrics = [name_map.get(m, m) for m in metrics_to_plot]

    display_names = { "baseline": "Baseline",
                     "pathcond": r"$\mathbf{Pathcond}$",
                     "enorm": "Enorm",
                     "bn_baseline": "Baseline",
                     "bn_pathcond": r"$\mathbf{Pathcond}$",
                     "bn_enorm": "Enorm",
                     "pathcond_telep_schedule": r"Pathcond $\times$ Schedule",
                     "bn_pathcond_telep_schedule": r"BN Pathcond $\times$ Sched" }
    style_map = { "baseline": {"color": "#1b9e77", "ls": "-", "marker": "o", "ms": 4},
                 "pathcond": {"color": "#d95f02", "ls": "-", "marker": "s", "ms": 4},
                 "enorm": {"color": "#7570b3", "ls": "-", "marker": "^", "ms": 4},
                 "bn_baseline": {"color": "#1b9e77", "ls": "--", "linewidth": 1.5, "zorder": 2, "marker": "o", "ms": 4},
                 "bn_pathcond": {"color": "#d95f02", "ls": "-", "linewidth": 1.5, "zorder": 1, "marker": "s", "ms": 4},
                 "bn_enorm": {"color": "#7570b3", "ls": "-", "linewidth": 1.5, "zorder": 1, "marker": "^", "ms": 4},
    }

    metric_config = {
        "train_loss": {"label": "Train Loss", "log": True},
        "test_loss": {"label": "Test Loss", "log": True},
        "train_accuracy": {"label": "Train Accuracy", "log": False},
        "test_accuracy": {"label": "Test Accuracy", "log": False},
        "grad_norm": {"label": "Grad. Norm", "log": True},
    }

    unique_lrs = sorted(df["lr"].unique()) if lrs_to_plot is None else sorted(lrs_to_plot)
    
    for lr_val in unique_lrs:
        df_lr = df[df["lr"] == lr_val]
        for arch in df_lr["architecture"].unique():
            df_plot = df_lr[df_lr["architecture"] == arch] if arch else df_lr

            fig, axes = plt.subplots(1, n_metrics, figsize=(fig_width, fig_height), squeeze=False)
            
            for i, metric in enumerate(normalized_metrics):
                ax = axes[0, i]
                df_m = df_plot[df_plot["metric"] == metric]
                
                if df_m.empty: continue

                for method_id, style in style_map.items():
                    subset = df_m[df_m["method"] == method_id]
                    if subset.empty: continue

                    stats = subset.groupby("step")["value"].agg(["mean", "std", "count"]).reset_index()        
                    y_raw = stats["mean"].values
                    if smoothing > 0:
                        ax.plot(stats["step"], y_raw, color=style["color"], alpha=0.2, lw=1.0, zorder=style["zorder"]-1)
                        y_plot = stats["mean"].ewm(alpha=1 - smoothing).mean()
                    else:
                        y_plot = y_raw

                    error = stats["std"] / (np.sqrt(stats["count"]) if use_sem else 1)

                    line, = ax.plot(stats["step"], y_plot, 
                                   label=display_names.get(method_id, method_id),
                                   color=style["color"], linestyle=style["ls"], lw=1.6, zorder=style["zorder"])
                    
                    if show_std:
                        lower = np.maximum(y_raw - error, y_raw * 0.1) if metric_config.get(metric, {}).get("log") else (y_raw - error)
                        ax.fill_between(stats["step"], lower, y_raw + error, 
                                        color=line.get_color(), alpha=0.12)

                config = metric_config.get(metric, {"label": metric, "log": False})
                ax.set_ylabel(config["label"])
                ax.set_xlabel("Epochs")
                if config["log"]: ax.set_yscale("log")
                sns.despine(ax=ax)

            handles, labels = axes[0, 0].get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
                       bbox_to_anchor=(0.5, -0.02), ncol=min(len(by_label), 4), 
                       frameon=False, columnspacing=1.0, handletextpad=0.4)

            plt.tight_layout(rect=[0, 0.05, 1, 0.98])
            
            safe_exp = str(experiment_name).replace("/", "_")
            out_dir = _ensure_outdir(f"figures/multi_metrics/{safe_exp}/")
            filename = f"lr_{lr_val}_arch_{str(arch).replace('/', '_')}_smooth_{smoothing}.pdf"
            
            plt.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
            plt.close(fig)
            print(f"Figure sauvegardée : {filename} (Smoothing: {smoothing})")




def truncated_cmap(cmap, minval=0, maxval=0.93, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}",
        cmap(np.linspace(minval, maxval, n))
    )


if __name__ == "__main__":
    EXP_NAME = "PC_cifar10_Conv_net"
    df = get_multiple_metrics_history_fast(EXP_NAME, metrics_list=["train_loss", "train_acc", "test_acc"])
    plot_multi_metrics_per_lr(df, metrics_to_plot=["train_loss", "train_acc", "test_acc"], experiment_name=EXP_NAME, lrs_to_plot=[0.001], smoothing=0)