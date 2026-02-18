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
from matplotlib.patches import Rectangle, ConnectionPatch



def fetch_run_metrics(run, metrics_list, client):
    """Fonction helper optimisée pour récupérer les métriques d'un seul run."""
    params = run.data.params
    # 1. Extraction et parsing unique (Hors de la boucle)
    lr = params.get("learning_rate") or params.get("lr")
    if lr is None:
        return []
    method = params.get("method", "unknown")
    seed = int(params.get("seed", 0))
    arch = params.get("architecture", None)
    n_params = int(params.get("n_params", -1))
    max_rescale = float(params.get("max_rescaling_on_hidden_neurons_init", np.nan))
    # Calcul de la norme L2 une seule fois par run
    l2_norm = np.nan
    rescaling_str = params.get("rescaling_on_hidden_neurons_init")
    if rescaling_str:
        try:
            # MLflow stocke les listes comme des strings "[0.1, 0.2, ...]"
            rescaling_list = ast.literal_eval(rescaling_str)
            l2_norm = float(np.linalg.norm(rescaling_list))
        except (ValueError, SyntaxError):
            l2_norm = np.nan

    run_results = []

    # 2. Récupération des métriques
    for metric_name in metrics_list:
        try:
            # L'appel API reste le goulot d'étranglement
            history = client.get_metric_history(run.info.run_id, metric_name)
            
            # Utilisation d'une compréhension de liste pour la vitesse
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

    # On récupère tous les runs d'un coup
    runs = client.search_runs(experiment.experiment_id)
    all_data = []
    # Parallélisation des appels API
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # On map la fonction sur chaque run
        results = list(executor.map(lambda r: fetch_run_metrics(r, metrics_list, client), runs)) 
    # Aplatir la liste de listes
    for run_data in results:
        all_data.extend(run_data)

    return pd.DataFrame(all_data)

def truncated_cmap(cmap, minval=0, maxval=0.93, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}",
        cmap(np.linspace(minval, maxval, n))
    )



def plot_multi_metrics_per_lr_same_fig(df, metrics_to_plot, lrs_to_plot=None, archs_to_plot=None, experiment_name=None, methods_to_plot=None, ax=None):
    if df.empty: return
    standalone = ax is None
    if standalone:
        fig, axes_list = plt.subplots(1, len(metrics_to_plot), figsize=(6.75, 2.2), squeeze=False)
        axes_list = axes_list.flatten()
    else:
        axes_list = list(ax) if isinstance(ax, (list, np.ndarray)) else [ax]
        metrics_to_plot = metrics_to_plot[:len(axes_list)]

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

    unique_lrs = sorted(df["lr"].unique()) if lrs_to_plot is None else sorted(lrs_to_plot)
    architectures = [a for a in (archs_to_plot or sorted(df["architecture"].unique())) if a in df["architecture"].unique()] if "architecture" in df.columns else [None]

    for lr_val in unique_lrs:
        df_for_lr = df[df["lr"] == lr_val]
        if methods_to_plot:
            df_for_lr = df_for_lr[df_for_lr["method"].isin(methods_to_plot)]
        if df_for_lr.empty: continue

        # --- BOUCLE SUR LES METRIQUES ---
        for i, metric in enumerate(metrics_to_plot):
            curr_ax = axes_list[i]
            df_metric = df_for_lr[df_for_lr["metric"] == metric]
            seen_labels = set()

            for arch in architectures:
                for method_id, style in style_map.items():
                    subset = df_metric[df_metric["method"] == method_id]
                    if arch is not None: subset = subset[subset["architecture"] == arch]
                    if subset.empty: continue

                    stats = subset.groupby("step")["value"].agg(["mean", "std"]).reset_index()
                    label = f"{display_names.get(method_id, method_id)} ({arch})" if len(architectures) > 1 else display_names.get(method_id, method_id)
                    plot_label = label if label not in seen_labels else None
                    seen_labels.add(label)

                    curr_ax.plot(stats["step"], stats["mean"], label=plot_label, color=style["color"],
                                   linestyle=style["ls"], linewidth=1.8, #, marker=style.get("marker"),
                                  # markevery=max(1, len(stats)//5),
                                  markersize=style.get("ms", 0), alpha=0.9, zorder=style.get("zorder", 1))
                    
                    if stats["std"].notnull().any():
                        curr_ax.fill_between(stats["step"], stats["mean"] - stats["std"], stats["mean"] + stats["std"], 
                                             color=style["color"], alpha=0.15, zorder=1)

            # Cosmétique
            curr_ax.set_xlabel("Epochs")
            # iff train acc -> train accuracy
            if metric == "train_acc":
                curr_ax.set_ylabel("Train Accuracy")
            else:
                curr_ax.set_ylabel(metric.replace("_", " ").title())
            if "loss" in metric.lower(): curr_ax.set_yscale("log")
            curr_ax.grid(True, linestyle='--', alpha=0.5)

        # Légende sur le dernier axe ou via fig.legend
        handles, labels = curr_ax.get_legend_handles_labels()
        if handles and standalone:
            curr_ax.legend(handles, labels, loc='best', frameon=True)
        
        if standalone:
            plt.tight_layout()
            out_dir = _ensure_outdir(f"figures/multi_metrics/{experiment_name}/")
            filename = f"lr_{lr_val}_multi_metrics.pdf"
            plt.savefig(out_dir / filename, bbox_inches='tight')
            plt.close(fig)

def get_depth(arch):
    if pd.isna(arch): return 0
    if isinstance(arch, list): return len(arch)
    # Si c'est une string type "[500, 500]", on compte les virgules + 1
    if isinstance(arch, str):
        if arch == "[]": return 0
        return arch.count(',') + 1
    return 1



def plot_convergence_speed_vs_params(
    df,
    target_metric="test_acc",
    target_value=0.9,
    lrs_to_plot=None,
    experiment_name=None,
    ax=None
):
    if df.empty:
        return
    df = df.copy()

    all_convergence = []
    grouped = df.groupby(["method", "lr", "seed", "n_params"])

    for (method, lr, seed, n_params), group in grouped:
        metric_data = (
            group[group["metric"] == target_metric]
            .sort_values("step")
        )

        if metric_data.empty:
            continue

        if "loss" in target_metric.lower():
            reached = metric_data[metric_data["value"] <= target_value]
        else:
            reached = metric_data[metric_data["value"] >= target_value]

        if not reached.empty:
            all_convergence.append({
                "method": method,
                "lr": lr,
                "seed": seed,
                "n_params": n_params,
                "convergence_step": reached["step"].iloc[0],
            })

    df_conv = pd.DataFrame(all_convergence)
    if df_conv.empty:
        print(f"❌ Aucun run n'a atteint le seuil {target_value}")
        return

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "legend.fontsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    display_names = {
        # "baseline": "Baseline",
        # "pathcond": r"$\mathbf{Pathcond}$",
        # "enorm": "Enorm",
        "bn_baseline": "Baseline",
        "bn_pathcond": r"$\mathbf{Pathcond}$",
        "bn_enorm": "Enorm",
        "pathcond_telep_schedule": r"$\mathbf{Pathcond \times Times}$",
    }

    style_map = {
        # "baseline":    {"color": "#1b9e77", "marker": "o"},
        # "pathcond":    {"color": "#d95f02", "marker": "s"},
        # "enorm":       {"color": "#7570b3", "marker": "^"},
        "bn_baseline": {"color": "#1b9e77", "marker": "o"},
        "bn_pathcond": {"color": "#d95f02", "marker": "s"},
        "bn_enorm":    {"color": "#7570b3", "marker": "^"},
        "pathcond_telep_schedule": {"color": "#a63603", "marker": "s"},
    }

    unique_lrs = (
        sorted(df_conv["lr"].unique())
        if lrs_to_plot is None
        else lrs_to_plot
    )

    standalone = ax is None
    
    if standalone:
        # Créer une figure avec subplots pour tous les lr
        n_lrs = len(unique_lrs)
        fig, axes = plt.subplots(1, n_lrs, figsize=(3.25 * n_lrs, 2.5), sharey=True)
        if n_lrs == 1:
            axes = [axes]
    else:
        axes = [ax]
        unique_lrs = unique_lrs[:1]  # Un seul lr si ax est fourni

    for idx, lr_val in enumerate(unique_lrs):
        df_lr = df_conv[df_conv["lr"] == lr_val]
        if df_lr.empty:
            continue

        current_ax = axes[idx] if standalone else ax

        for method_id, style in style_map.items():
            subset = df_lr[df_lr["method"] == method_id]
            if subset.empty:
                continue

            stats = (
                subset
                .groupby("n_params")["convergence_step"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("n_params")
            )

            current_ax.errorbar(
                stats["n_params"],
                stats["mean"],
                yerr=stats["std"],
                linestyle="none",
                marker=style["marker"],
                color=style["color"],
                markersize=7,
                capsize=2,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=display_names.get(method_id, method_id),
            )

        current_ax.set_xlabel("Number of parameters", fontsize=9)
        if idx == 0:
            current_ax.set_ylabel(f"Epochs to reach {target_value*100:.0f}% Train Acc", fontsize=9)
        
        current_ax.set_title(f"lr = {lr_val}", fontsize=10)

        current_ax.yaxis.grid(True, linestyle="--", alpha=0.5)
        current_ax.xaxis.grid(True, linestyle="--", alpha=0.5)
        current_ax.set_xscale("log")
        current_ax.set_yscale("log")

        def format_params(x, pos):
            if x >= 1e6:
                return f'{x/1e6:.1f}M'
            elif x >= 1e3:
                return f'{x/1e3:.0f}K'
            else:
                return f'{int(x)}'
        
        current_ax.xaxis.set_major_formatter(plt.FuncFormatter(format_params))

        # Supprimer les légendes individuelles
        if current_ax.get_legend():
            current_ax.get_legend().remove()

    if standalone:
        # Légende commune en bas
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc='lower center', ncol=len(labels),
                      fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02),
                      columnspacing=1.0, handletextpad=0.4)

        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        safe_exp_name = experiment_name
        out_dir = _ensure_outdir(f"figures/{safe_exp_name}/")

        filename = f"conv_speed_params_all_lrs.pdf"
        plt.savefig(out_dir / filename, bbox_inches="tight")
        plt.close()
        print(f"✅ Figure saved: {filename}")
    else:
        return ax











def plot_master_zoom_panel(df, lr_val, target_depth_str, target_metric, metrics, target_value, experiment_name, methods):
    n_right = len(metrics)
    width_ratios = [1.2] + [1] * n_right 
    fig, axes = plt.subplots(1, 1 + n_right, 
                         figsize=(6.75, 2.2), # Hauteur réduite pour l'aspect "largeur"
                         gridspec_kw={'width_ratios': width_ratios})
    ax_left = axes[0]
    axes_right = axes[1:]
    plot_convergence_speed_vs_params(
        df, target_value=target_value, lrs_to_plot=[lr_val], 
        ax=ax_left, experiment_name=experiment_name, target_metric=target_metric
    )
    plot_multi_metrics_per_lr_same_fig(
        df, metrics_to_plot=metrics, lrs_to_plot=[lr_val], 
        archs_to_plot=[target_depth_str], methods_to_plot=methods, 
        ax=axes_right
    )
    SMALL_SIZE = 7
    MEDIUM_SIZE = 8
    for ax in axes:
        ax.tick_params(axis='both', which='major', labelsize=SMALL_SIZE)
        ax.xaxis.label.set_size(MEDIUM_SIZE)
        ax.yaxis.label.set_size(MEDIUM_SIZE)
        ax.title.set_size(MEDIUM_SIZE)

    x_center = get_depth(target_depth_str)
    y_values = []
    for line in ax_left.get_lines():
        x_data = np.array(line.get_xdata())

        y_data = np.array(line.get_ydata())
        if len(x_data) > 0:
            idx = np.where(np.isclose(x_data, x_center))[0]
            if len(idx) > 0:
                val = y_data[idx[0]]
                if np.isfinite(val):
                    y_values.append(val)
    if len(y_values) > 0:
        y_center = np.mean(y_values) - 0.08 * (ax_left.get_ylim()[1] - ax_left.get_ylim()[0])
    else:
        print(f"⚠️ Warning: No data found for depth {x_center} on ax_left. Using axis center.")

    y_lims = ax_left.get_ylim()
    y_center = (y_lims[0] + y_lims[1]) / 1.7
    x_lims = ax_left.get_xlim()
    x_center = (x_lims[0] + x_lims[1]) / 2.5
    if not np.isfinite(y_center):
        y_center = 0
    y_range = ax_left.get_ylim()[1] - ax_left.get_ylim()[0]
    x_range = ax_left.get_xlim()[1] - ax_left.get_xlim()[0]
    rect_w = x_range * 0.15

    rect_h = y_range * 0.5
    rect = Rectangle((x_center - rect_w/2, y_center - rect_h/2), rect_w, rect_h,
                     linewidth=0.8, edgecolor='gray', facecolor='none', linestyle='--', alpha=1)
    ax_left.add_patch(rect)

    # Lignes de connexion
    common_style = dict(coordsA="data", coordsB="axes fraction", axesA=ax_left, 
                        axesB=axes_right[0], color="gray", linestyle="--", alpha=1, linewidth=0.8)
    
    fig.add_artist(ConnectionPatch(xyA=(x_center + rect_w/2, y_center + rect_h/2), xyB=(0, 1), **common_style))
    fig.add_artist(ConnectionPatch(xyA=(x_center + rect_w/2, y_center - rect_h/2), xyB=(0, 0), **common_style))

    # Nettoyage et Légende
    if ax_left.get_legend(): ax_left.get_legend().remove()
    handles, labels = axes_right[0].get_legend_handles_labels()
    if handles:
        # Légende plus petite et compacte, rapprochée de la figure
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
                   fontsize=SMALL_SIZE, frameon=False, bbox_to_anchor=(0.5, 0.01),
                   columnspacing=1.0, handletextpad=0.4)

    plt.tight_layout(rect=[0, 0.05, 1, 0.98])
    out_dir = _ensure_outdir(f"figures/combined/")
    plt.savefig(out_dir / f"{experiment_name}_zoom_panel_lr_{lr_val}.pdf", bbox_inches='tight', pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    experiment_name = "PC_cifar10_FC_varying_depth"
    df = get_multiple_metrics_history_fast(experiment_name, metrics_list=["train_loss", "train_acc", "test_acc"])
    plot_master_zoom_panel(lr_val=0.001, target_depth_str="[500, 500, 500]", target_metric="train_acc", metrics=["train_loss", "train_acc"], target_value=0.99, experiment_name=experiment_name, methods=["baseline", "pathcond", "enorm"])