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


def plot_multi_metrics_per_lr(df, metrics_to_plot, lrs_to_plot=None, experiment_name=None):
    if df.empty: 
        print("DataFrame vide, rien à tracer.")
        return

    # Configuration des noms et styles (inchangée)
    display_names = {
        "baseline": "Baseline",
        "pathcond": r"$\mathbf{Pathcond}$",
        "enorm": "Enorm",
        "bn_baseline": "Baseline",
        "bn_pathcond": r"$\mathbf{Pathcond}$",
        "bn_enorm": "Enorm",
        "pathcond_telep_schedule": r"$\mathbf{Pathcond\ \times Times}$",
        "bn_pathcond_telep_schedule": r"$\mathbf{BN\ Pathcond\ \times Times}$",
    }

    style_map = {
        "baseline":    {"color": "#1b9e77", "ls": "-", "linewidth": 2.0},
        "pathcond":    {"color": "#d95f02", "ls": "-"},
        "enorm":       {"color": "#7570b3", "ls": "-"},
        "bn_baseline": {"color": "#1b9e77", "ls": "--", "linewidth": 2.0},
        "bn_pathcond": {"color": "#d95f02", "ls": "--"},
        "bn_enorm":    {"color": "#7570b3", "ls": "--"},
        "pathcond_telep_schedule": {"color": "#a63603", "ls": "-"},
        "bn_pathcond_telep_schedule": {"color": "#a63603", "ls": "--"},
    }

    unique_lrs = sorted(df["lr"].unique()) if lrs_to_plot is None else sorted(lrs_to_plot)
    unique_architectures = df["architecture"].unique() if "architecture" in df.columns else [None]
    n_metrics = len(metrics_to_plot)

    for lr_val in unique_lrs:
        # On filtre par LR d'abord
        df_for_lr = df[df["lr"] == lr_val]
        if df_for_lr.empty: continue

        for arch in unique_architectures:
            # On filtre par architecture à partir du subset LR (sans écraser df_for_lr)
            if arch is not None:
                df_plot = df_for_lr[df_for_lr["architecture"] == arch]
                title_suffix = f" - Arch: {arch}"
            else:
                df_plot = df_for_lr
                title_suffix = ""

            if df_plot.empty: continue

            # Création de la figure
            fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 4.5), squeeze=False)
            fig.suptitle(f"Learning Rate: {lr_val}{title_suffix}", fontsize=14, fontweight='bold')

            for i, metric in enumerate(metrics_to_plot):
                ax = axes[0, i]
                df_metric = df_plot[df_plot["metric"] == metric]
                
                if df_metric.empty: continue

                for method_id, style in style_map.items():
                    subset = df_metric[df_metric["method"] == method_id]
                    if subset.empty: continue

                    # Calcul moyenne et std par step (pour les différents seeds)
                    stats = subset.groupby("step")["value"].agg(["mean", "std"]).reset_index()

                    ax.plot(stats["step"], stats["mean"], 
                            label=display_names.get(method_id, method_id),
                            color=style["color"], linestyle=style["ls"], lw=style.get("linewidth", 1.8))
                    
                    if stats["std"].notnull().any():
                        ax.fill_between(stats["step"], stats["mean"] - stats["std"], 
                                        stats["mean"] + stats["std"], 
                                        color=style["color"], alpha=0.15)

                ax.set_title(metric.replace("_", " ").upper(), fontsize=12)
                ax.set_xlabel("Epochs / Steps")
                ax.grid(True, alpha=0.3)
                if "loss" in metric.lower(): 
                    ax.set_yscale("log")

            # Gestion de la légende
            handles, labels = axes[0, 0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), frameon=False)
            
            plt.tight_layout(rect=[0, 0.03, 0.98, 0.93])
            
            # Sauvegarde
            
            # Mapping des noms d'expérience pour le fichier
            exp_map = {
                "PC_mnist_enc_dec_3_machines": "mnist_encoder_decoder",
                "PC_cifar10_petit_salon": "cifar10_resnet",
                "PC_moons_diamond": "moons_diamond",
                'PC_fixed_depth': 'fixed_depth',
                'PC_fixed_width': 'fixed_width',
            }
            safe_exp_name = exp_map.get(experiment_name, experiment_name or "experiment")

            out_dir = _ensure_outdir(f"figures/multi_metrics/{safe_exp_name}/")
            
            arch_name = str(arch).replace("/", "_") if arch else "default"
            filename = f"lr_{lr_val}_arch_{arch_name}.pdf"
            
            plt.savefig(out_dir / filename, bbox_inches='tight')
            plt.close(fig) # Important pour libérer la mémoire
            print(f"✅ Figure saved: {filename}")




def truncated_cmap(cmap, minval=0, maxval=0.93, n=256):
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc_{cmap.name}",
        cmap(np.linspace(minval, maxval, n))
    )




def plot_multi_metrics_per_lr_same_fig(df, metrics_to_plot, lrs_to_plot=None, archs_to_plot=None, experiment_name=None, methods_to_plot=None, ax=None):
    if df.empty: return

    standalone = ax is None
    if standalone:
        fig, axes_list = plt.subplots(1, len(metrics_to_plot), figsize=(5 * len(metrics_to_plot), 4.2), squeeze=False)
        axes_list = axes_list.flatten()
    else:
        axes_list = list(ax) if isinstance(ax, (list, np.ndarray)) else [ax]
        metrics_to_plot = metrics_to_plot[:len(axes_list)]

    display_names = { "baseline": "Baseline", "pathcond": r"$\mathbf{Pathcond}$", "enorm": "Enorm", "bn_baseline": "Baseline", "bn_pathcond": r"$\mathbf{Pathcond}$", "bn_enorm": "Enorm", "pathcond_telep_schedule": r"Pathcond $\times$ Schedule", "bn_pathcond_telep_schedule": r"BN Pathcond $\times$ Sched" }
    style_map = { "baseline": {"color": "#1b9e77", "ls": "-", "marker": "o", "ms": 4}, "pathcond": {"color": "#d95f02", "ls": "-", "marker": "s", "ms": 4}, "enorm": {"color": "#7570b3", "ls": "-", "marker": "^", "ms": 4}, "bn_baseline": {"color": "#1b9e77", "ls": "--", "linewidth": 1.5, "zorder": 2}, "bn_pathcond": {"color": "#d95f02", "ls": "-", "linewidth": 1.5, "zorder": 1}, "bn_enorm": {"color": "#7570b3", "ls": "-", "linewidth": 1.5, "zorder": 1} }

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
                                   linestyle=style["ls"], linewidth=1.8, marker=style.get("marker"),
                                   markevery=max(1, len(stats)//10), markersize=style.get("ms", 0), alpha=0.9, zorder=style.get("zorder", 1))
                    
                    if stats["std"].notnull().any():
                        curr_ax.fill_between(stats["step"], stats["mean"] - stats["std"], stats["mean"] + stats["std"], 
                                             color=style["color"], alpha=0.15, zorder=1)

            # Cosmétique
            curr_ax.set_xlabel("Steps")
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

def plot_best_acc_vs_depth(df, lrs_to_plot=None, experiment_name=None):
    if df.empty: return

    df_acc = df[df["metric"] == "test_acc"].copy()
    df_acc["depth"] = df_acc["architecture"].apply(get_depth)

    df_best = df_acc.groupby(["method", "lr", "seed", "depth"])["value"].max().reset_index()

    display_names = {
        "baseline": "Baseline",
        "pathcond": r"$\mathbf{Pathcond}$",
        "enorm": "Enorm",
        "bn_baseline": "Baseline",
        "bn_pathcond": r"$\mathbf{Pathcond}$",
        "bn_enorm": "Enorm",
        "pathcond_telep_schedule": r"$\mathbf{Pathcond\ \times Times}$",
        "bn_pathcond_telep_schedule": r"$\mathbf{BN\ Pathcond\ \times Times}$",
    }

    style_map = {
        "baseline":    {"color": "#1b9e77", "ls": "-", "linewidth": 2.0, "marker": "o"},
        "pathcond":    {"color": "#d95f02", "ls": "-", "marker": "s"},
        "enorm":       {"color": "#7570b3", "ls": "-", "marker": "^"},
        "bn_baseline": {"color": "#1b9e77", "ls": "--", "linewidth": 2.0, "marker": "o"},
        "bn_pathcond": {"color": "#d95f02", "ls": "--", "marker": "s"},
        "bn_enorm":    {"color": "#7570b3", "ls": "--", "marker": "^"},
        "pathcond_telep_schedule": {"color": "#a63603", "ls": "-", "marker": "s"},
        "bn_pathcond_telep_schedule": {"color": "#a63603", "ls": "--", "marker": "s"},
    }

    unique_lrs = sorted(df_best["lr"].unique()) if lrs_to_plot is None else sorted(lrs_to_plot)

    for lr_val in unique_lrs:
        df_lr = df_best[df_best["lr"] == lr_val]
        if df_lr.empty: continue

        plt.figure(figsize=(8, 6))
        
        for method_id, style in style_map.items():
            subset = df_lr[df_lr["method"] == method_id]
            if subset.empty: continue

            # Calcul moyenne et std entre les seeds pour chaque profondeur
            stats = subset.groupby("depth")["value"].agg(["mean", "std"]).reset_index()
            stats = stats.sort_values("depth")

            # Tracer la ligne avec points
            plt.plot(stats["depth"], stats["mean"], 
                     label=display_names.get(method_id, method_id),
                     color=style["color"], 
                     marker=style.get("marker", "o"),
                     linewidth=style.get("linewidth", 1.5),
                     linestyle=style.get("ls", "-"),
                     markersize=8)
            
            # Zone d'ombre pour la variance entre seeds
            plt.fill_between(stats["depth"], 
                             stats["mean"] - stats["std"], 
                             stats["mean"] + stats["std"], 
                             color=style["color"], alpha=0.1)

        plt.title(f"Best Test Accuracy vs Network Depth (LR: {lr_val})", fontsize=13, fontweight='bold')
        plt.xlabel("Number of Layers (Depth)", fontsize=11)
        plt.ylabel("Max Test Accuracy", fontsize=11)
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend(loc='best', frameon=True)
        
        # Forcer des entiers sur l'axe X (car ce sont des couches)
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        exp_map = {
                "PC_mnist_enc_dec_3_machines": "mnist_encoder_decoder",
                "PC_cifar10_petit_salon": "cifar10_resnet"
            }
        safe_exp_name = exp_map.get(experiment_name, experiment_name or "experiment")

        out_dir = _ensure_outdir(f"figures/multi_metrics/{safe_exp_name}/")
        filename = f"accuracy_vs_depth_lr_{lr_val}.pdf"
        plt.savefig(out_dir / filename, bbox_inches='tight')
        print(f"✅ Figure saved: {filename}")



def plot_convergence_speed_vs_depth(
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
    df["depth"] = df["architecture"].apply(get_depth)

    all_convergence = []
    grouped = df.groupby(["method", "lr", "seed", "depth"])

    for (method, lr, seed, depth), group in grouped:
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
                "depth": depth,
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
        "baseline": "Baseline",
        "pathcond": r"$\mathbf{Pathcond}$",
        "enorm": "Enorm",
        "bn_baseline": "Baseline",
        "bn_pathcond": r"$\mathbf{Pathcond}$",
        "bn_enorm": "Enorm",
        "pathcond_telep_schedule": r"$\mathbf{Pathcond \times Times}$",
 #       "bn_pathcond_telep_schedule": r"$\mathbf{BN\ Pathcond \times Times}$",
    }

    # Couleur = méthode, marqueur = BN / non-BN
    style_map = {
        "baseline":    {"color": "#1b9e77", "marker": "o"},
        "pathcond":    {"color": "#d95f02", "marker": "s"},
        "enorm":       {"color": "#7570b3", "marker": "^"},
        "bn_baseline": {"color": "#1b9e77", "marker": "o"},
        "bn_pathcond": {"color": "#d95f02", "marker": "s"},
        "bn_enorm":    {"color": "#7570b3", "marker": "^"},
        "pathcond_telep_schedule": {"color": "#a63603", "marker": "s"},
 #       "bn_pathcond_telep_schedule": {"color": "#a63603", "marker": "s"},
    }

    unique_lrs = (
        sorted(df_conv["lr"].unique())
        if lrs_to_plot is None
        else lrs_to_plot
    )

    for lr_val in unique_lrs:
        df_lr = df_conv[df_conv["lr"] == lr_val]
        if df_lr.empty:
            continue

        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(6.2, 4.2))

        for method_id, style in style_map.items():
            subset = df_lr[df_lr["method"] == method_id]
            if subset.empty:
                continue

            stats = (
                subset
                .groupby("depth")["convergence_step"]
                .agg(["mean", "std"])
                .reset_index()
                .sort_values("depth")
            )

            ax.errorbar(
                stats["depth"],
                stats["mean"],
                yerr=stats["std"],
                linestyle="-",           
                linewidth=0.8,           
                marker=style["marker"],
                color=style["color"],
                markersize=7,
                capsize=2,
                elinewidth=1,
                markeredgecolor="black",
                markeredgewidth=0.6,
                label=display_names.get(method_id, method_id),
            )

        ax.set_xlabel("Number of hidden layers")
        ax.set_ylabel(f"Epochs to reach {target_value}")

        ax.yaxis.grid(True, linestyle="--", alpha=0.25)
        ax.xaxis.grid(False)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

        ax.legend(
            loc="upper right",
            frameon=True,
            markerscale=1.2,
            handletextpad=0.6,
            labelspacing=0.4,
        )

        plt.tight_layout()

        exp_map = {
            "PC_mnist_enc_dec_3_machines": "mnist_encoder_decoder",
            "PC_cifar10_petit_salon": "cifar10_resnet",
            "PC_cifar10_FC_not_all_depth": "cifar10_fc",
        }
        safe_exp_name = exp_map.get(experiment_name, experiment_name or "experiment")
        out_dir = _ensure_outdir(f"figures/{safe_exp_name}/")

        filename = f"conv_speed_depth_lr_{lr_val}.pdf"
        if standalone:
            plt.savefig(out_dir / filename, bbox_inches="tight")
            plt.close()
        print(f"✅ Figure saved: {filename}")
        return ax 






def plot_final_loss_and_rescalings(df, df_rescale, experiment_name):
    if df.empty:
        return

    # --- Utils ---
    def extract_hidden_size(arch_str):
        try:
            arch = ast.literal_eval(arch_str)
            return arch[1] if len(arch) >= 2 else None
        except Exception:
            return None

    # --- Data Prep ---
    df = df.copy()
    df["hidden_size"] = df["architecture"].apply(extract_hidden_size)
    df = df[df["hidden_size"].notna()]

    df_rescale = df_rescale.copy()
    df_rescale["hidden_size"] = df_rescale["architecture"].apply(extract_hidden_size)
    df_rescale = df_rescale[df_rescale["hidden_size"].notna()]

    df_loss = df[df["metric"] == "train_loss"]
    df_final_loss = df_loss.loc[
        df_loss.groupby(["method", "lr", "seed", "hidden_size"])["step"].idxmax()
    ]

    learning_rates = sorted(df_final_loss["lr"].unique())
    INPUT_DIM = 784

    # --- Style ICML ---
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    display_names = {"baseline": "Baseline", "pathcond": "Pathcond"}
    style_map = {
        "baseline": {"color": "#1b9e77", "marker": "o"},
        "pathcond": {"color": "#d95f02", "marker": "s"},
    }

    # --- Plotting ---
    for lr in learning_rates:
        fig, (ax_loss, ax_rescale) = plt.subplots(
            2, 1, figsize=(6.5, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [2, 1]}
        )

        df_lr = df_final_loss[df_final_loss["lr"] == lr].copy()
        df_lr_rescale = df_rescale[df_rescale["lr"] == lr].copy()

        # Calcul des facteurs de compression
        df_lr["compression_factor"] = np.sqrt(INPUT_DIM / df_lr["hidden_size"].astype(float))
        df_lr_rescale["compression_factor"] = np.sqrt(INPUT_DIM / df_lr_rescale["hidden_size"].astype(float))
        
        cf_present = sorted(df_lr["compression_factor"].unique())

        # Top Plot: Loss
        for method, style in style_map.items():
            subset = df_lr[df_lr["method"] == method]
            if subset.empty: continue

            stats = subset.groupby("compression_factor")["value"].agg(["mean", "std"]).reset_index()

            ax_loss.plot(
                stats["compression_factor"], stats["mean"], 
                color=style["color"], linewidth=1.2, alpha=0.35, zorder=2
            )

            ax_loss.errorbar(
                stats["compression_factor"], stats["mean"], yerr=stats["std"],
                linestyle="None", marker=style["marker"], color=style["color"],
                markersize=7, capsize=3, elinewidth=1.2,
                markeredgecolor="black", markeredgewidth=0.6,
                label=display_names[method], zorder=3
            )

        ax_loss.set_ylabel("Final Training Loss")
        ax_loss.yaxis.grid(True, linestyle="--", alpha=0.2)
        ax_loss.legend(frameon=False, loc="upper left")

        # Bottom Plot: Boxplot Rescalings
        df_path = df_lr_rescale[df_lr_rescale["method"] == "pathcond"]
        if not df_path.empty:
            data_cf = sorted(df_path["compression_factor"].unique())
            data = [df_path[df_path["compression_factor"] == cf]["max_rescaling"].values for cf in data_cf]

            bp = ax_rescale.boxplot(
                data,
                positions=data_cf,
                widths=0.4, # On augmente un peu la largeur
                patch_artist=True,
                showfliers=False, # On cache les outliers car on va dessiner TOUS les points
                manage_ticks=False,
                zorder=2
            )

            for box in bp['boxes']:
                box.set(facecolor="#d95f02", alpha=0.3, edgecolor="black", linewidth=0.8)
            for median in bp['medians']:
                median.set(color="black", linewidth=1.5)

            for i, cf in enumerate(data_cf):
                y = data[i]
                x = np.random.normal(cf, 0.04, size=len(y)) 
                ax_rescale.scatter(
                    x, y, 
                    alpha=0.5, 
                    s=10, 
                    color="#d95f02", 
                    edgecolor="none",
                    zorder=3
                )

        ax_rescale.set_xlabel("Compression Factor")
        ax_rescale.set_ylabel(r"$\|\log(\text{rescaling})\|_\infty$") # Version simplifiée ou la tienne
        ax_rescale.yaxis.grid(True, linestyle="--", alpha=0.2)
        
        # Axis Limits
        if cf_present:
            diff = max(cf_present) - min(cf_present)
            margin = diff * 0.15 if diff > 0 else 0.5
            ax_rescale.set_xlim(min(cf_present) - margin, max(cf_present) + margin)
            ax_rescale.set_xticks(cf_present)
            ax_rescale.set_xticklabels([f"{x:.1f}" for x in cf_present])

        plt.tight_layout()
        
        # Save logic
        out_dir = f"figures/multi_metrics/{experiment_name}/"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/final_loss_and_rescalings_lr_{lr}.pdf", bbox_inches="tight")
        plt.close(fig)









from matplotlib.patches import Rectangle, ConnectionPatch

from matplotlib.patches import Rectangle, ConnectionPatch
import numpy as np

def plot_master_zoom_panel(df, lr_val, target_depth_str, target_metric, metrics, target_value, experiment_name, methods):
    n_right = len(metrics)
    width_ratios = [1.5] + [1] * n_right
    fig, axes = plt.subplots(1, 1 + n_right, figsize=(4 * (1 + n_right), 4.5), 
                             gridspec_kw={'width_ratios': width_ratios})
    
    ax_left = axes[0]
    axes_right = axes[1:]

    plot_convergence_speed_vs_depth(
        df, target_value=target_value, lrs_to_plot=[lr_val], 
        ax=ax_left, experiment_name=experiment_name, target_metric=target_metric
    )
    
    plot_multi_metrics_per_lr_same_fig(
        df, metrics_to_plot=metrics, lrs_to_plot=[lr_val], 
        archs_to_plot=[target_depth_str], methods_to_plot=methods, 
        ax=axes_right
    )

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
        y_center = np.mean(y_values) - 0.08 * (ax_left.get_ylim()[1] - ax_left.get_ylim()[0]) # Légère correction vers le bas
    else:
        print(f"⚠️ Warning: No data found for depth {x_center} on ax_left. Using axis center.")
        y_lims = ax_left.get_ylim()
        y_center = (y_lims[0] + y_lims[1]) / 2

    if not np.isfinite(y_center):
        y_center = 0 

    y_range = ax_left.get_ylim()[1] - ax_left.get_ylim()[0]
    rect_w = 0.8  
    rect_h = y_range * 0.5 

    rect = Rectangle(
        (x_center - rect_w/2, y_center - rect_h/2), rect_w, rect_h,
        linewidth=1.2, edgecolor='gray', facecolor='none', 
        linestyle='--', alpha=0.7, zorder=100
    )
    ax_left.add_patch(rect)

    common_style = dict(
        coordsA="data", 
        coordsB="axes fraction",
        axesA=ax_left, 
        axesB=axes_right[0], 
        color="gray", 
        linestyle="--", 
        alpha=0.6,         
        linewidth=1.0,
        zorder=100,         
        clip_on=False       
    )
    # Ligne haute
    con1 = ConnectionPatch(xyA=(x_center + rect_w/2, y_center + rect_h/2), xyB=(0, 1), **common_style)
    # Ligne basse
    con2 = ConnectionPatch(xyA=(x_center + rect_w/2, y_center - rect_h/2), xyB=(0, 0), **common_style)
    
    fig.add_artist(con1)
    fig.add_artist(con2)

    if ax_left.get_legend():
        ax_left.get_legend().remove()
    
    handles, labels = axes_right[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', ncol=len(labels), 
                   bbox_to_anchor=(0.5, -0.05), frameon=True, edgecolor='gray')

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    out_dir = _ensure_outdir(f"figures/combined/")
    filename = f"{experiment_name}_zoom_panel.pdf"
    plt.savefig(out_dir / filename, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Master Zoom Panel saved: {filename}")



# if __name__ == "__main__":
#     EXP_NAME = "PC_cifar10_FC_not_all_depth"
#     # On récupère toutes les métriques nécessaires d'un coup
#     METRICS = ["train_acc", "train_loss" ] 
#     TARGET_LR = 0.001
#     TARGET_DEPTH_STR = "[500, 500, 500]" # L'architecture pour le zoom
#     THRESHOLD = 0.99
    
#     df_all = get_multiple_metrics_history_fast(EXP_NAME, metrics_list=METRICS)
    
#     if not df_all.empty:
#         # plot_multi_metrics_per_lr_same_fig(...)
#         # plot_convergence_speed_vs_depth(...)

#         print(f"Generating Master Zoom Panel for {TARGET_DEPTH_STR}...")
        
#         plot_master_zoom_panel(
#             df=df_all, 
#             lr_val=TARGET_LR, 
#             target_depth_str=TARGET_DEPTH_STR, 
#             target_metric="train_acc",
#             metrics = METRICS,
#             target_value=THRESHOLD,
#             experiment_name=EXP_NAME,
#             methods=['bn_pathcond', 'bn_baseline', 'bn_enorm']
#         )
#     else:
#         print("Aucune donnée récupérée.")




if __name__ == "__main__":
    EXP_NAME = "PC_mnist_enc_dec_Encoder_decoder"
    df = get_multiple_metrics_history_fast(EXP_NAME, metrics_list=["train_loss"])
    df_rescale = get_multiple_metrics_history_fast("PC_mnist_enc_dec_Encoder_decoder_max_rescaling")
    plot_final_loss_and_rescalings(df, df_rescale, EXP_NAME)