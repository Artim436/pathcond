import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from utils import _ensure_outdir


def plot_mean_var_curves(ACC=None,
                         LOSS=None,
                         mood: str = "loss",            # "loss" or "accuracy"
                         ep_teleport: int = 1,
                         outdir: str = "results/",
                         fname_prefix: str = "curve"):
    """
    Plot mean ± std over runs for up to 4 series:
      index 0: SGD (ours)       | index 1: Ref SGD
      index 2: Adam (ours)      | index 3: Ref Adam

    Args:
        ACC:  torch.Tensor or np.ndarray of shape [nb_iter, epochs, 4] (required if mood="accuracy")
        LOSS: torch.Tensor or np.ndarray of shape [nb_iter, epochs, 4] (required if mood="loss")
        mood: "loss" or "accuracy"
        ep_teleport: epoch at which rescaling is applied (vertical line)
        outdir: output directory
        fname_prefix: filename prefix
    """
    assert mood in ("loss", "accuracy"), "mood must be 'loss' or 'accuracy'"
    data = LOSS if mood == "loss" else ACC
    assert data is not None, f"{'LOSS' if mood=='loss' else 'ACC'} is required for mood='{mood}'"

    # ---- to numpy on CPU ----
    try:
        import torch
        if isinstance(data, torch.Tensor):
            data_np = data.detach().cpu().numpy()
        else:
            data_np = np.asarray(data)
    except Exception:
        data_np = np.asarray(data)

    # Validate shape
    if data_np.ndim != 3 or data_np.shape[-1] not in (1, 2, 3, 4):
        raise ValueError("Expected data shape [nb_iter, epochs, C] with C in {1,2,3,4}.")
    nb_iter, epochs, C = data_np.shape

    # Mean/std across runs
    mean = np.nanmean(data_np, axis=0)         # [epochs, C]
    std  = np.nanstd(data_np, axis=0, ddof=0)  # [epochs, C]
    x = np.arange(1, epochs + 1)

    # ---- Output path ----
    Path(outdir).mkdir(parents=True, exist_ok=True)
    filename = f"{fname_prefix}_{'loss' if mood=='loss' else 'acc'}.pdf"
    save_path = Path(outdir) / filename

    # ---- Paper-friendly style ----
    plt.rcParams.update({
        "figure.figsize": (5.0, 3.4),  # single-column
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 1.8,
        "lines.markersize": 3.0,
    })

    # Labels per channel (up to 4)
    labels = [
        r"NT Path Dynamics SGD $\mathbf{(Ours)}$",
        "Baseline SGD",
        r"NT Path Dynamics Adam $\mathbf{(Ours)}$",
        "Baseline Adam",
    ]
    # Distinct markers/linestyles without forcing colors
    markers = ["o", "o", "s", "s"]
    linestyles = ["-", "--", "-", "--"]  # ours solid, baseline dashed

    # ---- Plot ----
    fig, ax = plt.subplots()

    for c in range(C):
        y = mean[:, c]
        s = std[:, c]

        # Skip series that are entirely zero/NaN
        if (np.allclose(y, 0.0) and np.allclose(s, 0.0)) or np.all(~np.isfinite(y)):
            continue

        ax.plot(x, y, marker=markers[c % len(markers)],
                linestyle=linestyles[c % len(linestyles)],
                label=labels[c])
        ax.fill_between(x, y - s, y + s, alpha=0.2)

    # Teleportation epoch line
    if 1 <= ep_teleport <= epochs:
        ax.axvline(x=ep_teleport, linestyle="--", linewidth=1.5, color="black",
                   label="Rescaling applied")

    ax.set_xlabel("Epoch")
    if mood == "accuracy":
        ax.set_ylabel("Accuracy")
        ax.set_title("Test Accuracy (mean ± std)")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss (mean ± std)")
        ax.set_yscale("log")

    # Nice epoch ticks
    step = max(1, int(np.ceil(epochs / 10)))
    ax.set_xticks(np.arange(1, epochs + 1, step))

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)

    return save_path







def _split_by_layers(
    lambdas_history: Sequence[float],
    layer_sizes: Optional[Sequence[int]] = None,
    num_hidden_layers: Optional[int] = None,
    num_iter_optim: int = 1,
) -> List[np.ndarray]:
    """
    Split the flat lambda list by hidden layers.

    Args:
        lambdas_history: flat list of per-hidden-neuron/channel rescalings
        layer_sizes: number of hidden neurons/channels per hidden layer (excludes output layer)
        num_hidden_layers: if layer_sizes is None, try to split evenly across this count

    Returns:
        List of per-layer numpy arrays (one array per hidden layer).
    """
    lambdas_history = np.asarray(lambdas_history, dtype=float)

    if layer_sizes is not None:
        parts = []
        start = 0
        for s in layer_sizes:
            parts.append(lambdas_history[start:start + s])
            start += s
        return parts

    # Fallback: treat everything as one "layer"
    return [lambdas_history]


def _ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


# --- helpers: couleurs & segments -------------------------------------------
def _maybe_set_seaborn():
    """Active un style et une palette Seaborn si dispo, sinon fait rien."""
    try:
        import seaborn as sns  # noqa
        sns.set_context("talk")
        sns.set_style("whitegrid")
        palette = sns.color_palette("deep")
        return True, palette
    except Exception:
        return False, None

def _compute_lambda_boundaries(layer_sizes, nb_iter_optim):
    """
    Renvoie la liste des index (dans lambdas_history) où se terminent
    chaque (layer, pass). Utile pour vlines.
    Exemple: layer_sizes=[5,3], nb_iter_optim=2 -> frontières: [5, 8, 13, 16]
    (on exclut la toute fin si tu veux éviter une vline terminale).
    """
    boundaries = []
    acc = 0
    for _ in range(nb_iter_optim):
        for s in layer_sizes:
            acc += s
            boundaries.append(acc)
    return boundaries

def _draw_vlines(ax, xs, label=None, lw=1.0, ls="--", alpha=0.6):
    """Dessine des lignes verticales sur un Axes, sans spammer la légende."""
    for i, x in enumerate(xs):
        ax.axvline(x, linewidth=lw, linestyle=ls, alpha=alpha,
                   label=(label if i == 0 and label is not None else None))


def plot_rescaling_analysis(
    final_model: nn.Module,
    lambdas_history: Sequence[float],
    norms_history: Sequence[float],
    nb_iter_optim: int = 1,
    name: str = "sgd",
) -> Dict[str, Path]:
    """
    Create publication-ready figures (separate PDFs) to visualize hidden-neuron rescaling results.
    """
    # _set_paper_style()
    used_sns, palette = _maybe_set_seaborn()  # couleurs + style si possible
    new_outdir = "results/"+name+"/"
    out = _ensure_outdir(new_outdir)
    saved = {}

    # --- détection des layers lineaires cachés (exclut la sortie)
    linear_indices = [i for i, layer in enumerate(final_model.model) if isinstance(layer, nn.Linear)]
    num_hidden_layers = len(linear_indices) - 1  # exclude output layer
    num_hidden_neurons = sum(final_model.model[i].out_features for i in linear_indices[:-1])
    assert num_hidden_layers > 0, "Model doit avoir au moins une couche cachée Linear."

    layer_sizes = [layer.out_features for i, layer in enumerate(final_model.model)
                   if isinstance(layer, nn.Linear) and i in linear_indices[:-1]]

    # --- frontières dans lambdas_history pour (layer, pass)
    lambda_boundaries = _compute_lambda_boundaries(layer_sizes, nb_iter_optim)
    # on évite la toute dernière frontière égalant len(lambdas_history)
    lambda_boundaries = [b for b in lambda_boundaries if b < len(lambdas_history)]

    # ---------------------------
    # 1) Histogram of rescaling factors (avec couleurs)
    # ---------------------------
    fig = plt.figure()
    color_hist = (palette[0] if used_sns else None)
    edgecolor = "black"  # reste lisible en N/B
    plt.hist(lambdas_history, bins=30, edgecolor=edgecolor, alpha=0.85, color=color_hist)
    # plt.axvline(1.0, linestyle="--", linewidth=1.5, label=r"$\lambda = 1$", color=(palette[3] if used_sns else None))
    plt.xlabel(r"Rescaling factor $\lambda$")
    plt.ylabel("Count")
    plt.title("Distribution of rescaling factors")
    path = out / ("rescaling_lambda_hist.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved["lambda_hist"] = path

    # ---------------------------
    # 1b) Timeline des λ (ordre d'optim) + vlines layer/pass
    # ---------------------------
    # Ce plot montre les λ dans l'ordre du vecteur plat, pour visualiser les segments.
    fig = plt.figure()
    ax = plt.gca()
    xs = np.arange(len(lambdas_history))
    color_pts = (palette[1] if used_sns else None)
    ax.plot(xs, lambdas_history, marker=".", linestyle="None", alpha=0.8, color=color_pts)
    _draw_vlines(ax, lambda_boundaries, label=f"Layers",
                  lw=0.8, ls="--", alpha=0.4)
    # ax.axhline(1.0, linestyle="--", linewidth=1.0,
    #            color=(palette[3] if used_sns else None), alpha=0.8, label=r"$\lambda=1$")
    ax.set_xlabel("Hidden neuron index")
    ax.set_ylabel(r"$\lambda$")
    ax.set_title(r"Timeline of rescaling factors $\lambda$")
    ax.legend()
    path = out / ("rescaling_lambda_timeline.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved["lambda_timeline"] = path

    # ---------------------------
    # 2) Objective convergence curve (+ vlines par layer si applicable)
    # ---------------------------
    initial_norm = (norms_history[0] if len(norms_history) > 0 else None)

    fig = plt.figure()
    ax = plt.gca()
    color_line = (palette[2] if used_sns else None)
    ax.plot(np.arange(len(norms_history)), norms_history,  color=color_line)

    # if initial_norm is not None:
    #     ax.axhline(initial_norm, linestyle="--", linewidth=1.5,
    #                label=f"Initial value = {initial_norm:.6g}",
    #                color=(palette[4] if used_sns else None))

    # On place des vlines si la longueur épouse nb_iter_optim * num_hidden_layers (p.ex. 1 point/Layer)

    bound = [k*num_hidden_neurons for k in range(1, nb_iter_optim)]

    _draw_vlines(ax, bound, label="Full pass",
                  lw=0.8, ls="--", alpha=0.4)

    ax.set_xlabel("Optimization step")
    ax.set_ylabel(r"$F(z)$")
    ax.set_title("Convergence of the objective")

    path = out / ("rescaling_objective_convergence.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved["objective_convergence"] = path

    # ---------------------------
    # 3) Layer-wise summary (mean ± std of λ) avec couleur
    # ---------------------------
    per_layer = _split_by_layers(lambdas_history, layer_sizes, num_hidden_layers, nb_iter_optim)
    layer_means = [np.mean(x) for x in per_layer]
    layer_stds  = [np.std(x) for x in per_layer]
    x = np.arange(len(per_layer))

    fig = plt.figure()
    ax = plt.gca()
    color_bar = (palette[0] if used_sns else None)
    ax.bar(x, layer_means, yerr=layer_stds, capsize=5, alpha=0.95, color=color_bar)
    # ax.axhline(1.0, linestyle="--", linewidth=1.5, label=r"$\lambda = 1$", color=(palette[3] if used_sns else None))
    ax.set_xlabel("Hidden layer index")
    ax.set_ylabel(r"Mean $\lambda$ (± std)")
    ax.set_title("Layer-wise rescaling factors")
    ax.set_xticks(x, [f"{i}" for i in range(len(per_layer))])

    path = out / ("rescaling_layerwise_summary.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    saved["layerwise_summary"] = path

    return saved




