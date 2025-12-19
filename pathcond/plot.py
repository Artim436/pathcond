import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from pathcond.utils import _ensure_outdir
from contextlib import nullcontext as _nullctx
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from pathlib import Path






def plot_mean_var_curves(ACC=None,
                         LOSS=None,
                         mood: str = "loss",            # "loss" or "accuracy"
                         ep_teleport: int = 1,
                         outdir: str = "images/",
                         fname_prefix: str = "curve",
                         lr= None,
                         balanced=None
                         ):
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

        ax.plot(x, y,
                marker=markers[c % len(markers)],
                linestyle=linestyles[c % len(linestyles)],
                label=labels[c])
        ax.plot(x, y,
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
        if lr is not None and balanced is not None:
            ax.set_title(f"Training Loss (mean ± std) — LR={lr:.0e} - {'Balanced' if balanced else 'Unbalanced'}")
        else:
            ax.set_title("Training Loss (mean ± std)")
        if lr is not None and balanced is not None:
            ax.set_title(f"Training Loss (mean ± std) — LR={lr:.0e} - {'Balanced' if balanced else 'Unbalanced'}")
        else:
            ax.set_title("Training Loss (mean ± std)")
        ax.set_yscale("log")

    # Nice epoch ticks
    step = max(1, int(np.ceil(epochs / 10)))
    ax.set_xticks(np.arange(1, epochs + 1, step))
    ax.set_xscale("log")
    ax.set_xscale("log")

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return save_path


def plot_mean_var_curves_all_lr(ACC=None,
                         LOSS=None,
                         mood: str = "loss",            # "loss" or "accuracy"
                         ep_teleport: int = 1,
                         outdir: str = "images/",
                         fname_prefix: str = "curve",
                         learning_rates=None
                         ):
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
    if data_np.ndim != 4 or data_np.shape[-1] not in (1, 2, 3, 4):
        raise ValueError("Expected data shape [nb_iter, epochs, C] with C in {1,2,3,4}.")
    nb_lr, nb_iter, epochs, C = data_np.shape

    # Mean/std across runs
    mean = np.nanmean(data_np, axis=1)         # [epochs, C]
    std  = np.nanstd(data_np, axis=1, ddof=0)  # [epochs, C]
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

    norm = LogNorm(vmin=min(learning_rates), vmax=max(learning_rates))
    cmap = plt.cm.viridis

    for lr_it, lr in enumerate(learning_rates):
        for c in range(C):
            y = mean[lr_it, :, c]
            s = std[lr_it, :, c]

            # Skip series that are entirely zero/NaN
            if (np.allclose(y, 0.0) and np.allclose(s, 0.0)) or np.all(~np.isfinite(y)):
                continue
            label = labels[c] if lr_it == 0 else None  # only label once
            ax.plot(
                x, y,
                linestyle=linestyles[c % len(linestyles)],
                color=cmap(norm(lr)),
                label=label
            )
            ax.fill_between(x, y - s, y + s, alpha=0.2, color=cmap(norm(lr)))

    # Ajouter une colorbar pour représenter les valeurs de lr
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])  # requis pour matplotlib < 3.6
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Learning rate", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    # Optionnel : ajuster la taille de police globale de la figure
    # ax.tick_params(labelsize=8)
    # ax.set_title("Évolution des métriques selon le learning rate", fontsize=10)


    # Teleportation epoch line
    # if 1 <= ep_teleport <= epochs:
    #     ax.axvline(x=ep_teleport, linestyle="--", linewidth=1.5, color="black",
    #                label="Rescaling applied")


    ax.set_xlabel("Epoch")
    if mood == "accuracy":
        ax.set_ylabel("Accuracy")
        ax.set_title("Test Accuracy (mean ± std)")
        ax.set_ylim(0.0, 1.0)
    else:
        ax.set_ylabel("Loss")

        ax.set_title("Training Loss (mean ± std)")
        # ax.set_yscale("log")

    # Nice epoch ticks
    step = max(1, int(np.ceil(epochs / 10)))
    ax.set_xticks(np.arange(1, epochs + 1, step))
    ax.set_xscale("log")

    ax.legend(loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

    return save_path

def plot_mean_var_curves_psnr(
    LOSS=None,
    ACC_TRAIN=None,
    ACC_TEST=None,
    learning_rates=None,
    ep_teleport: int = None,        # ex: 1 ou None pour ne pas tracer
    outdir: str = "images/",
    fname: str = "curves_triptych.pdf",
    show_panels=("loss", "train", "test"),  # 👈 nouveau paramètre
):
    """
    Crée une figure unique avec 1 à 3 sous-graphiques (LOSS, ACC_TRAIN, ACC_TEST),
    selon 'show_panels'. Colore les courbes via une colormap log(learning_rate),
    et ajoute une colorbar commune.

    Paramètres
    ----------
    show_panels : tuple ou str
        Exemples :
            - "loss" : n'affiche que la courbe de perte
            - ("loss", "train") : affiche 2 panels
            - ("loss", "train", "test") : les 3 (par défaut)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from pathlib import Path

    # ✅ Accepte show_panels="loss" au lieu de ("loss",)
    if isinstance(show_panels, str):
        show_panels = (show_panels,)

    assert learning_rates is not None and len(learning_rates) > 0, "learning_rates requis"
    if np.any(np.asarray(learning_rates) <= 0):
        raise ValueError("learning_rates doivent être strictement positifs pour LogNorm.")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_path = Path(outdir) / fname

    # Style global
    plt.rcParams.update({
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

    # --- Colormap log ---
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 0.85, 256))  # tronque avant le jaune
    cmap = LinearSegmentedColormap.from_list("trunc_plasma", colors)
    norm = LogNorm(vmin=float(np.min(learning_rates)), vmax=float(np.max(learning_rates)))

    # --- Conversion torch → numpy ---
    def _to_np(data):
        if data is None:
            return None
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except Exception:
            pass
        data = np.asarray(data)
        if data.ndim != 4 or data.shape[-1] not in (1, 2, 3, 4):
            raise ValueError("Chaque data doit être de shape [nb_lr, nb_iter, epochs, C].")
        return data

    # --- Sélection dynamique des panels ---
    panel_defs = {
        "loss":  (LOSS,      "Training Loss",  "Loss"),
        "train": (ACC_TRAIN, "Train PSNR", "PSNR"),
        "test":  (ACC_TEST,  "Test PSNR",  "PSNR"),
    }
    selected_panels = [panel_defs[k] for k in show_panels if k in panel_defs]

    if len(selected_panels) == 0:
        raise ValueError(f"Aucun panel valide dans show_panels={show_panels}")

    # --- Figure adaptée au nombre de panels ---
    fig_width = 4.0 * len(selected_panels)
    fig_height = 3.5
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)

    fig, axes = plt.subplots(1, len(selected_panels), sharey=False)
    if len(selected_panels) == 1:
        axes = [axes]

    # --- Légende une seule fois ---
    legend_done = False

    for ax, (data, title, ylab) in zip(axes, selected_panels):
        data_np = _to_np(data)
        if data_np is None:
            ax.set_visible(False)
            continue

        nb_lr, nb_iter, epochs, C = data_np.shape
        mean = np.nanmean(data_np, axis=1)
        std  = np.nanstd(data_np, axis=1, ddof=0)
        x = np.arange(0, epochs)

        for lr_it, lr in enumerate(learning_rates):
            for c in range(C):
                y = mean[lr_it, :, c]
                s = std[lr_it, :, c]
                if (np.allclose(y, 0.0) and np.allclose(s, 0.0)) or np.all(~np.isfinite(y)):
                    continue

                linestyle = "-" if c % 2 == 0 else "--"
                ax.plot(
                    x, y,
                    linestyle=linestyle,
                    color=cmap(norm(lr))
                )
                ax.fill_between(x, y - s, y + s,
                                alpha=0.2 if ylab != "PSNR" else 0.1,
                                color=cmap(norm(lr)))

        # Ligne de "teleport"
        if ep_teleport is not None and 1 <= ep_teleport <= x[-1]:
            ax.axvline(x=ep_teleport, linestyle="--", linewidth=1.2, color="black")

        # Légende une seule fois
        if not legend_done and title == "Training Loss":
            style_legend = [
                Line2D([0], [0], color='k', linestyle='--', label=r'NT Path Cond GD ($\mathbf{Ours}$)'),
                Line2D([0], [0], color='k', linestyle='-', label='Baseline GD')
            ]
            ax.legend(handles=style_legend, loc="upper left", frameon=False)
            legend_done = True

        # ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.set_ylabel(ylab)

    # --- Colorbar commune ---
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    vmin, vmax = np.min(learning_rates), np.max(learning_rates)
    log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar = fig.colorbar(
        sm,
        ax=[a for a in axes if a.get_visible()],
        orientation="horizontal",
        fraction=0.05,
        pad=0.10,
        aspect=30
    )
    cbar.set_label("Learning rate", fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks])

    # --- Mise en page ---
    # fig.suptitle("Scaling " , fontsize=12)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    fig.savefig(save_path)
    plt.close(fig)
    return save_path

def plot_mean_var_curves_triptych(
    LOSS=None,
    ACC_TRAIN=None,
    ACC_TEST=None,
    learning_rates=None,
    ep_teleport: int = None,        # ex: 1 ou None pour ne pas tracer
    outdir: str = "images/",
    fname: str = "curves_triptych.pdf",
    title_suffix: str = "with effects on training dynamics",
    show_panels=("loss", "train", "test"),  # 👈 nouveau paramètre
):
    """
    Crée une figure unique avec 1 à 3 sous-graphiques (LOSS, ACC_TRAIN, ACC_TEST),
    selon 'show_panels'. Colore les courbes via une colormap log(learning_rate),
    et ajoute une colorbar commune.

    Paramètres
    ----------
    show_panels : tuple ou str
        Exemples :
            - "loss" : n'affiche que la courbe de perte
            - ("loss", "train") : affiche 2 panels
            - ("loss", "train", "test") : les 3 (par défaut)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from pathlib import Path

    # ✅ Accepte show_panels="loss" au lieu de ("loss",)
    if isinstance(show_panels, str):
        show_panels = (show_panels,)

    assert learning_rates is not None and len(learning_rates) > 0, "learning_rates requis"
    if np.any(np.asarray(learning_rates) <= 0):
        raise ValueError("learning_rates doivent être strictement positifs pour LogNorm.")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_path = Path(outdir) / fname

    # Style global
    plt.rcParams.update({
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

    # --- Colormap log ---
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 0.85, 256))  # tronque avant le jaune
    cmap = LinearSegmentedColormap.from_list("trunc_plasma", colors)
    norm = LogNorm(vmin=float(np.min(learning_rates)), vmax=float(np.max(learning_rates)))

    # --- Conversion torch → numpy ---
    def _to_np(data):
        if data is None:
            return None
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except Exception:
            pass
        data = np.asarray(data)
        if data.ndim != 4 or data.shape[-1] not in (1, 2, 3, 4):
            raise ValueError("Chaque data doit être de shape [nb_lr, nb_iter, epochs, C].")
        return data

    # --- Sélection dynamique des panels ---
    panel_defs = {
        "loss":  (LOSS,      "Training Loss",  "Loss"),
        "train": (ACC_TRAIN, "Train Accuracy", "Accuracy"),
        "test":  (ACC_TEST,  "Test Accuracy",  "Accuracy"),
    }
    selected_panels = [panel_defs[k] for k in show_panels if k in panel_defs]

    if len(selected_panels) == 0:
        raise ValueError(f"Aucun panel valide dans show_panels={show_panels}")

    # --- Figure adaptée au nombre de panels ---
    fig_width = 4.0 * len(selected_panels)
    fig_height = 3.5
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)

    fig, axes = plt.subplots(1, len(selected_panels), sharey=False)
    if len(selected_panels) == 1:
        axes = [axes]

    # --- Légende une seule fois ---
    legend_done = False

    for ax, (data, title, ylab) in zip(axes, selected_panels):
        data_np = _to_np(data)
        if data_np is None:
            ax.set_visible(False)
            continue

        nb_lr, nb_iter, epochs, C = data_np.shape
        mean = np.nanmean(data_np, axis=1)
        std  = np.nanstd(data_np, axis=1, ddof=0)
        x = np.arange(0, epochs)

        for lr_it, lr in enumerate(learning_rates):
            for c in range(C):
                y = mean[lr_it, :, c]
                s = std[lr_it, :, c]
                if (np.allclose(y, 0.0) and np.allclose(s, 0.0)) or np.all(~np.isfinite(y)):
                    continue

                linestyle = "-" if c % 2 == 0 else "--"
                ax.plot(
                    x, y,
                    linestyle=linestyle,
                    color=cmap(norm(lr))
                )
                ax.fill_between(x, y - s, y + s,
                                alpha=0.2 if ylab != "Accuracy" else 0.1,
                                color=cmap(norm(lr)))

        # Ligne de "teleport"
        if ep_teleport is not None and 1 <= ep_teleport <= x[-1]:
            ax.axvline(x=ep_teleport, linestyle="--", linewidth=1.2, color="black")

        # Légende une seule fois
        if not legend_done and title == "Training Loss":
            style_legend = [
                Line2D([0], [0], color='k', linestyle='-', label=r'NT Path Cond GD ($\mathbf{Ours}$)'),
                Line2D([0], [0], color='k', linestyle='--', label='Baseline GD')
            ]
            ax.legend(handles=style_legend, loc="upper left", frameon=False)
            legend_done = True

        # ax.set_xscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if ylab == "Accuracy":
            ax.set_ylim(top=1.0)
        else:
            ax.set_ylim(bottom=0.0, top=0.9)

    # --- Colorbar commune ---
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    vmin, vmax = np.min(learning_rates), np.max(learning_rates)
    log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar = fig.colorbar(
        sm,
        ax=[a for a in axes if a.get_visible()],
        orientation="horizontal",
        fraction=0.05,
        pad=0.10,
        aspect=30
    )
    cbar.set_label("Learning rate", fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks])

    # --- Mise en page ---
    fig.suptitle("Scaling " + title_suffix, fontsize=12)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_mean_var_curves_triptych_init(
    LOSS=None,
    ACC_TRAIN=None,
    ACC_TEST=None,
    inits=None,
    ep_teleport: int = None,        # ex: 1 ou None pour ne pas tracer
    outdir: str = "images/",
    fname: str = "curves_triptych.pdf",
    title_suffix: str = "with effects on training dynamics",
    show_panels=("loss", "train", "test"),  # 👈 nouveau paramètre
):
    """
    Crée une figure unique avec 1 à 3 sous-graphiques (LOSS, ACC_TRAIN, ACC_TEST),
    selon 'show_panels'. Colore les courbes via une colormap log(learning_rate),
    et ajoute une colorbar commune.

    Paramètres
    ----------
    show_panels : tuple ou str
        Exemples :
            - "loss" : n'affiche que la courbe de perte
            - ("loss", "train") : affiche 2 panels
            - ("loss", "train", "test") : les 3 (par défaut)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from pathlib import Path

    # ✅ Accepte show_panels="loss" au lieu de ("loss",)
    if isinstance(show_panels, str):
        show_panels = (show_panels,)

    assert inits is not None and len(inits) > 0, "learning_rates requis"
    if np.any(np.asarray(inits) <= 0):
        raise ValueError("inits doivent être strictement positifs pour LogNorm.")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_path = Path(outdir) / fname

    # Style global
    plt.rcParams.update({
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

    # --- Colormap log ---
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 0.85, 256))  # tronque avant le jaune
    cmap = LinearSegmentedColormap.from_list("trunc_plasma", colors)
    norm = LogNorm(vmin=float(np.min(inits)), vmax=float(np.max(inits)))

    # --- Conversion torch → numpy ---
    def _to_np(data):
        if data is None:
            return None
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except Exception:
            pass
        data = np.asarray(data)
        if data.ndim != 4 or data.shape[-1] not in (1, 2, 3, 4):
            raise ValueError("Chaque data doit être de shape [nb_lr, nb_iter, epochs, C].")
        return data

    # --- Sélection dynamique des panels ---
    panel_defs = {
        "loss":  (LOSS,      "Training Loss",  "Loss"),
        "train": (ACC_TRAIN, "Train Accuracy", "Accuracy"),
        "test":  (ACC_TEST,  "Test Accuracy",  "Accuracy"),
    }
    selected_panels = [panel_defs[k] for k in show_panels if k in panel_defs]

    if len(selected_panels) == 0:
        raise ValueError(f"Aucun panel valide dans show_panels={show_panels}")

    # --- Figure adaptée au nombre de panels ---
    fig_width = 4.0 * len(selected_panels)
    fig_height = 3.5
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)

    fig, axes = plt.subplots(1, len(selected_panels), sharey=False)
    if len(selected_panels) == 1:
        axes = [axes]

    # --- Légende une seule fois ---
    legend_done = False

    for ax, (data, title, ylab) in zip(axes, selected_panels):
        data_np = _to_np(data)
        if data_np is None:
            ax.set_visible(False)
            continue

        nb_lr, nb_iter, epochs, C = data_np.shape
        mean = np.nanmean(data_np, axis=1)
        std  = np.nanstd(data_np, axis=1, ddof=0)
        x = np.arange(0, epochs)

        for lr_it, lr in enumerate(inits):
            for c in range(C):
                y = mean[lr_it, :, c]
                s = std[lr_it, :, c]
                if (np.allclose(y, 0.0) and np.allclose(s, 0.0)) or np.all(~np.isfinite(y)):
                    continue

                linestyle = "-" if c % 2 == 0 else "--"
                ax.plot(
                    x, y,
                    linestyle=linestyle,
                    color=cmap(norm(lr))
                )
                ax.fill_between(x, y - s, y + s,
                                alpha=0.2 if ylab != "Accuracy" else 0.1,
                                color=cmap(norm(lr)))

        # Ligne de "teleport"
        if ep_teleport is not None and 1 <= ep_teleport <= x[-1]:
            ax.axvline(x=ep_teleport, linestyle="--", linewidth=1.2, color="black")

        # Légende une seule fois
        if not legend_done and title == "Training Loss":
            style_legend = [
                Line2D([0], [0], color='k', linestyle='-', label=r'NT Path Cond GD ($\mathbf{Ours}$)'),
                Line2D([0], [0], color='k', linestyle='--', label='Baseline GD')
            ]
            ax.legend(handles=style_legend, loc="upper left", frameon=False)
            legend_done = True

        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if ylab == "Accuracy":
            ax.set_ylim(top=1.0)
        # else:
        #     ax.set_ylim(bottom=0.0, top=0.9)

    # --- Colorbar commune ---
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    vmin, vmax = np.min(inits), np.max(inits)
    log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar = fig.colorbar(
        sm,
        ax=[a for a in axes if a.get_visible()],
        orientation="horizontal",
        fraction=0.05,
        pad=0.10,
        aspect=30
    )
    cbar.set_label("Init", fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks])

    # --- Mise en page ---
    fig.suptitle("Scaling " + title_suffix, fontsize=12)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    fig.savefig(save_path)
    plt.close(fig)
    return save_path




def plot_mean_var_curves_triptych_epochs_times_lr(
    LOSS=None,
    ACC_TRAIN=None,
    ACC_TEST=None,
    learning_rates=None,
    ep_teleport: int = None,
    outdir: str = "images/",
    fname: str = "curves_triptych.pdf",
    title_suffix: str = "with no effects on training dynamics",
    show_panels=("loss", "train", "test"),
):
    """
    Crée une figure avec 1 à 3 sous-graphiques (LOSS, ACC_TRAIN, ACC_TEST),
    selon 'show_panels'. Chaque courbe est tracée à sa longueur maximale,
    sans troncature, et colorée selon log(learning_rate).
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from pathlib import Path

    # --- tolère show_panels="loss" au lieu de ("loss",)
    if isinstance(show_panels, str):
        show_panels = (show_panels,)

    assert learning_rates is not None and len(learning_rates) > 0, "learning_rates requis"
    if np.any(np.asarray(learning_rates) <= 0):
        raise ValueError("learning_rates doivent être strictement positifs pour LogNorm.")

    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_path = Path(outdir) / fname

    # --- Style global ---
    plt.rcParams.update({
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

    # --- Colormap ---
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 0.85, 256))  # tronque avant le jaune vif
    cmap = LinearSegmentedColormap.from_list("trunc_plasma", colors)
    norm = LogNorm(vmin=float(np.min(learning_rates)), vmax=float(np.max(learning_rates)))

    # --- Helper torch→numpy ---
    def _to_np(data):
        if data is None:
            return None
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except Exception:
            pass
        data = np.asarray(data)
        if data.ndim != 4 or data.shape[-1] not in (1, 2, 3, 4):
            raise ValueError("Chaque data doit être de shape [nb_lr, nb_iter, epochs, C].")
        return data

    # --- Sélection dynamique des panels ---
    panel_defs = {
        "loss":  (LOSS,      "Training Loss",  "Loss"),
        "train": (ACC_TRAIN, "Train Accuracy", "Accuracy"),
        "test":  (ACC_TEST,  "Test Accuracy",  "Accuracy"),
    }
    selected_panels = [panel_defs[k] for k in show_panels if k in panel_defs]

    if len(selected_panels) == 0:
        raise ValueError(f"Aucun panel valide dans show_panels={show_panels}")

    # --- Figure adaptée au nombre de panels ---
    fig_width = 4.0 * len(selected_panels)     # ~4 pouces par panel
    fig_height = 3.5
    plt.rcParams["figure.figsize"] = (fig_width, fig_height)

    fig, axes = plt.subplots(1, len(selected_panels), sharey=False)
    if len(selected_panels) == 1:
        axes = [axes]  # uniformiser l'itération

    # --- Légende une seule fois ---
    legend_shown = False

    for ax, (data, title, ylab) in zip(axes, selected_panels):
        data_np = _to_np(data)
        if data_np is None:
            ax.set_visible(False)
            continue

        nb_lr, nb_iter, epochs, C = data_np.shape
        mean = np.nanmean(data_np, axis=1)
        std  = np.nanstd(data_np, axis=1, ddof=0)

        for lr_it, lr in enumerate(learning_rates):
            stride = max(1, int(round(1 / lr)))
            idx = np.arange(0, epochs, stride)
            x_vals = np.arange(1, len(idx) + 1)

            # 🔽 Z-order inverse à la longueur (plus long = dessous)
            z = -len(idx)

            for c in range(C):
                y = mean[lr_it, idx, c]
                s = std[lr_it, idx, c]
                if (np.allclose(y, 0.0) and np.allclose(s, 0.0)) or np.all(~np.isfinite(y)):
                    continue

                linestyle = "-" if c % 2 == 0 else "--"
                ax.plot(
                    x_vals, y,
                    linestyle=linestyle,
                    color=cmap(norm(lr)),
                    zorder=z
                )

                fill_alpha = 0.1 if ylab == "Accuracy" else 0.2
                ax.fill_between(
                    x_vals,
                    y - (s/10 if ylab == "Accuracy" else s),
                    y + (s/10 if ylab == "Accuracy" else s),
                    alpha=fill_alpha,
                    color=cmap(norm(lr)),
                    zorder=z
                )


        # Ligne de "teleport" si demandée
        if ep_teleport is not None:
            ax.axvline(x=ep_teleport, linestyle="--", linewidth=1.2, color="black")

        # Légende une seule fois
        if not legend_shown and title == "Training Loss":
            style_legend = [
                Line2D([0], [0], color='k', linestyle='-', label=r'NT Path Cond GD ($\mathbf{Ours}$)'),
                Line2D([0], [0], color='k', linestyle='--', label='Baseline GD'),
            ]
            ax.legend(handles=style_legend, loc="upper left", frameon=False)
            legend_shown = True

        ax.set_xscale("log")
        ax.set_xlabel(r"Epoch $\times$ learning rate")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if ylab == "Accuracy":
            ax.set_ylim(top=1.0)
        else:
            ax.set_ylim(bottom=0.0, top=0.7)

    # --- Colorbar commune ---
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    vmin, vmax = np.min(learning_rates), np.max(learning_rates)
    log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar = fig.colorbar(
        sm,
        ax=[a for a in axes if a.get_visible()],
        orientation="horizontal",
        fraction=0.05,
        pad=0.10,
        aspect=30
    )
    cbar.set_label("Learning rate", fontsize=9)
    cbar.ax.tick_params(labelsize=9)
    cbar.set_ticks(log_ticks)
    # cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks])

    # --- Mise en page et sauvegarde ---
    # fig.suptitle("Scaling " + title_suffix, fontsize=12)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    fig.savefig(save_path)
    plt.close(fig)
    return save_path



def plot_mean_var_curves_triptych_epochs_times_lr_archi(
    LOSS=None,
    ACC_TRAIN=None,
    ACC_TEST=None,
    learning_rates=None,
    nb_params=None,  # 👈 optionnel si plusieurs architectures
    ep_teleport: int = None,
    outdir: str = "images/",
    fname: str = "curves_triptych.pdf",
    title_suffix: str = "with no effects on training dynamics",
    show_panels=("loss", "train", "test"),
):
    """
    Si LOSS.shape = (len(architectures), nb_lr, nb_iter, epochs, C),
    la colorbar reflète la taille du réseau (nb_params) au lieu du LR.
    Pour chaque courbe : axe des abscisses = epochs * lr.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from pathlib import Path

    if isinstance(show_panels, str):
        show_panels = (show_panels,)

    assert learning_rates is not None and len(learning_rates) > 0, "learning_rates requis"

    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_path = Path(outdir) / fname

    # --- Style global ---
    plt.rcParams.update({
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

    # --- Colormap ---
    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 0.85, 256))
    cmap = LinearSegmentedColormap.from_list("trunc_plasma", colors)

    # --- Helper torch→numpy ---
    def _to_np(data):
        if data is None:
            return None
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(data)

    # --- Panels à tracer ---
    panel_defs = {
        "loss":  (LOSS,      "Training Loss",  "Loss"),
        "train": (ACC_TRAIN, "Train Accuracy", "Accuracy"),
        "test":  (ACC_TEST,  "Test Accuracy",  "Accuracy"),
    }
    selected_panels = [panel_defs[k] for k in show_panels if k in panel_defs]
    if not selected_panels:
        raise ValueError(f"Aucun panel valide dans show_panels={show_panels}")

    # --- Figure ---
    fig, axes = plt.subplots(1, len(selected_panels), figsize=(4.0 * len(selected_panels), 3.5))
    if len(selected_panels) == 1:
        axes = [axes]

    legend_shown = False
    use_arch_mode = False

    for ax, (data, title, ylab) in zip(axes, selected_panels):
        data_np = _to_np(data)
        if data_np is None:
            ax.set_visible(False)
            continue

        if data_np.ndim == 5:
            # mode multi-architectures
            use_arch_mode = True
            n_arch, nb_lr, nb_iter, epochs, C = data_np.shape
            print(f"Tracé en mode multi-architectures avec n_arch={n_arch}")
        elif data_np.ndim == 4:
            n_arch, nb_lr, nb_iter, epochs, C = 1, *data_np.shape
            data_np = data_np[None, ...]
        else:
            raise ValueError("data doit être [nb_lr, nb_iter, epochs, C] ou [n_arch, nb_lr, nb_iter, epochs, C].")

        mean = np.nanmean(data_np, axis=2)
        std = np.nanstd(data_np, axis=2, ddof=0)

        # --- Couleurs : selon arch ou selon lr ---
        if use_arch_mode:
            assert nb_params is not None and len(nb_params) == n_arch, \
                f"nb_params doit être fourni et correspondre à len(architectures)={n_arch} \noteq len(nb_params)={len(nb_params) if nb_params is not None else 'None'}"
            norm = LogNorm(vmin=float(np.min(nb_params)), vmax=float(np.max(nb_params)))
            color_vals = cmap(norm(nb_params))
        else:
            norm = LogNorm(vmin=float(np.min(learning_rates)), vmax=float(np.max(learning_rates)))
            color_vals = cmap(norm(learning_rates))

        # --- Tracé ---

        for a in range(n_arch):
            for lr_it, lr in enumerate(learning_rates):
                x_vals = np.arange(epochs)
                for c in range(C):
                    linestyle = "-" if c % 2 == 0 else "--"
                    color = color_vals[a] if use_arch_mode else cmap(norm(lr))
                    y = mean[a, lr_it, :, c]
                    s = std[a, lr_it, :, c]
                    if np.all(~np.isfinite(y)):
                        continue
                    ax.plot(x_vals, y, linestyle=linestyle, color=color, alpha=0.9)
                    ax.fill_between(
                        x_vals,
                        y - (s/2),
                        y + (s/2),
                        color=color, alpha=0.15
                    )

        if ep_teleport is not None:
            ax.axvline(x=ep_teleport, linestyle="--", linewidth=1.2, color="black")

        if not legend_shown and title == "Training Loss":
            style_legend = [
                Line2D([0], [0], color='k', linestyle='-', label=r'NT Path Cond GD ($\mathbf{Ours}$)'),
                Line2D([0], [0], color='k', linestyle='--', label='Baseline GD'),
            ]
            ax.legend(handles=style_legend, loc="lower left", frameon=False)  # 👈 légende déplacée
            legend_shown = True

        #ax.set_xscale("log")
        ax.set_xlabel(r"Epoch")
        # ax.set_yscale("log" if ylab == "Loss" else "linear")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if ylab == "Accuracy":
            ax.set_ylim(top=1.0)
            ax.set_xlim(left=0.0)
        else:
            ax.set_xlim(left=0.0)

    # --- Colorbar (verticale, échelle log, bien positionnée) ---
    vmin, vmax = (
        float(min(nb_params)), float(max(nb_params))
    ) if use_arch_mode else (
        float(np.min(learning_rates)), float(np.max(learning_rates))
    )

    # Échelle logarithmique (toujours en log, que ce soit nb_params ou lr)
    from matplotlib.colors import LogNorm
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    # Création de la colorbar
    cbar = fig.colorbar(
        sm,
        ax=[a for a in axes if a.get_visible()],
        orientation="horizontal",
        fraction=0.05,
        pad=0.5,
        aspect=35,
        anchor=(0.5, 0.0)
    )

    log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks])

    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(True)

    # Label selon le mode
    if use_arch_mode:
        cbar.set_label("Number of parameters", fontsize=9)
    else:
        cbar.set_label("Learning rate", fontsize=9)


    # fig.suptitle("Scaling " + title_suffix, fontsize=12)
    fig.subplots_adjust(bottom=0.3, wspace=0.3)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def plot_scatter_archi(
    LOSS=None,
    ACC_TRAIN=None,
    ACC_TEST=None,
    learning_rates=None,
    nb_params=None,
    ep_teleport: int = None,
    outdir: str = "images/",
    fname: str = "curves_triptych.pdf",
    title_suffix: str = "with no effects on training dynamics",
    show_panels=("loss", "train", "test"),
):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, LinearSegmentedColormap
    from matplotlib.cm import ScalarMappable
    from matplotlib.lines import Line2D
    from pathlib import Path

    if isinstance(show_panels, str):
        show_panels = (show_panels,)

    assert learning_rates is not None and len(learning_rates) > 0

    Path(outdir).mkdir(parents=True, exist_ok=True)
    save_path = Path(outdir) / fname

    plt.rcParams.update({
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "lines.linewidth": 2.0,
        "lines.markersize": 7.0,
    })

    cmap = plt.cm.plasma
    colors = cmap(np.linspace(0, 0.85, 256))
    cmap = LinearSegmentedColormap.from_list("trunc_plasma", colors)
    marker_styles = ['o', 'X']

    def _to_np(data):
        if data is None:
            return None
        try:
            import torch
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
        except Exception:
            pass
        return np.asarray(data)

    panel_defs = {
        "loss":  (LOSS,      "Training Loss",  "Loss"),
        "train": (ACC_TRAIN, "Train Accuracy", "Accuracy"),
        "test":  (ACC_TEST,  "Test Accuracy",  "Accuracy"),
    }
    selected_panels = [panel_defs[k] for k in show_panels if k in panel_defs]

    fig, axes = plt.subplots(1, len(selected_panels), figsize=(4.2 * len(selected_panels), 3.6))
    if len(selected_panels) == 1:
        axes = [axes]

    legend_shown = False
    use_arch_mode = False

    for ax, (data, title, ylab) in zip(axes, selected_panels):
        data_np = _to_np(data)
        if data_np is None:
            ax.set_visible(False)
            continue

        if data_np.ndim == 5:
            use_arch_mode = True
            n_arch, nb_lr, nb_iter, epochs, C = data_np.shape
        elif data_np.ndim == 4:
            n_arch, nb_lr, nb_iter, epochs, C = 1, *data_np.shape
            data_np = data_np[None, ...]
        else:
            raise ValueError("Data format incompatible.")

        mean = np.nanmean(data_np, axis=2)

        if use_arch_mode:
            norm = LogNorm(vmin=float(np.min(nb_params)), vmax=float(np.max(nb_params)))
            color_vals = cmap(norm(nb_params))
        else:
            norm = LogNorm(vmin=float(np.min(learning_rates)), vmax=float(np.max(learning_rates)))
            color_vals = cmap(norm(learning_rates))

        for a in range(n_arch):
            for lr_it, lr in enumerate(learning_rates):
                for c in range(C):
                    color = color_vals[a] if use_arch_mode else cmap(norm(lr))
                    y = mean[a, lr_it, :, c]
                    if np.all(~np.isfinite(y)):
                        continue
                    ax.scatter(
                        nb_params[a] if use_arch_mode else learning_rates[lr_it],
                        y[500],
                        color=color,
                        alpha=0.95,
                        marker=marker_styles[c % len(marker_styles)],
                        edgecolor="white",
                        linewidth=0.7,
                        s=55,
                    )

        if ep_teleport is not None:
            ax.axvline(x=ep_teleport, linestyle="--", linewidth=1.3, color="black")

        if not legend_shown and title == "Training Loss":
            style_legend = [
                Line2D([0], [0], color='k', marker='o', markersize=7, label=r'NT Path Cond GD ($Ours$)', linewidth=0),
                Line2D([0], [0], color='k', marker='X', markersize=7, label='Baseline GD', linewidth=0),
            ]
            ax.legend(handles=style_legend, loc="lower left", frameon=False)
            legend_shown = True

        ax.set_xscale("log")
        ax.set_xlabel("Number of parameters" if use_arch_mode else "Learning rate")
        ax.set_title(title)
        ax.set_ylabel(ylab)
        if ylab == "Accuracy":
            ax.set_ylim(top=1.0)
            ax.set_xlim(left=0.0)
        else:
            ax.set_xlim(left=0.0)

    vmin, vmax = (float(np.min(nb_params)), float(np.max(nb_params))) if use_arch_mode else (float(np.min(learning_rates)), float(np.max(learning_rates)))
    norm = LogNorm(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=[a for a in axes if a.get_visible()],
        orientation="horizontal",
        fraction=0.05,
        pad=0.5,
        aspect=35,
        anchor=(0.5, 0.0)
    )

    log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
    cbar.set_ticks(log_ticks)
    cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks])
    cbar.ax.tick_params(labelsize=10)

    cbar.set_label("Number of parameters" if use_arch_mode else "Learning rate", fontsize=10)

    fig.subplots_adjust(bottom=0.3, wspace=0.32)
    fig.savefig(save_path)
    plt.close(fig)
    return save_path




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



def plot_boxplots(
    results,                 # torch.Tensor or np.ndarray: (n_lrs, n_runs, n_epochs, n_methods)
    mood: str = "loss",      # "loss" or "accuracy"
    n=None,
    lrs_subset=None,         # indices OU valeurs si lr_values est fourni
    box_width=0.65,
    group_gap=1.3,
    save_path="results/boxplots.pdf",
    *,
    lr_values=None,          # liste de valeurs de LR (len == n_lrs) ; sinon indices 0..n_lrs-1
    last_k=1,               # nb d’epochs pour la moyenne finale
    method_names=None,       # ex: ["sgd","adam","diag_up_sgd","diag_up_adam"]
    method_order=None,       # ordre explicite des méthodes (liste de NOMS)
    method_renames=None,     # dict {"old_name": "Pretty Name"}
    ylim=None,
    rotate_xticks=0,
    dpi=300,
    transparent=True,
    fontsize=11
):
    """
    results: tensor (n_lrs, n_runs, n_epochs, n_methods)
    - Pour chaque lr et méthode, on récupère un vecteur (n_runs,) de 'final losses'
      = moyenne sur les last_k derniers epochs par run.

    lrs_subset: si lr_values est fourni -> sous-ensemble de VALEURS; sinon -> sous-ensemble d'indices.
    """

    # --------- Normalisation d'entrée ---------
    if isinstance(results, torch.Tensor):
        R = results.detach().cpu().numpy()
    else:
        R = np.asarray(results)

    if R.ndim != 4:
        raise ValueError(f"`results` doit être (n_lrs, n_runs, n_epochs, n_methods), reçu {R.shape}")

    n_lrs, n_runs, n_epochs, n_methods = R.shape
    k = int(min(last_k, max(1, n_epochs)))  # borne

    # Méthodes: noms par défaut
    if method_names is None:
        method_names = [f"method_{i}" for i in range(n_methods)]
    else:
        if len(method_names) != n_methods:
            raise ValueError("`method_names` doit avoir la taille n_methods = results.shape[-1].")

    # Ordre des méthodes (sur les NOMS)
    if method_order is None:
        methods = list(method_names)
    else:
        # on garde uniquement celles présentes
        methods = [m for m in method_order if m in method_names]
        # et on ajoute les restantes pour éviter d'en perdre
        methods += [m for m in method_names if m not in methods]

    # LRs: valeurs affichées
    if lr_values is None:
        lr_vals = np.arange(n_lrs, dtype=float)
    else:
        lr_vals = np.asarray(lr_values, dtype=float)
        if lr_vals.shape[0] != n_lrs:
            raise ValueError("`lr_values` doit avoir la longueur n_lrs = results.shape[0].")

    # Sélection des LRs
    if lrs_subset is not None:
        subset = np.array(list(lrs_subset))
        if lr_values is None:
            # sous-ensemble d'indices
            lr_idx_all = np.unique(subset.astype(int))
        else:
            # sous-ensemble de VALEURS -> map vers indices
            idx_map = {v: i for i, v in enumerate(lr_vals)}
            try:
                lr_idx_all = np.array([idx_map[float(v)] for v in subset], dtype=int)
            except KeyError as e:
                raise ValueError(f"Valeur de LR non trouvée dans `lr_values`: {e}")
    elif n is not None:
        # n premiers (dans l'ordre croissant des valeurs de LR)
        lr_idx_all = np.argsort(lr_vals)[:int(n)]
    else:
        lr_idx_all = np.arange(n_lrs, dtype=int)

    # Renommage légende
    method_renames = method_renames or {}
    legend_names = [method_renames.get(m, m) for m in methods]

    # --------- Construire data & positions ---------
    positions, data, method_indices = [], [], []
    intra_step = 1.0  # écart intra-groupe (entre méthodes)
    M = len(methods)

    # map nom->index colonne méthode dans results
    name_to_col = {name: idx for idx, name in enumerate(method_names)}

    for i, lr_idx in enumerate(lr_idx_all):
        group_start = i * (intra_step * M + group_gap)
        # pour chaque méthode dans l'ordre demandé
        for j, m_name in enumerate(methods):
            if m_name not in name_to_col:
                continue
            m_col = name_to_col[m_name]

            # (n_runs, n_epochs) -> moyenne sur les k derniers epochs
            runs_series = R[lr_idx, :, :, m_col]  # shape (n_runs, n_epochs)
            tail = runs_series[:, -k:]            # (n_runs, k)
            vals = np.nanmean(tail, axis=1)       # (n_runs,)

            # filtrer les +/- inf / NaN
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue

            positions.append(group_start + j * intra_step)
            data.append(vals)
            method_indices.append(j)

    if not data:
        raise ValueError("Aucune donnée exploitable (vérifie lrs_subset, NaN/Inf, last_k, etc.).")

    # --------- Style & tracé ---------
    # petit contexte sans dépendre d'objets externes (neurips_rc)
    _ctx = _nullctx()

    with _ctx:
        fig = plt.figure(figsize=(12, 6))
        ax = plt.gca()

        bp = ax.boxplot(
            data,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            showfliers=True
        )

        # Palette stable (tab10)
        colors = plt.cm.tab10(np.linspace(0, 1, max(10, M)))[:M]

        for i_box, box in enumerate(bp['boxes']):
            m_idx = method_indices[i_box]
            box.set_facecolor(colors[m_idx])
            box.set_alpha(0.85)
            box.set_linewidth(0.9)

        for med in bp['medians']:
            med.set_linewidth(1.6)

        for whisk in bp['whiskers']:
            whisk.set_linewidth(0.9)
        for cap in bp['caps']:
            cap.set_linewidth(0.9)

        for fl in bp.get('fliers', []):
            fl.set_markersize(2.5)
            fl.set_alpha(0.6)

        # Grille Y + log
        ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)
        # ax.set_yscale('log')
        if ylim is not None:
            ax.set_ylim(*ylim)

        # Ticks X: un par groupe de LR
        xticks = [
            i * (intra_step * M + group_gap) + intra_step * (M - 1) / 2
            for i in range(len(lr_idx_all))
        ]
        lr_labels = lr_vals[lr_idx_all]
        xtick_labels = [
            (f"{v:.0e}" if v < 1e-2 else f"{v:.3f}".rstrip('0').rstrip('.'))
            for v in lr_labels
        ]
        ax.set_xticks(xticks, xtick_labels)
        if rotate_xticks:
            plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha='right')

        ax.set_xlabel("Learning rate")
        ax.set_ylabel(f"Final {'loss' if mood=='loss' else 'accuracy'} (mean over last "f"{k} epochs)")

        # Légende (ordre méthodes + couleurs cohérentes)
        legend_handles = []
        for name, c in zip(legend_names, colors[:M]):
            disp = name
            if disp == 1:
                disp = "Baseline (no rescale)"
            if disp == "diag_up_sgd":
                disp = r"Path Dyn.$\mathbf{(Ours)}$"
            if disp == "diag_up_adam":
                disp = r"Path Dyn. Adam$\mathbf{(Ours)}$"
            h, = ax.plot([], [], linewidth=8, color=c, label=disp)
            legend_handles.append(h)
        ax.legend(handles=legend_handles, title="Methods", ncol=min(4, M))

        # fig.tight_layout()


        # Sauvegarde
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ext = os.path.splitext(save_path)[1].lower()
            if ext in [".png", ".jpg", ".jpeg"]:
                fig.savefig(save_path, dpi=dpi, transparent=transparent, bbox_inches="tight")
            else:
                fig.savefig(save_path, transparent=transparent, bbox_inches="tight")

        plt.show()



# def plot_boxplots_ax(
#     ax,
#     results,                 # (n_lrs, n_runs, n_epochs, n_methods)
#     mood: str = "loss",      # "loss" ou "accuracy"
#     n=None,
#     lrs_subset=None,
#     box_width=0.65,
#     group_gap=1.3,
#     *,
#     lr_values=None,          # valeurs des LRs; sinon indices 0..n_lrs-1
#     last_k=1,
#     method_names=None,       # ex: ["sgd","adam","diag_up_sgd","diag_up_adam"]
#     method_order=None,       # ordre explicite (liste de NOMS)
#     method_renames=None,     # {"old": "Pretty"}
#     ylim=None,
#     rotate_xticks=0,
#     fontsize=11,
#     colors=None,             # optionnel: palette commune pour tous les panels
#     showfliers=True,
#     yscale='linear',  # ou 'log'
#     showfliers=True,
#     yscale='linear'  # ou 'log'
# ):
#     """Trace un boxplot groupé sur l'Axes `ax`.
#     Retourne un dict avec infos utiles (limites y, handles de légende, etc.)."""

#     # --------- Normalisation d'entrée ---------
#     if isinstance(results, torch.Tensor):
#         R = results.detach().cpu().numpy()
#     else:
#         R = np.asarray(results)

#     if R.ndim != 4:
#         raise ValueError(f"`results` doit être (n_lrs, n_runs, n_epochs, n_methods), reçu {R.shape}")

#     n_lrs, n_runs, n_epochs, n_methods = R.shape
#     k = int(min(last_k, max(1, n_epochs)))  # borne

#     # Méthodes: noms par défaut
#     if method_names is None:
#         method_names = [f"method_{i}" for i in range(n_methods)]
#     else:
#         if len(method_names) != n_methods:
#             raise ValueError("`method_names` doit avoir la taille n_methods = results.shape[-1].")

#     # Ordre des méthodes
#     if method_order is None:
#         methods = list(method_names)
#     else:
#         methods = [m for m in method_order if m in method_names]
#         methods += [m for m in method_names if m not in methods]

#     # LRs: valeurs affichées
#     if lr_values is None:
#         lr_vals = np.arange(n_lrs, dtype=float)
#     else:
#         lr_vals = np.asarray(lr_values, dtype=float)
#         if lr_vals.shape[0] != n_lrs:
#             raise ValueError("`lr_values` doit avoir longueur n_lrs = results.shape[0].")

#     # Sélection des LRs
#     if lrs_subset is not None:
#         subset = np.array(list(lrs_subset))
#         if lr_values is None:
#             lr_idx_all = np.unique(subset.astype(int))
#         else:
#             idx_map = {float(v): i for i, v in enumerate(lr_vals)}
#             try:
#                 lr_idx_all = np.array([idx_map[float(v)] for v in subset], dtype=int)
#             except KeyError as e:
#                 raise ValueError(f"Valeur de LR non trouvée dans `lr_values`: {e}")
#     elif n is not None:
#         lr_idx_all = np.argsort(lr_vals)[:int(n)]
#     else:
#         lr_idx_all = np.arange(n_lrs, dtype=int)

#     # Renommage légende
#     method_renames = method_renames or {}
#     legend_names = [method_renames.get(m, m) for m in methods]
#     for i, name in enumerate(legend_names):
#         if name == 1:
#             legend_names[i] = "Baseline (no rescale)"
#         if name == "diag_up_sgd":
#             legend_names[i] = r"Path Dyn.$\mathbf{(Ours)}$"
#         if name == "diag_up_adam":
#             legend_names[i] = r"Path Dyn. Adam$\mathbf{(Ours)}$"
#         if name == "baseline":
#             legend_names[i] = "Baseline (no rescale)"
#         if name == "pathcond":
#             legend_names[i] = r"Path Dyn.$\mathbf{(Ours)}$"
#         if name == "equinorm":
#             legend_names[i] = r"EquiNorm"
#         if name=="extreme":
#             legend_names[i] = r"$\lambda \to 0 (\mathbf{Ours})$"
#         if name == "baseline":
#             legend_names[i] = "Baseline (no rescale)"
#         if name == "pathcond":
#             legend_names[i] = r"Path Dyn.$\mathbf{(Ours)}$"
#         if name == "equinorm":
#             legend_names[i] = r"EquiNorm"
#         if name=="extreme":
#             legend_names[i] = r"$\lambda \to 0 (\mathbf{Ours})$"

#     # --------- Construire data & positions ---------
#     positions, data, method_indices = [], [], []
#     intra_step = 1.0  # écart intra-groupe (entre méthodes)
#     M = len(methods)

#     name_to_col = {name: idx for idx, name in enumerate(method_names)}

#     for i, lr_idx in enumerate(lr_idx_all):
#         group_start = i * (intra_step * M + group_gap)
#         for j, m_name in enumerate(methods):
#             if m_name not in name_to_col:
#                 continue
#             m_col = name_to_col[m_name]

#             runs_series = R[lr_idx, :, :, m_col]  # (n_runs, n_epochs)
#             tail = runs_series[:, -k:]            # (n_runs, k)
#             vals = np.nanmean(tail, axis=1)       # (n_runs,)

#             vals = np.asarray(vals, dtype=float)
#             vals = vals[np.isfinite(vals)]
#             if vals.size == 0:
#                 continue

#             positions.append(group_start + j * intra_step)
#             data.append(vals)
#             method_indices.append(j)

#     if not data:
#         raise ValueError("Aucune donnée exploitable (vérifie lrs_subset, NaN/Inf, last_k, etc.).")

#     # --------- Tracé ---------
#     bp = ax.boxplot(
#         data,
#         positions=positions,
#         widths=box_width,
#         patch_artist=True,
#         showfliers=showfliers
#     )

#     # Palette stable (tab10) — commune entre panels si fournie
#     if colors is None:
#         colors = plt.cm.tab10(np.linspace(0, 1, max(10, M)))[:M]

#     for i_box, box in enumerate(bp['boxes']):
#         m_idx = method_indices[i_box]
#         box.set_facecolor(colors[m_idx])
#         box.set_alpha(0.85)
#         box.set_linewidth(0.9)

#     for med in bp['medians']:
#         med.set_linewidth(1.6)
#     for whisk in bp['whiskers']:
#         whisk.set_linewidth(0.9)
#     for cap in bp['caps']:
#         cap.set_linewidth(0.9)
#     for fl in bp.get('fliers', []):
#         fl.set_markersize(2.5)
#         fl.set_alpha(0.6)

#     # Grille & axes
#     ax.grid(axis='y', linestyle='--', linewidth=0.6, alpha=0.4)
#     if ylim is not None:
#         ax.set_ylim(*ylim)

#     # Ticks X (un par groupe de LR)
#     xticks = [i*(intra_step*M + group_gap) + intra_step*(M-1)/2 for i in range(len(lr_idx_all))]
#     lr_labels = lr_vals[lr_idx_all]
#     xtick_labels = [(f"{v:.0e}" if v < 1e-2 else f"{v:.3f}".rstrip('0').rstrip('.')) for v in lr_labels]
#     ax.set_xticks(xticks, xtick_labels)
#     if rotate_xticks:
#         plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha='right')

#     ax.set_xlabel("Learning rate", fontsize=fontsize)
#     ax.set_ylabel(f"Final {'loss' if mood=='loss' else 'accuracy'} (mean over last {k} epochs)", fontsize=fontsize)
#     if yscale in ('linear', 'log'):
#         ax.set_yscale(yscale)
#     if yscale in ('linear', 'log'):
#         ax.set_yscale(yscale)

#     # Légende (mêmes couleurs que boxes)
#     legend_handles = []
#     for m_idx, name in enumerate(legend_names):
#         disp = name
#         if disp == 1:
#             disp = "Baseline (no rescale)"
#         if disp == "diag_up_sgd":
#             disp = r"Path Dyn.$\mathbf{(Ours)}$"
#         h, = ax.plot([], [], linewidth=8, color=colors[m_idx], label=disp)
#         legend_handles.append(h)

#     return {
#         "handles": legend_handles,
#         "labels": legend_names,
#         "ylim": ax.get_ylim(),
#         "colors": colors,
#     }

# ---------------------------------------------
# 2) Composite 2×2 : balanced/unbalanced × loss/accuracy
# ---------------------------------------------
# def plot_boxplots_2x2(
#     LOSS_bal, ACC_bal,
#     LOSS_bal, ACC_bal,
#     *,
#     method_names,
#     method_order=None,
#     method_renames=None,
#     lr_values=None,
#     last_k=1,
#     lrs_subset=None,
#     figsize=(18, 5),            # ← (1) plus large, moins haut
#     figsize=(18, 5),            # ← (1) plus large, moins haut
#     share_ylim_loss=True,
#     share_ylim_acc=True,
#     rotate_xticks=30,           # ← (2) rotation par défaut
#     rotate_xticks=30,           # ← (2) rotation par défaut
#     out_pdf="images/boxplots_moons_2x2.pdf",
#     out_png="images/boxplots_moons_2x2.png",
#     dpi=300,
#     transparent=True,
#     patience=100, rel_tol=0.0, abs_tol=0.1, min_epoch=0, final_k=10,
#     transparent=True,
#     patience=100, rel_tol=0.0, abs_tol=0.1, min_epoch=0, final_k=10,
# ):
#     n_methods = LOSS_bal.shape[-1]
#     colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_methods)))[:n_methods]

#     # Constrained layout OK, mais on garde la main sur les marges
#     fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)
#     # Constrained layout OK, mais on garde la main sur les marges
#     fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=False)

#     info00 = plot_boxplots_ax(
#         axes[0], LOSS_bal, mood="loss",
#         axes[0], LOSS_bal, mood="loss",
#         lr_values=lr_values, last_k=last_k, lrs_subset=lrs_subset,
#         method_names=method_names, method_order=method_order, method_renames=method_renames,
#         rotate_xticks=rotate_xticks, colors=colors
#     )
#     axes[0].set_title("Train Loss")
#     axes[0].set_title("Train Loss")

#     plot_convergence_vs_final_boxplots_ax(
#         axes[1], LOSS_bal,
#     plot_convergence_vs_final_boxplots_ax(
#         axes[1], LOSS_bal,
#         method_names=method_names, method_order=method_order, method_renames=method_renames,
#         lr_values=lr_values, lrs_subset=lrs_subset,
#         patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#         lr_values=lr_values, lrs_subset=lrs_subset,
#         patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#         rotate_xticks=rotate_xticks, colors=colors
#     )
#     axes[1].set_title("Epochs to convergence (loss)")
#     axes[1].set_title("Epochs to convergence (loss)")

#     info01 = plot_boxplots_ax(
#         axes[2], ACC_bal, mood="accuracy",
#     info01 = plot_boxplots_ax(
#         axes[2], ACC_bal, mood="accuracy",
#         lr_values=lr_values, last_k=last_k, lrs_subset=lrs_subset,
#         method_names=method_names, method_order=method_order, method_renames=method_renames,
#         rotate_xticks=rotate_xticks, colors=colors
#     )
#     axes[2].set_title("Test Accuracy")

#     # (2) Pivoter + aligner les labels x et micro-marges
#     for ax in axes:
#         for lab in ax.get_xticklabels():
#             lab.set_rotation(rotate_xticks)
#             lab.set_horizontalalignment('center')
#         ax.margins(x=0.02)
#         ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)  # petit plus esthétique
#         ax.spines['top'].set_visible(False)                         # désépaissir
#         ax.spines['right'].set_visible(False)

#     # (4) Légende commune en dessous
#     handles = info00["handles"]
#     labels  = info00["labels"]
#     fig.legend(
#         handles, labels, title="Methods",
#         ncol=min(5, len(labels)),
#         loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False
#     )

#     # (3) Espace en bas pour ticks + légende
#     fig.subplots_adjust(bottom=0.1, wspace=0.2)

#     if out_pdf:
#         Path(os.path.dirname(out_pdf)).mkdir(parents=True, exist_ok=True)
#         fig.savefig(out_pdf, bbox_inches="tight")

#     return fig, axes



# def plot_boxplots_toy(
#     LOSS,
#     *,
#     method_names,
#     method_order=None,
#     method_renames=None,
#     lr_values=None,
#     last_k=1,
#     lrs_subset=None,
#     figsize=(18, 5),            # ← (1) plus large, moins haut
#     share_ylim_loss=True,
#     share_ylim_acc=True,
#     rotate_xticks=30,           # ← (2) rotation par défaut
#     out_pdf="images/boxplots_moons_2x2.pdf",
#     out_png="images/boxplots_moons_2x2.png",
#     dpi=300,
#     transparent=True,
#     patience=100, rel_tol=0.0, abs_tol=0.1, min_epoch=0, final_k=10,
#     yscale='linear'  # ou 'log'
# ):
#     n_methods = LOSS.shape[-1]
#     colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_methods)))[:n_methods]

#     # Constrained layout OK, mais on garde la main sur les marges
#     fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)

#     info00 = plot_boxplots_ax(
#         axes[0], LOSS, mood="loss",
#     axes[2].set_title("Test Accuracy")

#     # (2) Pivoter + aligner les labels x et micro-marges
#     for ax in axes:
#         for lab in ax.get_xticklabels():
#             lab.set_rotation(rotate_xticks)
#             lab.set_horizontalalignment('center')
#         ax.margins(x=0.02)
#         ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)  # petit plus esthétique
#         ax.spines['top'].set_visible(False)                         # désépaissir
#         ax.spines['right'].set_visible(False)

#     # (4) Légende commune en dessous
#     handles = info00["handles"]
#     labels  = info00["labels"]
#     fig.legend(
#         handles, labels, title="Methods",
#         ncol=min(5, len(labels)),
#         loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False
#     )

#     # (3) Espace en bas pour ticks + légende
#     fig.subplots_adjust(bottom=0.1, wspace=0.2)

#     if out_pdf:
#         Path(os.path.dirname(out_pdf)).mkdir(parents=True, exist_ok=True)
#         fig.savefig(out_pdf, bbox_inches="tight")

#     return fig, axes



# def plot_boxplots_toy(
#     LOSS,
#     *,
#     method_names,
#     method_order=None,
#     method_renames=None,
#     lr_values=None,
#     last_k=1,
#     lrs_subset=None,
#     figsize=(18, 5),            # ← (1) plus large, moins haut
#     share_ylim_loss=True,
#     share_ylim_acc=True,
#     rotate_xticks=30,           # ← (2) rotation par défaut
#     out_pdf="images/boxplots_moons_2x2.pdf",
#     out_png="images/boxplots_moons_2x2.png",
#     dpi=300,
#     transparent=True,
#     patience=100, rel_tol=0.0, abs_tol=0.1, min_epoch=0, final_k=10,
#     yscale='linear'  # ou 'log'
# ):
#     n_methods = LOSS.shape[-1]
#     colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_methods)))[:n_methods]

#     # Constrained layout OK, mais on garde la main sur les marges
#     fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=False)

#     info00 = plot_boxplots_ax(
#         axes[0], LOSS, mood="loss",
#         lr_values=lr_values, last_k=last_k, lrs_subset=lrs_subset,
#         method_names=method_names, method_order=method_order, method_renames=method_renames,
#         rotate_xticks=rotate_xticks, colors=colors, yscale=yscale
#     )
#     axes[0].set_title("Train Loss")

#     plot_convergence_vs_final_boxplots_ax(
#         axes[1], LOSS,
#         method_names=method_names, method_order=method_order, method_renames=method_renames,
#         lr_values=lr_values, lrs_subset=lrs_subset,
#         patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#         rotate_xticks=rotate_xticks, colors=colors, yscale=yscale
#     )
#     axes[0].set_title("Train Loss")

#     plot_convergence_vs_final_boxplots_ax(
#         axes[1], LOSS,
#         method_names=method_names, method_order=method_order, method_renames=method_renames,
#         lr_values=lr_values, lrs_subset=lrs_subset,
#         patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#         rotate_xticks=rotate_xticks, colors=colors
#     )
#     axes[1].set_title("Epochs to convergence (loss)")


#     # (2) Pivoter + aligner les labels x et micro-marges
#     for ax in axes:
#         for lab in ax.get_xticklabels():
#             lab.set_rotation(rotate_xticks)
#             lab.set_horizontalalignment('center')
#         ax.margins(x=0.02)
#         ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)  # petit plus esthétique
#         ax.spines['top'].set_visible(False)                         # désépaissir
#         ax.spines['right'].set_visible(False)
#     axes[1].set_title("Epochs to convergence (loss)")


#     # (2) Pivoter + aligner les labels x et micro-marges
#     for ax in axes:
#         for lab in ax.get_xticklabels():
#             lab.set_rotation(rotate_xticks)
#             lab.set_horizontalalignment('center')
#         ax.margins(x=0.02)
#         ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)  # petit plus esthétique
#         ax.spines['top'].set_visible(False)                         # désépaissir
#         ax.spines['right'].set_visible(False)

#     # (4) Légende commune en dessous
#     # (4) Légende commune en dessous
#     handles = info00["handles"]
#     labels  = info00["labels"]
#     fig.legend(
#         handles, labels, title="Methods",
#         ncol=min(5, len(labels)),
#         loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False
#     )

#     # (3) Espace en bas pour ticks + légende
#     fig.subplots_adjust(bottom=0.1, wspace=0.2)

#     if out_pdf:
#         Path(os.path.dirname(out_pdf)).mkdir(parents=True, exist_ok=True)
#         fig.savefig(out_pdf, bbox_inches="tight")

#     return fig, axes


# # ========= Détection convergence (définition B) pour UN run =========
# def _epoch_to_convergence_vs_final(
#     y,                      # (n_epochs,)
#     *,
#     patience=100,             # nb d'epochs consécutifs requis
#     rel_tol=0.0,           # tolérance relative vs valeur finale
#     abs_tol=0.0,            # tolérance absolue additionnelle
#     min_epoch=100,            # burn-in: ignorer les premiers epochs
#     final_k=5               # moyenne sur les final_k derniers epochs pour estimer y_final
# ):
#     """
#     Retourne (t_conv, y_final):
#       - t_conv: 1-based, premier epoch t tel que pour tout u∈[t, t+patience),
#                 |y[u] - y_final| <= max(abs_tol, rel_tol * max(1, |y_final|)).
#                 Si jamais stable -> len(y).
#       - y_final: moyenne des final_k dernières valeurs (estimation cible).
#     """
#     y = np.asarray(y, dtype=float)
#     n = y.size
#     if n == 0:
#         return 0, np.nan
#     k = int(max(1, min(final_k, n)))
#     y_final = float(np.nanmean(y[-k:]))
#     tau = max(abs_tol, rel_tol * abs(y_final))

#     pat = int(max(1, patience))
#     start = int(min_epoch)
#     last_start = max(0, n - pat)

#     for t in range(start, last_start + 1):
#         seg = y[t:t+pat]
#         if np.all(np.abs(seg - y_final) <= tau):
#             return t + 1, y_final  # 1-based
#     return n, y_final  # jamais stable avant la fin

# # ========= Stats regroupées (LR × méthode) =========
# def _convergence_vs_final_stats_per_group(
#     R, lr_idx_all, methods, name_to_col,
#     *,
#     patience=100, rel_tol=1e-2, abs_tol=1e-2, min_epoch=100, final_k=10,
#     intra_step=1.0, group_gap=1.3
# ):
#     positions, epochs_to_conv, method_indices = [], [], []
#     finals_values, box_groups = [], []
#     M = len(methods)
#     data_ptr = 0

#     for i, lr_idx in enumerate(lr_idx_all):
#         group_start = i * (intra_step * M + group_gap)
#         for j, m_name in enumerate(methods):
#             if m_name not in name_to_col:
#                 continue
#             m_col = name_to_col[m_name]

#             runs = R[lr_idx, :, :, m_col]  # (n_runs, n_epochs)
#             cur_epochs, cur_final = [], []
#             for r in range(runs.shape[0]):
#                 y = runs[r]
#                 t_conv, y_fin = _epoch_to_convergence_vs_final(
#                     y,
#                     patience=patience, rel_tol=rel_tol, abs_tol=abs_tol,
#                     min_epoch=min_epoch, final_k=final_k
#                 )
#                 cur_epochs.append(t_conv)
#                 cur_final.append(y_fin)

#             n_new = len(cur_epochs)
#             positions.append(group_start + j * intra_step)
#             epochs_to_conv.extend(cur_epochs)
#             finals_values.extend(cur_final)
#             method_indices.extend([j] * n_new)
#             box_groups.append(slice(data_ptr, data_ptr + n_new))
#             data_ptr += n_new

#     return {
#         "positions": positions,
#         "epochs_to_conv": np.asarray(epochs_to_conv, dtype=float),
#         "method_indices": np.asarray(method_indices, dtype=int),
#         "finals_values": np.asarray(finals_values, dtype=float),  # y_final par run
#         "box_groups": box_groups,
#         "intra_step": intra_step,
#         "group_gap": group_gap,
#     }

# # ========= Tracé sur UN Axes =========
# def plot_convergence_vs_final_boxplots_ax(
#     ax,
#     results,                   # torch.Tensor/np.ndarray (n_lrs, n_runs, n_epochs, n_methods)
#     *,
#     method_names=None,    # ex: ["sgd","adam","diag_up_sgd","diag_up_adam"]
#     method_order=None,
#     method_renames=None,
#     lr_values=None,
#     lrs_subset=None, n=None,
#     patience=100, rel_tol=0.0, abs_tol=0.1, min_epoch=0, final_k=10,
#     box_width=0.65, rotate_xticks=0, fontsize=11,
#     colors=None, showfliers=True,
#     show_threshold_labels=True,      # affiche θ = y_final (médiane par boîte)
#     threshold_fmt="{:.3g}",
# ):
#     # --- Normalisation ---
#     R = results.detach().cpu().numpy() if isinstance(results, torch.Tensor) else np.asarray(results)
#     if R.ndim != 4:
#         raise ValueError(f"`results` doit être (n_lrs, n_runs, n_epochs, n_methods), reçu {R.shape}")
#     n_lrs, _, _, n_methods = R.shape

#     if method_names is None:
#         method_names = [f"method_{i}" for i in range(n_methods)]
#     elif len(method_names) != n_methods:
#         raise ValueError("`method_names` doit avoir len == n_methods.")
#     name_to_col = {n: i for i, n in enumerate(method_names)}

#     if method_order is None:
#         methods = list(method_names)
#     else:
#         methods = [m for m in method_order if m in name_to_col] + [m for m in method_names if m not in (method_order or [])]

#     if lr_values is None:
#         lr_vals = np.arange(n_lrs, dtype=float)
#     else:
#         lr_vals = np.asarray(lr_values, dtype=float)
#         if lr_vals.shape[0] != n_lrs:
#             raise ValueError("`lr_values` longueur != n_lrs")

#     if lrs_subset is not None:
#         subset = np.array(list(lrs_subset))
#         if lr_values is None:
#             lr_idx_all = np.unique(subset.astype(int))
#         else:
#             idx_map = {float(v): i for i, v in enumerate(lr_vals)}
#             lr_idx_all = np.array([idx_map[float(v)] for v in subset], dtype=int)
#     elif n is not None:
#         lr_idx_all = np.argsort(lr_vals)[:int(n)]
#     else:
#         lr_idx_all = np.arange(n_lrs, dtype=int)

#     method_renames = method_renames or {}
#     legend_names = [method_renames.get(m, m) for m in methods]

#     # --- Stats de convergence vs finale ---
#     stats = _convergence_vs_final_stats_per_group(
#         R, lr_idx_all, methods, name_to_col,
#         patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k
#     )

#     # Data pour boxplot (epochs) + θ par boîte (médiane des y_final runs)
#     data_epochs, theta_medians = [], []
#     for g in stats["box_groups"]:
#         data_epochs.append(stats["epochs_to_conv"][g])
#         theta_medians.append(np.median(stats["finals_values"][g]))

#     # --- Tracé ---
#     bp = ax.boxplot(
#         data_epochs,
#         positions=stats["positions"],
#         widths=box_width,
#         patch_artist=True,
#         showfliers=showfliers
#     )
#     M = len(methods)
#     if colors is None:
#         colors = plt.cm.tab10(np.linspace(0, 1, max(10, M)))[:M]

#     for i_box, box in enumerate(bp["boxes"]):
#         m_idx = stats["method_indices"][stats["box_groups"][i_box].start]
#         box.set_facecolor(colors[m_idx]); box.set_alpha(0.85); box.set_linewidth(0.9)
#     for med in bp["medians"]: med.set_linewidth(1.6)
#     for w in bp["whiskers"]: w.set_linewidth(0.9)
#     for c in bp["caps"]: c.set_linewidth(0.9)
#     for fl in bp.get("fliers", []): fl.set_markersize(2.5); fl.set_alpha(0.6)

#     ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
#     ax.set_ylabel("Epochs to convergence (vs final)", fontsize=fontsize)
#     ax.set_xlabel("Learning rate", fontsize=fontsize)
#     ax.set_yscale("log")

#     # Ticks de groupes LR
#     intra_step, group_gap = stats["intra_step"], stats["group_gap"]
#     n_groups = len(lr_idx_all)
#     group_centers = [i*(intra_step*M + group_gap) + intra_step*(M-1)/2 for i in range(n_groups)]
#     lr_labels = lr_vals[lr_idx_all]
#     xtick_labels = [(f"{v:.0e}" if v < 1e-2 else f"{v:.3f}".rstrip('0').rstrip('.')) for v in lr_labels]
#     ax.set_xticks(group_centers, xtick_labels)
#     if rotate_xticks:
#         plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

#     # Légende méthodes
#     handles = [Line2D([0],[0], color=colors[i], linewidth=8, label=legend_names[i]) for i in range(len(legend_names))]
#     # ax.legend(handles, legend_names, title="Methods", ncol=min(4, len(legend_names)), frameon=False)

#     # --- Affichage θ (valeur finale cible) par boîte ---
#     if show_threshold_labels:
#         # Récupère les bornes Y actuelles
#         y0, y1 = ax.get_ylim()
#         y_span = y1 - y0

#         # Position du texte : en-dessous de l’axe x (plus bas que y0)
#         y_text = y0 - 0.04 * y_span   # marge plus grande
#         # ax.set_ylim(y0 - 0.20 * y_span, y1)  # <-- étend l’axe Y vers le bas pour laisser la place

#         # for pos, theta in zip(stats["positions"], theta_medians):
#         #     ax.annotate(
#         #         f"{theta:.3f}",
#         #         xy=(pos, y_text),
#         #         xycoords=("data", "data"),
#         #         ha="center", va="top",       # texte au-dessus du point y_text
#         #         fontsize=max(9, fontsize-2),
#         #         rotation=90,
#         #         color="dimgray"
#         #     )


#     return {"colors": colors, "legend_names": legend_names}

# # ========= Composite 2×2 (Balanced/Unbalanced × Loss/Acc) =========
# def plot_convergence_vs_final_boxplots_2x2(
#     LOSS_bal, ACC_bal, LOSS_unb, ACC_unb,
#     *,
#     method_names,
#     method_order=None,
#     method_renames=None,
#     lr_values=None,
#     lrs_subset=None,
#     patience=5, rel_tol=1e-2, abs_tol=1e-1, min_epoch=0, final_k=5,
#     figsize=(12, 8),
#     rotate_xticks=0,
#     share_ylim=True,
#     out_pdf="images/convergence_vs_final_2x2.pdf",
#     out_png="images/convergence_vs_final_2x2.png",
#     dpi=300, transparent=True
# ):
#     n_methods = LOSS_bal.shape[-1]
#     colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_methods)))[:n_methods]

#     fig, axes = plt.subplots(1, 1)


#     #plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

#     plot_convergence_vs_final_boxplots_ax(
#         axes, LOSS_bal,
#         method_names=method_names, method_order=method_order, method_renames=method_renames,
#         lr_values=lr_values, lrs_subset=lrs_subset,
#         patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#         rotate_xticks=rotate_xticks, colors=colors
#     )

#     # plot_convergence_vs_final_boxplots_ax(
#     #     axes[0,1], ACC_bal,
#     #     method_names=method_names, method_order=method_order, method_renames=method_renames,
#     #     lr_values=lr_values, lrs_subset=lrs_subset,
#     #     patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#     #     rotate_xticks=rotate_xticks, colors=colors
#     # ); axes[0,1].set_title("Balanced — Accuracy")

#     # plot_convergence_vs_final_boxplots_ax(
#     #     axes[1], LOSS_unb,
#     #     method_names=method_names, method_order=method_order, method_renames=method_renames,
#     #     lr_values=lr_values, lrs_subset=lrs_subset,
#     #     patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#     #     rotate_xticks=rotate_xticks, colors=colors
#     # ); axes[1].set_title("Unbalanced — Loss")

#     # plot_convergence_vs_final_boxplots_ax(
#     #     axes[1,1], ACC_unb,
#     #     method_names=method_names, method_order=method_order, method_renames=method_renames,
#     #     lr_values=lr_values, lrs_subset=lrs_subset,
#     #     patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#     #     rotate_xticks=rotate_xticks, colors=colors
#     # ); axes[1,1].set_title("Unbalanced — Accuracy")

#     # Légende commune
#     legend_labels = [str(method_renames.get(n, n) if method_renames else n) for n in method_names]
#     for i, name in enumerate(legend_labels):
#         if name == 1:
#             legend_labels[i] = "Baseline (no rescale)"
#         if name == "diag_up_sgd":
#             legend_labels[i] = r"Path Dyn.$\mathbf{(Ours)}$"
#         if name == "diag_up_adam":
#             legend_labels[i] = r"Path Dyn. Adam$\mathbf{(Ours)}$"
#     legend_handles = [Line2D([0],[0], color=colors[i], linewidth=8, label=legend_labels[i]) for i in range(len(legend_labels))]
#     fig.legend(legend_handles, legend_labels, title="Methods", ncol=min(4, len(legend_labels)), loc="upper center")

#     # Harmoniser Y si demandé
#     # if share_ylim:
#     #     ymins, ymaxs = [], []
#     #     for ax in axes.ravel():
#     #         y0, y1 = ax.get_ylim(); ymins.append(y0); ymaxs.append(y1)
#     #     common = (min(ymins), max(ymaxs))
#     #     for ax in axes.ravel():
#     #         ax.set_ylim(common)

#     # fig.suptitle(r"Convergence speed — within $\varepsilon$ of final ($\theta$ annoté)", fontsize=14)
#     fig.legend(
#         handles, labels, title="Methods",
#         ncol=min(5, len(labels)),
#         loc="upper center", bbox_to_anchor=(0.5, 0.0), frameon=False
#     )

#     # (3) Espace en bas pour ticks + légende
#     fig.subplots_adjust(bottom=0.1, wspace=0.2)

#     if out_pdf:
#         Path(os.path.dirname(out_pdf)).mkdir(parents=True, exist_ok=True)
#         fig.savefig(out_pdf, bbox_inches="tight")

#     return fig, axes


# # ========= Détection convergence (définition B) pour UN run =========
# def _epoch_to_convergence_vs_final(
#     y,                      # (n_epochs,)
#     *,
#     patience=100,             # nb d'epochs consécutifs requis
#     rel_tol=0.0,           # tolérance relative vs valeur finale
#     abs_tol=0.0,            # tolérance absolue additionnelle
#     min_epoch=100,            # burn-in: ignorer les premiers epochs
#     final_k=5               # moyenne sur les final_k derniers epochs pour estimer y_final
# ):
#     """
#     Retourne (t_conv, y_final):
#       - t_conv: 1-based, premier epoch t tel que pour tout u∈[t, t+patience),
#                 |y[u] - y_final| <= max(abs_tol, rel_tol * max(1, |y_final|)).
#                 Si jamais stable -> len(y).
#       - y_final: moyenne des final_k dernières valeurs (estimation cible).
#     """
#     y = np.asarray(y, dtype=float)
#     n = y.size
#     if n == 0:
#         return 0, np.nan
#     k = int(max(1, min(final_k, n)))
#     y_final = float(np.nanmean(y[-k:]))
#     tau = max(abs_tol, rel_tol * abs(y_final))

#     pat = int(max(1, patience))
#     start = int(min_epoch)
#     last_start = max(0, n - pat)

#     for t in range(start, last_start + 1):
#         seg = y[t:t+pat]
#         if np.all(np.abs(seg - y_final) <= tau):
#             return t + 1, y_final  # 1-based
#     return n, y_final  # jamais stable avant la fin

# # ========= Stats regroupées (LR × méthode) =========
# def _convergence_vs_final_stats_per_group(
#     R, lr_idx_all, methods, name_to_col,
#     *,
#     patience=100, rel_tol=1e-2, abs_tol=1e-2, min_epoch=100, final_k=10,
#     intra_step=1.0, group_gap=1.3
# ):
#     positions, epochs_to_conv, method_indices = [], [], []
#     finals_values, box_groups = [], []
#     M = len(methods)
#     data_ptr = 0

#     for i, lr_idx in enumerate(lr_idx_all):
#         group_start = i * (intra_step * M + group_gap)
#         for j, m_name in enumerate(methods):
#             if m_name not in name_to_col:
#                 continue
#             m_col = name_to_col[m_name]

#             runs = R[lr_idx, :, :, m_col]  # (n_runs, n_epochs)
#             cur_epochs, cur_final = [], []
#             for r in range(runs.shape[0]):
#                 y = runs[r]
#                 t_conv, y_fin = _epoch_to_convergence_vs_final(
#                     y,
#                     patience=patience, rel_tol=rel_tol, abs_tol=abs_tol,
#                     min_epoch=min_epoch, final_k=final_k
#                 )
#                 cur_epochs.append(t_conv)
#                 cur_final.append(y_fin)

#             n_new = len(cur_epochs)
#             positions.append(group_start + j * intra_step)
#             epochs_to_conv.extend(cur_epochs)
#             finals_values.extend(cur_final)
#             method_indices.extend([j] * n_new)
#             box_groups.append(slice(data_ptr, data_ptr + n_new))
#             data_ptr += n_new

#     return {
#         "positions": positions,
#         "epochs_to_conv": np.asarray(epochs_to_conv, dtype=float),
#         "method_indices": np.asarray(method_indices, dtype=int),
#         "finals_values": np.asarray(finals_values, dtype=float),  # y_final par run
#         "box_groups": box_groups,
#         "intra_step": intra_step,
#         "group_gap": group_gap,
#     }

# # ========= Tracé sur UN Axes =========
# def plot_convergence_vs_final_boxplots_ax(
#     ax,
#     results,                   # torch.Tensor/np.ndarray (n_lrs, n_runs, n_epochs, n_methods)
#     *,
#     method_names=None,    # ex: ["sgd","adam","diag_up_sgd","diag_up_adam"]
#     method_order=None,
#     method_renames=None,
#     lr_values=None,
#     lrs_subset=None, n=None,
#     patience=100, rel_tol=0.0, abs_tol=0.1, min_epoch=0, final_k=10,
#     box_width=0.65, rotate_xticks=0, fontsize=11,
#     colors=None, showfliers=True,
#     show_threshold_labels=True,      # affiche θ = y_final (médiane par boîte)
#     threshold_fmt="{:.3g}",
# ):
#     # --- Normalisation ---
#     R = results.detach().cpu().numpy() if isinstance(results, torch.Tensor) else np.asarray(results)
#     if R.ndim != 4:
#         raise ValueError(f"`results` doit être (n_lrs, n_runs, n_epochs, n_methods), reçu {R.shape}")
#     n_lrs, _, _, n_methods = R.shape

#     if method_names is None:
#         method_names = [f"method_{i}" for i in range(n_methods)]
#     elif len(method_names) != n_methods:
#         raise ValueError("`method_names` doit avoir len == n_methods.")
#     name_to_col = {n: i for i, n in enumerate(method_names)}

#     if method_order is None:
#         methods = list(method_names)
#     else:
#         methods = [m for m in method_order if m in name_to_col] + [m for m in method_names if m not in (method_order or [])]

#     if lr_values is None:
#         lr_vals = np.arange(n_lrs, dtype=float)
#     else:
#         lr_vals = np.asarray(lr_values, dtype=float)
#         if lr_vals.shape[0] != n_lrs:
#             raise ValueError("`lr_values` longueur != n_lrs")

#     if lrs_subset is not None:
#         subset = np.array(list(lrs_subset))
#         if lr_values is None:
#             lr_idx_all = np.unique(subset.astype(int))
#         else:
#             idx_map = {float(v): i for i, v in enumerate(lr_vals)}
#             lr_idx_all = np.array([idx_map[float(v)] for v in subset], dtype=int)
#     elif n is not None:
#         lr_idx_all = np.argsort(lr_vals)[:int(n)]
#     else:
#         lr_idx_all = np.arange(n_lrs, dtype=int)

#     method_renames = method_renames or {}
#     legend_names = [method_renames.get(m, m) for m in methods]

#     # --- Stats de convergence vs finale ---
#     stats = _convergence_vs_final_stats_per_group(
#         R, lr_idx_all, methods, name_to_col,
#         patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k
#     )

#     # Data pour boxplot (epochs) + θ par boîte (médiane des y_final runs)
#     data_epochs, theta_medians = [], []
#     for g in stats["box_groups"]:
#         data_epochs.append(stats["epochs_to_conv"][g])
#         theta_medians.append(np.median(stats["finals_values"][g]))

#     # --- Tracé ---
#     bp = ax.boxplot(
#         data_epochs,
#         positions=stats["positions"],
#         widths=box_width,
#         patch_artist=True,
#         showfliers=showfliers
#     )
#     M = len(methods)
#     if colors is None:
#         colors = plt.cm.tab10(np.linspace(0, 1, max(10, M)))[:M]

#     for i_box, box in enumerate(bp["boxes"]):
#         m_idx = stats["method_indices"][stats["box_groups"][i_box].start]
#         box.set_facecolor(colors[m_idx]); box.set_alpha(0.85); box.set_linewidth(0.9)
#     for med in bp["medians"]: med.set_linewidth(1.6)
#     for w in bp["whiskers"]: w.set_linewidth(0.9)
#     for c in bp["caps"]: c.set_linewidth(0.9)
#     for fl in bp.get("fliers", []): fl.set_markersize(2.5); fl.set_alpha(0.6)

#     ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.4)
#     ax.set_ylabel("Epochs to convergence (vs final)", fontsize=fontsize)
#     ax.set_xlabel("Learning rate", fontsize=fontsize)
#     ax.set_yscale("log")

#     # Ticks de groupes LR
#     intra_step, group_gap = stats["intra_step"], stats["group_gap"]
#     n_groups = len(lr_idx_all)
#     group_centers = [i*(intra_step*M + group_gap) + intra_step*(M-1)/2 for i in range(n_groups)]
#     lr_labels = lr_vals[lr_idx_all]
#     xtick_labels = [(f"{v:.0e}" if v < 1e-2 else f"{v:.3f}".rstrip('0').rstrip('.')) for v in lr_labels]
#     ax.set_xticks(group_centers, xtick_labels)
#     if rotate_xticks:
#         plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

#     # Légende méthodes
#     handles = [Line2D([0],[0], color=colors[i], linewidth=8, label=legend_names[i]) for i in range(len(legend_names))]
#     # ax.legend(handles, legend_names, title="Methods", ncol=min(4, len(legend_names)), frameon=False)

#     # --- Affichage θ (valeur finale cible) par boîte ---
#     if show_threshold_labels:
#         # Récupère les bornes Y actuelles
#         y0, y1 = ax.get_ylim()
#         y_span = y1 - y0

#         # Position du texte : en-dessous de l’axe x (plus bas que y0)
#         y_text = y0 - 0.04 * y_span   # marge plus grande
#         # ax.set_ylim(y0 - 0.20 * y_span, y1)  # <-- étend l’axe Y vers le bas pour laisser la place

#         # for pos, theta in zip(stats["positions"], theta_medians):
#         #     ax.annotate(
#         #         f"{theta:.3f}",
#         #         xy=(pos, y_text),
#         #         xycoords=("data", "data"),
#         #         ha="center", va="top",       # texte au-dessus du point y_text
#         #         fontsize=max(9, fontsize-2),
#         #         rotation=90,
#         #         color="dimgray"
#         #     )


#     return {"colors": colors, "legend_names": legend_names}

# # ========= Composite 2×2 (Balanced/Unbalanced × Loss/Acc) =========
# def plot_convergence_vs_final_boxplots_2x2(
#     LOSS_bal, ACC_bal, LOSS_unb, ACC_unb,
#     *,
#     method_names,
#     method_order=None,
#     method_renames=None,
#     lr_values=None,
#     lrs_subset=None,
#     patience=5, rel_tol=1e-2, abs_tol=1e-1, min_epoch=0, final_k=5,
#     figsize=(12, 8),
#     rotate_xticks=0,
#     share_ylim=True,
#     out_pdf="images/convergence_vs_final_2x2.pdf",
#     out_png="images/convergence_vs_final_2x2.png",
#     dpi=300, transparent=True
# ):
#     n_methods = LOSS_bal.shape[-1]
#     colors = plt.cm.tab10(np.linspace(0, 1, max(10, n_methods)))[:n_methods]

#     fig, axes = plt.subplots(1, 1)


#     #plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

#     plot_convergence_vs_final_boxplots_ax(
#         axes, LOSS_bal,
#         method_names=method_names, method_order=method_order, method_renames=method_renames,
#         lr_values=lr_values, lrs_subset=lrs_subset,
#         patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#         rotate_xticks=rotate_xticks, colors=colors
#     )

#     # plot_convergence_vs_final_boxplots_ax(
#     #     axes[0,1], ACC_bal,
#     #     method_names=method_names, method_order=method_order, method_renames=method_renames,
#     #     lr_values=lr_values, lrs_subset=lrs_subset,
#     #     patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#     #     rotate_xticks=rotate_xticks, colors=colors
#     # ); axes[0,1].set_title("Balanced — Accuracy")

#     # plot_convergence_vs_final_boxplots_ax(
#     #     axes[1], LOSS_unb,
#     #     method_names=method_names, method_order=method_order, method_renames=method_renames,
#     #     lr_values=lr_values, lrs_subset=lrs_subset,
#     #     patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#     #     rotate_xticks=rotate_xticks, colors=colors
#     # ); axes[1].set_title("Unbalanced — Loss")

#     # plot_convergence_vs_final_boxplots_ax(
#     #     axes[1,1], ACC_unb,
#     #     method_names=method_names, method_order=method_order, method_renames=method_renames,
#     #     lr_values=lr_values, lrs_subset=lrs_subset,
#     #     patience=patience, rel_tol=rel_tol, abs_tol=abs_tol, min_epoch=min_epoch, final_k=final_k,
#     #     rotate_xticks=rotate_xticks, colors=colors
#     # ); axes[1,1].set_title("Unbalanced — Accuracy")

#     # Légende commune
#     legend_labels = [str(method_renames.get(n, n) if method_renames else n) for n in method_names]
#     for i, name in enumerate(legend_labels):
#         if name == 1:
#             legend_labels[i] = "Baseline (no rescale)"
#         if name == "diag_up_sgd":
#             legend_labels[i] = r"Path Dyn.$\mathbf{(Ours)}$"
#         if name == "diag_up_adam":
#             legend_labels[i] = r"Path Dyn. Adam$\mathbf{(Ours)}$"
#     legend_handles = [Line2D([0],[0], color=colors[i], linewidth=8, label=legend_labels[i]) for i in range(len(legend_labels))]
#     fig.legend(legend_handles, legend_labels, title="Methods", ncol=min(4, len(legend_labels)), loc="upper center")

#     # Harmoniser Y si demandé
#     # if share_ylim:
#     #     ymins, ymaxs = [], []
#     #     for ax in axes.ravel():
#     #         y0, y1 = ax.get_ylim(); ymins.append(y0); ymaxs.append(y1)
#     #     common = (min(ymins), max(ymaxs))
#     #     for ax in axes.ravel():
#     #         ax.set_ylim(common)

#     # fig.suptitle(r"Convergence speed — within $\varepsilon$ of final ($\theta$ annoté)", fontsize=14)

#     # Sauvegardes
#     Path(os.path.dirname(out_pdf)).mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_pdf, transparent=transparent, bbox_inches="tight")
#     Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_png, dpi=dpi, transparent=transparent, bbox_inches="tight")
#     Path(os.path.dirname(out_pdf)).mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_pdf, transparent=transparent, bbox_inches="tight")
#     Path(os.path.dirname(out_png)).mkdir(parents=True, exist_ok=True)
#     fig.savefig(out_png, dpi=dpi, transparent=transparent, bbox_inches="tight")

#     return fig, axes



# def plot_grad_stats(
#     GRAD: torch.Tensor,
#     lr_values=None,                   # (optionnel; supersédé si learning_rates est fourni)
#     figsize=(13, 5),
#     title="Évolution des statistiques de gradients",
#     dpi=160,
#     alpha_band=0.2,
#     eps=1e-12,
#     savepath=None,
#     learning_rates=None               # ← ta source de vérité pour les couleurs/étiquettes
# ):
#     """
#     Trace deux sous-figures:
#       1) Similarité cosinus(g1, g2) vs epochs (moyenne ± std sur les runs)
#       2) ||g2|| * lr / ||g1|| vs epochs (moyenne ± std sur les runs)
#          (ordre conforme au code que tu as donné : g1 = ...0, g2 = ...1)

#     Paramètres
#     ----------
#     GRAD : torch.Tensor
#         Tensor de forme (nb_lr, nb_iter, epochs, 2, nb_params),
#         où GRAD[..., 0, :] = g1 et GRAD[..., 1, :] = g2.
#     learning_rates : array-like
#         Valeurs numériques des LR (longueur = nb_lr). Utilisées pour les couleurs et le scaling LogNorm.
#         Si None, on tente lr_values; sinon on fallback à des index 0..nb_lr-1 (non log-scalable).
#     """
#     assert GRAD.ndim == 5, "GRAD doit être de forme (nb_lr, nb_iter, epochs, 2, nb_params)"
#     nb_lr, nb_iter, epochs, two, nb_params = GRAD.shape
#     assert two == 2, "La 4e dimension doit être de taille 2 (g1, g2)"

#     # --- Prépare les LR (étiquettes + valeurs numériques pour la colormap) ---
#     # 1) valeurs numériques (pour couleurs + ratio scaling)
#     if learning_rates is not None:
#         lr_numeric = np.asarray(learning_rates, dtype=float)
#     elif lr_values is not None:
#         lr_numeric = np.asarray(lr_values, dtype=float)
#     else:
#         # fallback propre (utile si on veut voir qqch même sans LRs)
#         lr_numeric = np.arange(nb_lr, dtype=float) + 1.0

#     assert len(lr_numeric) == nb_lr, "learning_rates doit avoir longueur nb_lr"


#     # --- Mise en dtype/CPU pour stabilité ---
#     GRAD_cpu = GRAD.detach().to("cpu", dtype=torch.float64)

#     g1 = GRAD_cpu[:, :, :, 0, :]  # (nb_lr, nb_iter, epochs, nb_params)  -- pathcond
#     g2 = GRAD_cpu[:, :, :, 1, :]  # (nb_lr, nb_iter, epochs, nb_params)  -- SGD (baseline)

#     # Similarité cosinus (nb_lr, nb_iter, epochs)
#     cos_all = F.cosine_similarity(g1, g2, dim=-1, eps=eps)

#     # Normes (nb_lr, nb_iter, epochs)
#     n1 = torch.linalg.vector_norm(g1, dim=-1)
#     n2 = torch.linalg.vector_norm(g2, dim=-1)

#     # Ratio conforme à ton dernier code: (||g2|| * lr) / (||g1||)
#     lr_tensor = torch.tensor(lr_numeric, dtype=g1.dtype).view(nb_lr, 1, 1)
#     ratio_all = (n2 * lr_tensor) / (n1 + eps)

#     # Moyenne/écart-type sur les runs
#     cos_mean = cos_all.nanmean(dim=1)                # (nb_lr, epochs)
#     cos_std  = cos_all.std(dim=1, unbiased=False)
#     ratio_mean = ratio_all.nanmean(dim=1)
#     ratio_std  = ratio_all.std(dim=1, unbiased=False)

#     # to numpy
#     cos_mean  = cos_mean.numpy()
#     cos_std   = cos_std.numpy()
#     ratio_mean = ratio_mean.numpy()
#     ratio_std  = ratio_std.numpy()

#     x = np.arange(1, epochs+1)

#     # --- Style conf-ready ---
#     plt.rcParams.update({
#         "figure.dpi": dpi,
#         "savefig.dpi": dpi,
#         "font.size": 12,
#         "axes.titlesize": 13,
#         "axes.labelsize": 12,
#         "legend.fontsize": 10,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "axes.grid": True,
#         "grid.linestyle": "--",
#         "grid.alpha": 0.3,
#         "lines.linewidth": 2.0,
#         "lines.markersize": 4.0,
#     })

#     fig, axes = plt.subplots(1, 2, figsize=figsize, constrained_layout=True)

#     # --- Colormap & normalisation log ---
#     cmap = plt.cm.plasma
#     colors = cmap(np.linspace(0, 0.85, 256))  # 0 → 0.85 garde jusqu'à avant le jaune
#     cmap = LinearSegmentedColormap.from_list("trunc_plasma", colors)
#     norm = LogNorm(vmin=float(np.min(learning_rates)), vmax=float(np.max(learning_rates)))

#     # --- Sous-figure 1 : cosine similarity ---
#     ax = axes[0]
#     for i in range(nb_lr):
#         color = cmap(norm(float(lr_numeric[i])))
#         m = cos_mean[i]
#         s = cos_std[i]
#         ax.plot(x, m, color=color)
#         ax.fill_between(x, m - s, m + s, alpha=alpha_band, edgecolor="none", facecolor=color)
#     ax.set_title(r"Cosine similarity between $\nabla L(\theta)$ and $\nabla L(D_\lambda \theta)$")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel("Cosine similarity")
#     ax.set_ylim(-1.05, 1.05)
#     ax.set_xlim(0, epochs-1)

#     # --- Sous-figure 2 : ratio ||g2|| * lr / ||g1|| ---
#     ax = axes[1]
#     for i in range(nb_lr):
#         color = cmap(norm(float(lr_numeric[i])))
#         m = ratio_mean[i]
#         s = ratio_std[i]
#         ax.plot(x, m,  color=color)
#         ax.fill_between(x, m - s, m + s, alpha=alpha_band, edgecolor="none", facecolor=color)
#         ax.axhline(learning_rates[i], color=color, linestyle="--", alpha=0.5, linewidth=1.2)
#     ax.set_title(r"Theoretical rescaled learning rate")
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel(r"$\|\nabla L( \theta)\|\cdot \eta \,/\, \|\nabla L(D_\lambda \theta)\|$")
#     ax.set_yscale("log")
#     ax.set_ylim(1e-5, 1e1)
#     ax.set_xlim(0, epochs-1)

#     # --- Colorbar horizontale commune ---
#     sm = ScalarMappable(norm=norm, cmap=cmap)
#     sm.set_array([])

#     # positions log-spaced pour les ticks
#     vmin, vmax = np.min(learning_rates), np.max(learning_rates)
#     log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)

#     cbar = fig.colorbar(
#         sm,
#         ax=[a for a in axes if a.get_visible()],
#         orientation="horizontal",
#         fraction=0.05,
#         pad=0.10,
#         aspect=30
#     )
#     cbar.set_label("Learning rate", fontsize=9)
#     cbar.ax.tick_params(labelsize=9)
#     cbar.set_ticks(log_ticks)
#     cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks])

#     fig.suptitle(title, y=1.04)

#     if savepath is not None:
#         fig.savefig(savepath, bbox_inches="tight")

#     return fig, axes




# def plot_grad_path_length(
#     GRAD: torch.Tensor,
#     lr_values=None,                   # (optionnel; supersédé si learning_rates est fourni)
#     figsize=(7.5, 5),
#     title="Path length of gradients",
#     dpi=160,
#     alpha_band=0.2,
#     eps=1e-12,
#     savepath=None,
#     learning_rates=None,              # ← source de vérité pour couleurs/étiquettes
#     diag_g=None,
# ):
#     """
#     Pour chaque LR, trace la somme cumulée sur les epochs :
#         PathLen_g1(e) = sum_{t=0..e} lr * ||∇L(D_λ θ_t)||
#         PathLen_g2(e) = sum_{t=0..e} lr * ||∇L(θ_t)||

#     Moyenne ± écart-type sur les runs (dimension nb_iter).
#     g1 = GRAD[..., 0, :], g2 = GRAD[..., 1, :].
#     """

#     assert GRAD.ndim == 5, "GRAD doit être de forme (nb_lr, nb_iter, epochs, 2, nb_params)"
#     nb_lr, nb_iter, epochs, two, nb_params = GRAD.shape
#     assert two == 2, "La 4e dimension doit être 2 (g1, g2)"

#     # --- Learning rates numériques pour couleurs et facteur de somme ---
#     if learning_rates is not None:
#         lr_numeric = np.asarray(learning_rates, dtype=float)
#     elif lr_values is not None:
#         lr_numeric = np.asarray(lr_values, dtype=float)
#     else:
#         lr_numeric = np.arange(nb_lr, dtype=float) + 1.0  # fallback
#     assert len(lr_numeric) == nb_lr, "learning_rates doit avoir longueur nb_lr"

#     # Labels éventuels (pas affichés en légende car on utilise une colorbar)
#     if lr_values is None:
#         lr_labels = [f"lr={v:g}" for v in lr_numeric]
#     else:
#         lr_labels = [str(v) for v in (learning_rates if learning_rates is not None else lr_values)]

#     # --- CPU & dtype stable ---
#     GRAD_cpu = GRAD.detach().to("cpu", dtype=torch.float64)

#     g1 = GRAD_cpu[:, :, :, 0, :]  # (nb_lr, nb_iter, epochs, nb_params)
#     g2 = GRAD_cpu[:, :, :, 1, :]

#     if diag_g is not None:
#         diag_g1 = diag_g.detach().to("cpu", dtype=torch.float64)[:, :, :, 0, :]
#         diag_g2 = diag_g.detach().to("cpu", dtype=torch.float64)[:, :, :, 1, :]
#         root_diag_g1 = torch.sqrt(diag_g1)
#         root_diag_g2 = torch.sqrt(diag_g2)


#     # Normes par epoch/run
#     if diag_g is  None:
#         n1 = torch.linalg.vector_norm(g1, dim=-1)  # (nb_lr, nb_iter, epochs)
#         n2 = torch.linalg.vector_norm(g2, dim=-1)
#     else:
#         n1 = torch.linalg.vector_norm(g1 * root_diag_g1, dim=-1)
#         n2 = torch.linalg.vector_norm(g2 * root_diag_g2, dim=-1)
#     # Incréments lr * ||grad|| puis cumul sur epochs (incl. t=0..e)
#     lr_tensor = torch.tensor(lr_numeric, dtype=n1.dtype).view(nb_lr, 1, 1)
#     incr1 = lr_tensor * n1
#     incr2 = lr_tensor * n2
#     path1_all = torch.cumsum(incr1, dim=2)  # (nb_lr, nb_iter, epochs)
#     path2_all = torch.cumsum(incr2, dim=2)

#     # --- Moyenne & std (compatible anciennes versions, gère NaN) ---
#     def mean_and_std(x, dim):
#         mask = ~torch.isnan(x)
#         cnt = torch.clamp(mask.sum(dim=dim), min=1)
#         x_filled = torch.where(mask, x, torch.zeros_like(x))
#         mean = x_filled.sum(dim=dim) / cnt
#         var = ((torch.where(mask, x, torch.zeros_like(x)) - mean.unsqueeze(dim))**2).sum(dim=dim) / cnt
#         return mean, torch.sqrt(var)

#     path1_mean, path1_std = mean_and_std(path1_all, dim=1)  # (nb_lr, epochs)
#     path2_mean, path2_std = mean_and_std(path2_all, dim=1)

#     path1_mean = path1_mean.numpy(); path1_std = path1_std.numpy()
#     path2_mean = path2_mean.numpy(); path2_std = path2_std.numpy()
#     x = np.arange(1, epochs+1)

#     # --- Style conf-ready ---
#     plt.rcParams.update({
#         "figure.dpi": dpi,
#         "savefig.dpi": dpi,
#         "font.size": 12,
#         "axes.titlesize": 13,
#         "axes.labelsize": 12,
#         "xtick.labelsize": 10,
#         "ytick.labelsize": 10,
#         "axes.grid": True,
#         "grid.linestyle": "--",
#         "grid.alpha": 0.3,
#         "lines.linewidth": 2.0,
#         "lines.markersize": 4.0,
#     })

#     fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

#     # --- Colormap plasma tronquée (évite le jaune), normalisation log ---
#     base_cmap = plt.cm.plasma
#     colors = base_cmap(np.linspace(0.0, 0.85, 256))  # 0 → 0.85 : plus lisible
#     cmap = LinearSegmentedColormap.from_list("trunc_plasma", colors)


#     vmax = float(np.max(learning_rates if learning_rates is not None else lr_numeric))
#     vmin = float(np.min(learning_rates if learning_rates is not None else lr_numeric))
#     norm = LogNorm(vmin=vmin, vmax=vmax)

#     # --- Tracé des courbes: g1 plein, g2 pointillés, bandes ±σ ---
#     for i in range(nb_lr):
#         color = cmap(norm(float(lr_numeric[i])))

#         m1, s1 = path1_mean[i], path1_std[i]
#         m2, s2 = path2_mean[i], path2_std[i]

#         # g1: plein
#         ax.plot(x, m1, color=color, linestyle='-')
#         ax.fill_between(x, m1 - s1, m1 + s1, alpha=alpha_band, color=color, edgecolor="none")

#         # g2: pointillés
#         ax.plot(x, m2, color=color, linestyle='--')
#         ax.fill_between(x, m2 - s2, m2 + s2, alpha=alpha_band*0.7, color=color, edgecolor="none")

#     # Titres/labels
#     ax.set_title(title)
#     ax.set_xlabel("Epoch")
#     ax.set_ylabel(r"Cumulative path length")
#     ax.set_yscale("log")
#     ax.set_xscale("log")

#     # Petite légende de style (ligne pleine = g1, pointillés = g2)
#     style_legend = [
#         Line2D([0], [0], color='k', linestyle='-', label=r'NT Path Cond GD ($\mathbf{Ours}$)'),
#         Line2D([0], [0], color='k', linestyle='--', label='Baseline GD')
#     ]
#     ax.legend(handles=style_legend, loc="upper left", frameon=False)

#     # --- Colorbar horizontale (ticks log, 10^k) ---
#     sm = ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
#     log_ticks = np.logspace(np.log10(vmin), np.log10(vmax), num=5)
#     cbar = fig.colorbar(
#         sm,
#         ax=ax,
#         orientation="horizontal",
#         fraction=0.05,
#         pad=0.10,
#         aspect=30
#     )
#     cbar.set_label("Learning rate", fontsize=9)
#     cbar.ax.tick_params(labelsize=9)
#     cbar.set_ticks(log_ticks)
#     cbar.set_ticklabels([f"$10^{{{int(np.log10(t))}}}$" for t in log_ticks])

#     if savepath is not None:
#         fig.savefig(savepath, bbox_inches="tight")

#     return fig, ax


# def plot_time_vs_params(
#     TIME,
#     EPOCHS,
#     nb_params,
#     method_names=("Method 1", "Method 2"),
#     invert_axes=False,
#     outdir="images/",
#     fname="scatter_time_params.pdf",
# ):
#     """
#     TIME:   tensor shape (n_arch, 1, nb_iter, 1, 2)
#     EPOCHS: tensor shape (n_arch, 1, nb_iter, 1, 2)
#     nb_params: list/array of length n_arch

#     invert_axes:
#         False → params (x), time (y)
#         True  → epoch (x), time (y)
#     """

#     import numpy as np
#     import matplotlib.pyplot as plt
#     from matplotlib.colors import Normalize, LinearSegmentedColormap
#     from pathlib import Path

#     # -----------------------------------------------------------
#     # Convert torch → numpy
#     # -----------------------------------------------------------
#     def _to_np(x):
#         try:
#             import torch
#             if isinstance(x, torch.Tensor):
#                 return x.detach().cpu().numpy()
#         except Exception:
#             pass
#         return np.asarray(x)

#     TIME = _to_np(TIME)
#     EPOCHS = _to_np(EPOCHS)
#     nb_params = _to_np(nb_params)

#     n_arch, _, nb_iter, _, _ = TIME.shape

#     # -----------------------------------------------------------
#     # Mean time over iterations
#     # -----------------------------------------------------------
#     mean_t1 = np.nanmedian(TIME[..., 0], axis=2)  # shape (n_arch)
#     mean_t2 = np.nanmedian(TIME[..., 1], axis=2)

#     # -----------------------------------------------------------
#     # Mean epoch for coloring
#     # -----------------------------------------------------------
#     mean_epoch1 = np.nanmedian(EPOCHS[..., 0], axis=2).reshape(n_arch)
#     mean_epoch2 = np.nanmedian(EPOCHS[..., 1], axis=2).reshape(n_arch)

#     # -----------------------------------------------------------
#     # Plot style NeurIPS
#     # -----------------------------------------------------------
#     plt.rcParams.update({
#         "savefig.bbox": "tight",
#         "pdf.fonttype": 42,
#         "ps.fonttype": 42,
#         "font.size": 11,
#         "axes.labelsize": 11,
#         "axes.titlesize": 11,
#         "legend.fontsize": 10,
#         "grid.alpha": 0.25,
#         "axes.grid": True,
#     })

#     cmap = plt.cm.plasma
#     colors = cmap(np.linspace(0, 0.9, 256))
#     cmap = LinearSegmentedColormap.from_list("plasma_trunc", colors)

#     # Epoch colors normalized
#     if invert_axes:
#         norm_epoch = Normalize(
#             vmin=min(mean_t1.min(), mean_t2.min()),
#             vmax=max(mean_t1.max(), mean_t2.max())
#         )
#     else:
#         norm_epoch = Normalize(
#             vmin=min(mean_epoch1.min(), mean_epoch2.min()),
#             vmax=max(mean_epoch1.max(), mean_epoch2.max())
#         )


#     Path(outdir).mkdir(parents=True, exist_ok=True)

#     fig, ax = plt.subplots(figsize=(4.2, 3.6))

#     markers = ['o', 'X']

#     def scatter_one(method_idx, mean_times, mean_epochs, marker):
#         for p, t, e in zip(nb_params, mean_times, mean_epochs):
#             if invert_axes:
#                 col = cmap(norm_epoch(t))
#                 ax.scatter(
#                     p, e,
#                     marker=marker,
#                     s=65, alpha=0.9,
#                     edgecolor="white", linewidth=0.7
#                 )
#             else:
#                 col = cmap(norm_epoch(e))
#                 ax.scatter(
#                     p, t,
#                     marker=marker,
#                     s=65, alpha=0.9,
#                     edgecolor="white", linewidth=0.7
#                 )

#     scatter_one(0, mean_t1, mean_epoch1, markers[0])
#     scatter_one(1, mean_t2, mean_epoch2, markers[1])


#     import matplotlib.lines as mlines
#     legend_elems = [
#         mlines.Line2D([], [], color='k', marker='o', linestyle='None',
#                       markersize=8, label=method_names[0]),
#         mlines.Line2D([], [], color='k', marker='X', linestyle='None',
#                       markersize=8, label=method_names[1]),
#     ]
#     ax.legend(handles=legend_elems, frameon=False, loc='best')

#     if invert_axes:
#         ax.set_xlabel("#Parameters")
#         ax.set_ylabel(r"Median Epoch")
#     else:
#         ax.set_xlabel("#Parameters")
#         ax.set_ylabel("Median Time")

#     ax.set_xscale("log")

#     outpath = Path(outdir) / fname
#     fig.savefig(outpath)
#     plt.close(fig)

#     return outpath
