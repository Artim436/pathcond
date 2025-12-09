import argparse
import torch
import matplotlib.pyplot as plt
from .models import MNISTMLP
from pathlib import Path
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from .rescaling import compute_G_matrix, compute_diag_G, apply_neuron_rescaling_mlp, apply_neuron_rescaling_on_matrix_G


def main():
    p = argparse.ArgumentParser(description="Compare G and its diagonal")
    p.add_argument("--h1", type=int, default=5, help="Hidden size 1")
    p.add_argument("--h2", type=int, default=5, help="Hidden size 2")
    args = p.parse_args()
    compare_layerwise_avg(hidden=(args.h1, args.h2))


def compare_layerwise_avg(hidden=(5, 5)):
    """
    Compare layer-wise the effect of single-neuron rescaling on:
      - ||G - I||_F      (full path-kernel)
      - ||diag(G) - 1||  (diagonal approximation)

    For each layer and each hidden neuron, we sweep lambda in logspace,
    apply the neuron-wise rescaling, recompute the metrics, and then
    plot mean ± std across neurons, as a function of lambda.

    Saves: results/compare_g_vs_diag_g_layerwise.pdf
    """
    # ------------------ Model & constants ------------------
    model_ref = MNISTMLP(hidden[0], hidden[1], p_drop=0.0, seed=0)
    G0 = compute_G_matrix(model_ref)

    # linear layers index (Sequential)
    linear_indices = [i for i, layer in enumerate(model_ref.model) if isinstance(layer, nn.Linear)]
    if len(linear_indices) < 2:
        raise RuntimeError("Expected at least two nn.Linear layers in MNISTMLP(model_ref).")

    out_feats = [
        model_ref.model[linear_indices[0]].out_features,  # hidden[0]
        model_ref.model[linear_indices[1]].out_features,  # hidden[1]
    ]

    # Lambda sweep (logspace)
    LAMBDA = torch.logspace(-2, 2, steps=30)
    x = LAMBDA.cpu().numpy()

    # Identity & ones for norms
    I = torch.eye(G0.shape[0], dtype=G0.dtype, device=G0.device)

    # ------------------ Output dir ------------------
    outdir = "results/"
    try:
        out = _ensure_outdir(outdir)  # if your helper exists
    except NameError:
        Path(outdir).mkdir(parents=True, exist_ok=True)
        out = Path(outdir)

    # ------------------ Plot style ------------------
    plt.rcParams.update({
        "figure.figsize": (5.6, 3.8),  # single-column friendly
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.grid": True, "grid.alpha": 0.25,
        "lines.linewidth": 1.8, "lines.markersize": 3.0,
    })

    fig, ax = plt.subplots()

    # ------------------ Main loop ------------------
    for layer_idx, n_neurons in enumerate(out_feats):
        if n_neurons <= 0:
            continue

        # collect per-neuron curves over lambda
        g_norms_all = []       # shape -> (n_neurons, n_lambda)
        diag_norms_all = []    # shape -> (n_neurons, n_lambda)

        for neuron_idx in tqdm(range(n_neurons)):
            g_norm_history = []
            diag_g_norm_history = []

            for lam in LAMBDA:
                lam_val = float(lam.item())

                # (1) Rescale in G-space without rebuilding model (fast path)
                # apply_neuron_rescaling_on_matrix_G is assumed to return (.., G_rescaled)
                _, G_rescaled = apply_neuron_rescaling_on_matrix_G(
                    G_mat=G0, model=model_ref,
                    layer_idx=layer_idx, neuron_idx=neuron_idx,
                    lamda=lam_val
                )

                # (2) Rescale model weights (for diag(G)) and compute diag(G)
                m_rescaled = apply_neuron_rescaling_mlp(
                    model_ref, layer_idx=layer_idx, neuron_idx=neuron_idx, lamda=lam_val
                ).eval()
                diag_G = compute_diag_G(m_rescaled)  # 1D vector expected

                # Norms
                g_norm = torch.norm(G_rescaled - I, p="fro").item()
                diag_norm = torch.norm(diag_G - torch.ones_like(diag_G)).item()

                g_norm_history.append(g_norm)
                diag_g_norm_history.append(diag_norm)

            g_norms_all.append(g_norm_history)
            diag_norms_all.append(diag_g_norm_history)

        # to numpy: (n_neurons, n_lambda)
        g_norms_all = np.asarray(g_norms_all, dtype=float)
        diag_norms_all = np.asarray(diag_norms_all, dtype=float)

        # mean ± std over neurons
        g_mean, g_std = g_norms_all.mean(axis=0), g_norms_all.std(axis=0, ddof=0)
        d_mean, d_std = diag_norms_all.mean(axis=0), diag_norms_all.std(axis=0, ddof=0)

        # plot: solid for G, dashed for diag(G)
        lbl_full = fr"Layer {layer_idx+1}: $\|G - I\|_F$"
        lbl_diag = fr"Layer {layer_idx+1}: $\|\mathrm{{diag}}(G) - \mathbf{{1}}\|$"

        ax.plot(x, g_mean,  linestyle="-", label=lbl_full)
        ax.fill_between(x, g_mean - g_std, g_mean + g_std, alpha=0.2)

        ax.plot(x, d_mean, linestyle="--", label=lbl_diag)
        ax.fill_between(x, d_mean - d_std, d_mean + d_std, alpha=0.2)

    # ------------------ Axes & labels ------------------
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Rescaling factor $\lambda$")
    ax.set_ylabel(r"Mean $\pm$ std of norms across neurons")
    ax.set_title(r"Layer-wise comparison: $\|G - I\|_F$ vs. $\|\mathrm{diag}(G) - \mathbf{1}\|$")
    ax.legend(frameon=False, ncol=1)
    fig.tight_layout()

    # ------------------ Save ------------------
    out_file = out / "compare_g_vs_diag_g_layerwise.pdf"
    fig.savefig(out_file)
    plt.show()
    plt.close(fig)
    print(f"✅ Figure saved at {out_file}")

    return out_file



def _ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p