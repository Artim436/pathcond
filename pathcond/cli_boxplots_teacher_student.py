import argparse
from pathcond.train import fit_with_telportation, rescaling_path_dynamics
from pathcond.plot import plot_mean_var_curves, plot_boxplots, plot_boxplots_2x2, plot_convergence_vs_final_boxplots_2x2
from pathlib import Path
from pathcond.utils import _ensure_outdir
import torch
<<<<<<< HEAD
from pathcond.plot import plot_mean_var_curves_triptych, plot_mean_var_curves_triptych_init
import numpy as np
import matplotlib.pyplot as plt


# --- 1. FONCTION DE COMPARAISON ET DE STATISTIQUES (Modifiée pour conserver les données brutes) ---
def analyze_gradient_similarity_raw(GRAD):
    """
    Calcule la similarité cosinus et la norme L2 de la différence
    entre les paires de gradients sur 'nb_iter', pour chaque 'nb_init'.

    Args:
        GRAD (torch.Tensor): Tenseur de forme (nb_init, nb_iter, 2, nb_params).

    Returns:
        dict: Contenant les données brutes de similarité cosinus et de norme L2 
              de la différence, chacune de forme (nb_init, nb_iter).
    """
    nb_init, nb_iter, _, nb_params = GRAD.shape

    G1 = GRAD[:, :, 0, :]  # Shape: (nb_init, nb_iter, nb_params)
    G2 = GRAD[:, :, 1, :]  # Shape: (nb_init, nb_iter, nb_params)

    # --- Similarité Cosinus ---
    dot_product = torch.sum(G1 * G2, dim=2)
    norm_G1 = torch.linalg.norm(G1, dim=2)
    norm_G2 = torch.linalg.norm(G2, dim=2)
    eps = 1e-8
    cosine_similarity = dot_product / (norm_G1 * norm_G2 + eps)  # Shape: (nb_init, nb_iter)
    
    # --- min de la norm sur le max ---
    norm_max = torch.max(norm_G1, norm_G2)
    norm_min = torch.min(norm_G1, norm_G2)
    ratio_min_max = norm_min / (norm_max + eps)  # Shape: (nb_init, nb_iter)

    results = {
        # Convertir en liste de NumPy arrays, où chaque élément est un vecteur des 'nb_iter' 
        # pour une initialisation donnée. Cela est nécessaire pour l'entrée de Matplotlib boxplot.
        'cosine_data_for_boxplot': [cosine_similarity[i, :].numpy() for i in range(nb_init)],
        'norm_data_for_boxplot': [ratio_min_max[i, :].numpy() for i in range(nb_init)],
        'nb_init': nb_init
    }
    
    return results

# --- 2. FONCTION DE PLOT BOXPLOT ---
def plot_boxplot_results(results):
    """
    Crée un boxplot des similarités cosinus et des normes L2 
    pour chaque initialisation.
    """
    
    nb_init = results['nb_init']
    init_labels = [f'Init {i+1}' for i in range(nb_init)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plt.style.use('seaborn-v0_8-whitegrid')

    # --- Plot 1: Similarité Cosinus (COS SIM) ---
    cos_data = results['cosine_data_for_boxplot']
    
    # Création du boxplot
    axes[0].boxplot(cos_data, 
                    labels=init_labels, 
                    patch_artist=True, # Permet de colorier les boîtes
                    boxprops=dict(facecolor='lightblue', color='blue'),
                    medianprops=dict(color='red'),
                    flierprops=dict(marker='o', markersize=4, markerfacecolor='darkred', alpha=0.5))
                         
    axes[0].axhline(y=1.0, color='gray', linestyle='--', linewidth=0.8, label=r'Similarité Parfaite')
    
    axes[0].set_title(r"Distribution de la Similarité Cosinus entre les Gradients $\mathbf{g}_1$ et $\mathbf{g}_2$", fontsize=14)
    axes[0].set_xlabel("Initialisation", fontsize=12)
    axes[0].set_ylabel("Similarité Cosinus (sur les itérations)", fontsize=12)
    axes[0].set_ylim(-1.0, 1.1) # La similarité cosinus est toujours entre -1 et 1
    axes[0].grid(axis='y', linestyle='--')
    axes[0].legend(loc='lower right')

    # --- Plot 2: Norme L2 de la Différence (NORM DIFF) ---
    norm_data = results['norm_data_for_boxplot']
    
    # Création du boxplot
    axes[1].boxplot(norm_data, 
                    labels=init_labels, 
                    patch_artist=True,
                    boxprops=dict(facecolor='lightsalmon', color='darkorange'),
                    medianprops=dict(color='red'),
                    flierprops=dict(marker='o', markersize=4, markerfacecolor='darkred', alpha=0.5))
                         
    axes[1].set_title(r"Distribution du Ratio de la Norme Min sur la Norme Max des Gradients $\mathbf{g}_1$ et $\mathbf{g}_2$", fontsize=14)
    axes[1].set_xlabel("Initialisation", fontsize=12)
    axes[1].set_ylabel("Ratio de la Norm L2 (sur les itérations)", fontsize=12)
    axes[1].grid(axis='y', linestyle='--')
    
    plt.suptitle("Comparaison de la Similarité et de la Distance des Gradients par Initialisation", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(_ensure_outdir("images/") / "gradient_trajectory_analysis.pdf")
=======
from pathcond.plot import plot_boxplots_toy


# 
>>>>>>> 3a474d523143713e08b0ccde1da67b383ab8e54e

def main():
    resdir = Path("results")
    images = Path("images"); images.mkdir(exist_ok=True, parents=True)
    imgdir = Path("images"); imgdir.mkdir(parents=True, exist_ok=True)

    LOSS = torch.load(resdir / "multi_lr_ts_loss.pt")
<<<<<<< HEAD

    GRAD_init = torch.load(resdir / "multi_lr_ts_grad.pt")
    
    nb_init = LOSS.shape[0]
    inits = torch.logspace(-2, -1, nb_init).tolist()



    save_path = plot_mean_var_curves_triptych_init(
    LOSS=LOSS,
    ACC_TRAIN=None,
    ACC_TEST=None,
    inits=inits,
    ep_teleport=None,          # ou un entier, ex: 1
    outdir=imgdir,
    fname="ts_triptych.pdf",
    show_panels="loss",  # "loss", "acc_train", "acc_test"
    
)
    print("Figure enregistrée dans images/ts_triptych.pdf")


    # --- Analyse des gradients ---
    results = analyze_gradient_similarity_raw(GRAD_init)
    plot_boxplot_results(results)

=======
    
    nb_lr = LOSS.shape[0]
    learning_rates = torch.logspace(-4, -1, nb_lr).numpy()

    method_names = ["pathcond", "baseline", "equinorm", "extreme"]

    plot_boxplots_toy(
        LOSS,
        method_names=method_names,
        method_order=None,             # ou liste explicite
        lr_values=learning_rates,
        last_k=5,
        lrs_subset=None,               # ou ex: [1e-4, 1e-3, 1e-2]
        figsize=(18, 5),
        rotate_xticks=0,
        out_pdf=str(images / "boxplots_ts.pdf"),
        out_png=str(images / "boxplots_ts.png"),
        dpi=300,
        patience=100, rel_tol=0, abs_tol=1e-4, min_epoch=100,
        yscale='log',
    )
    print("Figure enregistrée dans images/boxplots_ts.{pdf,png}")
>>>>>>> 3a474d523143713e08b0ccde1da67b383ab8e54e
