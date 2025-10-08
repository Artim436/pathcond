import argparse
from pathcond.train import fit_with_telportation, rescaling_path_dynamics
from pathcond.plot import plot_mean_var_curves, plot_boxplots, plot_boxplots_2x2, plot_convergence_vs_final_boxplots_2x2
from pathlib import Path
from pathcond.utils import _ensure_outdir
import torch
from pathcond.plot import plot_boxplots_toy


# 

def main():
    resdir = Path("results")
    images = Path("images"); images.mkdir(exist_ok=True, parents=True)
    imgdir = Path("images"); imgdir.mkdir(parents=True, exist_ok=True)

    LOSS = torch.load(resdir / "multi_lr_ts_loss.pt")
    
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
