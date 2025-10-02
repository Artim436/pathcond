import argparse
from pathcond.train import fit_with_telportation, rescaling_path_dynamics
from pathcond.plot import plot_mean_var_curves, plot_boxplots, plot_boxplots_2x2
from pathlib import Path
from pathcond.utils import _ensure_outdir
import torch


# 

def main():
    resdir = Path("results")
    images = Path("images"); images.mkdir(exist_ok=True, parents=True)

    LOSS_bal = torch.load(resdir / "multi_lr_moons_loss.pt")
    ACC_bal  = torch.load(resdir / "multi_lr_moons_acc.pt")
    LOSS_unb = torch.load(resdir / "multi_lr_moons_loss_unbalanced.pt")
    ACC_unb  = torch.load(resdir / "multi_lr_moons_acc_unbalanced.pt")

    # Noms de méthodes comme dans ton code
    if LOSS_bal.shape[3] == 4:
        method_names = ["diag_up_sgd", 1, "diag_up_adam", "ref_adam"]
    else:
        method_names = ["diag_up_sgd", 1]

    nb_lr = LOSS_bal.shape[0]
    learning_rates = torch.logspace(-4, -1, nb_lr).numpy()

    plot_boxplots_2x2(
        LOSS_bal, ACC_bal, LOSS_unb, ACC_unb,
        method_names=method_names,
        method_order=None,             # ou liste explicite
        method_renames={1: "Baseline"},# renommages jolis si tu veux
        lr_values=learning_rates,
        last_k=1,
        lrs_subset=None,               # ou ex: [1e-4, 1e-3, 1e-2]
        figsize=(12, 8),
        rotate_xticks=0,
        out_pdf=str(images / "boxplots_moons_2x2.pdf"),
        out_png=str(images / "boxplots_moons_2x2.png"),
        dpi=300
    )
    print("Figure enregistrée dans images/boxplots_moons_2x2.{pdf,png}")



# def main():

#     LOSS, ACC = torch.load(_ensure_outdir("results/") / "multi_lr_moons_loss.pt"), torch.load(_ensure_outdir("results/") / "multi_lr_moons_acc.pt")
#     if LOSS.shape[3] == 4:
#         method_names = ["diag_up_sgd", 1, "diag_up_adam", "ref_adam"]
#     else:
#         method_names = ["diag_up_sgd", 1]
#     nb_lr = LOSS.shape[0]
#     learning_rates = torch.logspace(-4, -1, nb_lr)
#     plot_boxplots(LOSS, mood="loss", save_path="images/boxplots_loss_moons.pdf", last_k=1, method_names=method_names, lr_values=learning_rates)
#     plot_boxplots(ACC, mood="accuracy", save_path="images/boxplots_accuracy_moons.pdf", last_k=1, method_names=method_names, lr_values=learning_rates)

#     LOSS, ACC = torch.load(_ensure_outdir("results/") / "multi_lr_moons_loss_unbalanced.pt"), torch.load(_ensure_outdir("results/") / "multi_lr_moons_acc_unbalanced.pt")
#     if LOSS.shape[3] == 4:
#         method_names = ["diag_up_sgd", 1, "diag_up_adam", "ref_adam"]
#     else:
#         method_names = ["diag_up_sgd", 1]
#     nb_lr = LOSS.shape[0]
#     learning_rates = torch.logspace(-4, -1, nb_lr)  
#     plot_boxplots(LOSS, mood="loss", save_path="images/boxplots_loss_moons_unbalanced.pdf", last_k=1, method_names=method_names, lr_values=learning_rates)
#     plot_boxplots(ACC, mood="accuracy", save_path="images/boxplots_accuracy_moons_unbalanced.pdf", last_k=1, method_names=method_names, lr_values=learning_rates)