import argparse
from .train import fit_with_telportation, rescaling_path_dynamics
from ..pathcond.plot import plot_mean_var_curves_triptych, plot_mean_var_curves_triptych_epochs_times_lr
from ..pathcond.utils import _ensure_outdir
import torch
from pathlib import Path



def main():
    resdir = Path("results/"); resdir.mkdir(exist_ok=True, parents=True)
    images = Path("images"); images.mkdir(exist_ok=True, parents=True)
    imgdir = Path("images"); imgdir.mkdir(parents=True, exist_ok=True)

    LOSS = torch.load(_ensure_outdir("results/") / "mnist_loss.pt")
    
    ACC_TRAIN = torch.load(_ensure_outdir("results/") / "mnist_acc_train.pt")
    
    ACC_TEST = torch.load(_ensure_outdir("results/") / "mnist_acc_test.pt")

    learning_rates = torch.logspace(-1, -1, LOSS.shape[0]).tolist()

    LOSS_TRUNC = LOSS[-4:]
    ACC_TRAIN_TRUNC = ACC_TRAIN[-4:]
    ACC_TEST_TRUNC = ACC_TEST[-4:]

    learning_rates_trunc = learning_rates[-4:]

    save_path = plot_mean_var_curves_triptych(
    LOSS=LOSS,
    ACC_TRAIN=ACC_TRAIN,
    ACC_TEST=ACC_TEST,
    learning_rates=learning_rates,
    ep_teleport=None,          # ou un entier, ex: 1
    outdir=imgdir,
    fname="mnist_triptych.pdf",
    
)
    print("Saved to:", save_path)

    save_path = plot_mean_var_curves_triptych_epochs_times_lr(
    LOSS=LOSS_TRUNC,
    ACC_TRAIN=ACC_TRAIN_TRUNC,
    ACC_TEST=ACC_TEST_TRUNC,
    learning_rates=learning_rates_trunc,
    ep_teleport=None,          # ou un entier, ex: 1
    outdir=imgdir,
    fname="mnist_triptych_epochs.pdf",
)