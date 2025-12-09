import argparse
from .train import fit_with_telportation, rescaling_path_dynamics
from .plot import plot_mean_var_curves_triptych, plot_mean_var_curves_triptych_epochs_times_lr
from .utils import _ensure_outdir
import torch
from pathlib import Path



def main():
    resdir = Path("results/resnet/"); resdir.mkdir(exist_ok=True, parents=True)
    images = Path("images/resnet"); images.mkdir(exist_ok=True, parents=True)
    imgdir = Path("images/resnet"); imgdir.mkdir(parents=True, exist_ok=True)

    LOSS = torch.load(_ensure_outdir("results/resnet/") / "mnist_loss.pt")
    
    ACC_TRAIN = torch.load(_ensure_outdir("results/resnet/") / "mnist_acc_train.pt")
    
    ACC_TEST = torch.load(_ensure_outdir("results/resnet/") / "mnist_acc_test.pt")

    learning_rates = torch.logspace(-4, -1, LOSS.shape[0]).tolist()


    learning_rates_trunc = learning_rates[-4:]

    save_path = plot_mean_var_curves_triptych(
    LOSS=LOSS,
    ACC_TRAIN=ACC_TRAIN,
    ACC_TEST=ACC_TEST,
    learning_rates=learning_rates,
    ep_teleport=None,          # ou un entier, ex: 1
    outdir=imgdir,
    fname="resnet_mnist_triptych.pdf",
    
)
    print("Saved to:", save_path)

    save_path = plot_mean_var_curves_triptych_epochs_times_lr(
    LOSS=LOSS,
    ACC_TRAIN=ACC_TRAIN,
    ACC_TEST=ACC_TEST,
    learning_rates=learning_rates,
    ep_teleport=None,          # ou un entier, ex: 1
    outdir=imgdir,
    fname="resnet_mnist_triptych_epochs.pdf",
)