import argparse
from .train import fit_with_telportation, rescaling_path_dynamics
from .plot import plot_mean_var_curves
from .utils import _ensure_outdir
import torch


def main():
    LOSS, ACC = torch.load(_ensure_outdir("results/") / "mnist_loss.pt"), torch.load(_ensure_outdir("results/") / "mnist_acc.pt")
    plot_mean_var_curves(LOSS=LOSS, mood="loss", ep_teleport=0, outdir="results/", fname_prefix="mnist")
    plot_mean_var_curves(ACC=ACC, mood="accuracy", ep_teleport=0, outdir="results/", fname_prefix="mnist")

