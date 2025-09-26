import argparse
from train import fit_with_telportation, rescaling_path_dynamics
from plot import plot_mean_var_curves
from utils import _ensure_outdir
import torch


def main():
    p = argparse.ArgumentParser(description="Train MNIST MLP")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--h1", type=int, default=256, help="Hidden size 1")
    p.add_argument("--h2", type=int, default=128, help="Hidden size 2")
    p.add_argument("--teleport-epoch", type=int, default=0, help="Epoch to apply path rescaling")
    p.add_argument("--nb-iter-optim-rescaling", type=int, default=1, help="Number of iterations for the path rescaling optimization")
    p.add_argument("--nb-iter", type=int, default=1, help="Number of repetitions of the whole experiment (for statistics)")
    args = p.parse_args()
    LOSS, ACC = fit_with_telportation(
        epochs=args.epochs,
        lr=args.lr,
        hidden=(args.h1, args.h2),
        ep_teleport=args.teleport_epoch,
        nb_iter_optim_rescaling=args.nb_iter_optim_rescaling,
        nb_iter=args.nb_iter,
    )
    torch.save(LOSS, _ensure_outdir("results/") / "mnist_loss.pt")
    torch.save(ACC, _ensure_outdir("results/") / "mnist_acc.pt")
    plot_mean_var_curves(LOSS=LOSS, mood="loss", ep_teleport=args.teleport_epoch, outdir="results/", fname_prefix="mnist")
    plot_mean_var_curves(ACC=ACC, mood="accuracy", ep_teleport=args.teleport_epoch, outdir="results/", fname_prefix="mnist")
    # test_neuron_rescaling_functionality()
