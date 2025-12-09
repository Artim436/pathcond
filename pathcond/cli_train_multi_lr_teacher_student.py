import argparse
from pathcond.train_multi_lr_teacher_student import fit_with_telportation
from pathcond.plot import plot_mean_var_curves, plot_boxplots
from pathcond.utils import _ensure_outdir
from pathcond.cli_boxplots_teacher_student import main as plot_boxplots_toy
import torch


def main():
    p = argparse.ArgumentParser(description="Boxplot Moons MLP")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--nb-init", type=int, default=10, help="Numnber of inits")
    p.add_argument("--frac", type=float, default=0.01, help="Fraction of the dataset to use for training")
    p.add_argument("--teleport-epoch", type=int, default=0, help="Epoch to apply path rescaling")
    p.add_argument("--nb-iter-optim-rescaling", type=int, default=100, help="Number of iterations for the path rescaling optimization")
    p.add_argument("--nb-iter", type=int, default=1, help="Number of repetitions of the whole experiment (for statistics)")
    args = p.parse_args()
    LOSS, GRAD = fit_with_telportation(
        epochs=args.epochs,
        nb_init=args.nb_init,
        frac=args.frac,
        hidden=128,
        ep_teleport=args.teleport_epoch,
        nb_iter_optim_rescaling=args.nb_iter_optim_rescaling,
        nb_iter=args.nb_iter
    )
    torch.save(LOSS, _ensure_outdir("results/") / "multi_lr_ts_loss.pt")

    torch.save(GRAD, _ensure_outdir("results/") / "multi_lr_ts_grad.pt")

    plot_boxplots_toy()
    
