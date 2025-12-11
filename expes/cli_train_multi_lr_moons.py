import argparse
from expes.train_multi_lr_moons import fit_with_telportation, rescaling_path_dynamics
from pathcond.plot import plot_mean_var_curves, plot_boxplots
from pathcond.utils import _ensure_outdir
from expes.cli_boxplots_moons import main as plot_boxplots_moons
import torch


def main():
    p = argparse.ArgumentParser(description="Boxplot Moons MLP")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--nb-lr", type=int, default=10, help="Numnber of learning rates")
    p.add_argument("--frac", type=float, default=0.01, help="Fraction of the dataset to use for training")
    p.add_argument("--teleport-epoch", type=int, default=0, help="Epoch to apply path rescaling")
    p.add_argument("--nb-iter-optim-rescaling", type=int, default=100, help="Number of iterations for the path rescaling optimization")
    p.add_argument("--nb-iter", type=int, default=1, help="Number of repetitions of the whole experiment (for statistics)")
    args = p.parse_args()
    TIME, EPOCHS = fit_with_telportation(
        epochs=args.epochs,
        nb_lr=args.nb_lr,
        frac=args.frac,
        hidden=(32, 32),
        ep_teleport=args.teleport_epoch,
        nb_iter_optim_rescaling=args.nb_iter_optim_rescaling,
        nb_iter=args.nb_iter,
        data="moons",
        balanced=True,
    )
    # torch.save(LOSS, _ensure_outdir("results/") / "multi_lr_moons_loss.pt")
    # torch.save(ACC_TRAIN, _ensure_outdir("results/") / "multi_lr_moons_acc_train.pt")
    # torch.save(ACC_TEST, _ensure_outdir("results/") / "multi_lr_moons_acc_test.pt")
    torch.save(TIME, _ensure_outdir("results/") / "multi_lr_moons_time.pt")
    torch.save(EPOCHS, _ensure_outdir("results/") / "multi_lr_moons_epochs.pt")
    # torch.save(GRAD, _ensure_outdir("results/first_fig/") / "multi_lr_moons_grad.pt")
    # torch.save(DIAG_G, _ensure_outdir("results/first_fig/") / "multi_lr_moons_diag_g.pt")


    plot_boxplots_moons()  # Pass learning_rates to the plotting function
