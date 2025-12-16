import argparse
from expes.train_resnet import fit_with_telportation, rescaling_path_dynamics
from pathcond.plot import plot_mean_var_curves
from pathcond.utils import _ensure_outdir
import torch
from expes.cli_plot_resnet_cifar import main as plot_main


def main():
    p = argparse.ArgumentParser(description="Train MNIST MLP")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--nb-iter", type=int, default=1, help="Number of repetitions of the whole experiment (for statistics)")
    p.add_argument("--frac", type=float, default=1.0, help="Fraction of the dataset to use for training (between 0 and 1)")
    p.add_argument("--nb-lr", type=int, default=10, help="Number of learning rates to test for the rescaling analysis")
    args = p.parse_args()
    LOSS, ACC_TRAIN, ACC_TEST, TIME, EPOCHS = fit_with_telportation(
        epochs=args.epochs,
        nb_iter=args.nb_iter,
        frac=args.frac, data="cifar10",
        nb_lr = args.nb_lr,
    )
    torch.save(LOSS, _ensure_outdir("results/resnet/cifar10/") / "loss.pt")
    torch.save(ACC_TEST, _ensure_outdir("results/resnet/cifar10/") / "acc_test.pt")
    torch.save(ACC_TRAIN, _ensure_outdir("results/resnet/cifar10/") / "acc_train.pt")
    torch.save(TIME, _ensure_outdir("results/resnet/cifar10/") / "time.pt")
    torch.save(EPOCHS, _ensure_outdir("results/resnet/cifar10/") / "epochs.pt")

    plot_main()