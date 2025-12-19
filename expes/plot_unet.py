import argparse
from pathcond.plot import plot_mean_var_curves_psnr, plot_mean_var_curves_triptych_epochs_times_lr
from pathcond.utils import _ensure_outdir
import torch
from pathlib import Path
import numpy as np





def main():

    file = "lr_1"

    dic = torch.load(_ensure_outdir(f"results/unet/deblurring/{file}/") / f"dic_{file}.pt", weights_only=False)
    imgdir = _ensure_outdir(f"images/unet/deblurring/{file}/")

    LOSS = dic["LOSS"]
    ACC_TRAIN = dic["PSNR_TRAIN"]
    ACC_TEST = dic["PSNR_TEST"]
    nb_lr = LOSS.shape[0]
    learning_rates = np.logspace(-4, 0, nb_lr)
    print(learning_rates)

    save_path = plot_mean_var_curves_psnr(
    LOSS=LOSS,
    ACC_TRAIN=ACC_TRAIN,
    ACC_TEST=ACC_TEST,
    learning_rates=learning_rates,
    ep_teleport=None,          # ou un entier, ex: 1
    outdir=imgdir,
    fname="curves_triptych_deblurring_unet.pdf",
    show_panels=("loss", "train", "test"),
)
    print(f"Saved plot to {save_path}")



if __name__ == "__main__":
    main()
