import argparse
from pathcond.train import fit_with_telportation, rescaling_path_dynamics
from pathcond.plot import plot_mean_var_curves, plot_boxplots, plot_boxplots_2x2, plot_convergence_vs_final_boxplots_2x2, plot_mean_var_curves_all_lr
from pathcond.plot import plot_mean_var_curves_triptych, plot_mean_var_curves_triptych_epochs_times_lr, plot_grad_path_length
from pathcond.plot import plot_mean_var_curves_triptych_epochs_times_lr_archi, plot_time_vs_params
from pathlib import Path
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model
from pathcond.utils import _ensure_outdir
import torch
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import image as mpimg
from pathcond.models import Moons_MLP, MLP
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def main():
    resdir = Path("results")
    images = Path("images"); images.mkdir(exist_ok=True, parents=True)
    imgdir = Path("images"); imgdir.mkdir(parents=True, exist_ok=True)
    resdir_first_fig = Path("results/first_fig")


    # LOSS_FF = torch.load(resdir_first_fig / "multi_lr_moons_loss.pt")
    # ACC_TRAIN_FF  = torch.load(resdir_first_fig / "multi_lr_moons_acc_train.pt")
    # ACC_TEST_FF  = torch.load(resdir_first_fig / "multi_lr_moons_acc_test.pt")


    LOSS = torch.load(resdir / "multi_lr_moons_loss.pt")
    ACC_TRAIN  = torch.load(resdir / "multi_lr_moons_acc_train.pt")
    ACC_TEST  = torch.load(resdir / "multi_lr_moons_acc_test.pt")
    TIME = torch.load(resdir / "multi_lr_moons_time.pt")
    EPOCHS = torch.load(resdir / "multi_lr_moons_epochs.pt")



    learning_rates = torch.logspace(-3, np.log10(0.02), LOSS.shape[1]).tolist()

    architectures = [
        [2, 8, 16, 8, 2],
        [2, 16, 32, 32, 16, 2],
        [2, 16, 32, 64, 32, 16, 2],
        [2, 32, 64, 128, 64, 32, 2],
    ]
    # archi_learning_rates = [0.1, 0.03, 0.02, 0.015]
    # archi_learning_rates = torch.tensor(archi_learning_rates)
    # EPOCHS = EPOCHS * archi_learning_rates.reshape(-1, 1, 1, 1, 1)

    nb_params = []
    for archi in architectures:
        model = MLP(archi)
        nb_params.append(sum(p.numel() for p in model.parameters()))

    print(nb_params)



    # GRAD_DO = torch.load(resdir / "multi_lr_moons_grad_do.pt")

    # nb_lr = LOSS_DONOT.shape[0]
    # learning_rates = torch.logspace(-4, 0, nb_lr).tolist()
    # learning_rates_trunc = learning_rates[-4:]



#     save_path = plot_mean_var_curves_triptych_epochs_times_lr_archi(
#     LOSS=LOSS_FF,
#     ACC_TRAIN=ACC_TRAIN_FF,
#     ACC_TEST=ACC_TEST_FF,
#     learning_rates=learning_rates,
#     ep_teleport=None,          # ou un entier, ex: 1
#     outdir=imgdir,
#     nb_params=nb_params,
#     fname="epochs_first_fig.pdf",
#     title_suffix="First Figure",
#     show_panels=("loss",  "test")
# )
#     print("Saved to:", save_path)

    save_path = plot_time_vs_params(TIME, EPOCHS, nb_params, method_names=("Ours", "Baseline"), fname="time_vs_params_epochs_lr.pdf", invert_axes=False)
    print("Saved to:", save_path)

    save_path = plot_time_vs_params(TIME, EPOCHS, nb_params, method_names=("Ours", "Baseline"), fname="time_vs_params_epochs_lr_inverted.pdf", invert_axes=True)
    print("Saved to:", save_path)


#     LOSS = LOSS[0]
#     ACC_TRAIN = ACC_TRAIN[0]
#     ACC_TEST = ACC_TEST[0]

#     save_path = plot_mean_var_curves_triptych(
#     LOSS=LOSS,
#     ACC_TRAIN=ACC_TRAIN,
#     ACC_TEST=ACC_TEST,
#     learning_rates=learning_rates,
#     ep_teleport=None,          # ou un entier, ex: 1
#     outdir=imgdir,
#     fname="loss_archi_1.pdf",
#     title_suffix="with effects on training dynamics",
#     show_panels=("loss", "test", "train")
# )
#     print("Saved to:", save_path)


#     save_path = plot_mean_var_curves_triptych_epochs_times_lr(
#     LOSS=LOSS,
#     ACC_TRAIN=ACC_TRAIN,
#     ACC_TEST=ACC_TEST,
#     learning_rates=learning_rates,
#     ep_teleport=None,          # ou un entier, ex: 1
#     outdir=imgdir,
#     fname="epochs_archi_1.pdf",
#     title_suffix="with effects on training dynamics",
#     show_panels=("loss", "test", "train")
# )
#     print("Saved to:", save_path)


#     LOSS_DONOT = LOSS_FF[0]
#     ACC_TRAIN_DONOT = ACC_TRAIN_FF[0]
#     ACC_TEST_DONOT = ACC_TEST_FF[0]

#     save_path = plot_mean_var_curves_triptych(
#     LOSS=LOSS_DONOT,
#     ACC_TRAIN=ACC_TRAIN_DONOT,
#     ACC_TEST=ACC_TEST_DONOT,
#     learning_rates=learning_rates,
#     ep_teleport=None,          # ou un entier, ex: 1
#     outdir=imgdir,
#     fname="loss_donot.pdf",
#     title_suffix="without effects on training dynamics",
#     show_panels=("loss")
# )
#     print("Saved to:", save_path)

#     save_path = plot_mean_var_curves_triptych_epochs_times_lr(
#     LOSS=LOSS_DONOT,
#     ACC_TRAIN=ACC_TRAIN_DONOT,
#     ACC_TEST=ACC_TEST_DONOT,
#     learning_rates=learning_rates,
#     ep_teleport=None,          # ou un entier, ex: 1
#     outdir=imgdir,
#     fname="all_curves_triptych_epochs_times_lr_donot.pdf",
#     title_suffix="without effects on training dynamics",
#     show_panels=("loss",)
# )
#     print("Saved to:", save_path)

    

    # plot_grad_path_length(
    #     GRAD=GRAD_DO,
    #     learning_rates=learning_rates,
    #     savepath=imgdir / "grad_path_length_do.pdf",
    #     title="Gradient path length with effects on training dynamics"
    # )
    # print("Saved to:", imgdir / "grad_path_length_do.pdf")

    # plot_grad_path_length(
    #     GRAD=GRAD_DO,
    #     learning_rates=learning_rates,
    #     diag_g=DIAG_G_DO,
    #     savepath=imgdir / "grad_path_length_in_phi_space_do.pdf",
    #     title=r"Gradient path length in $\Phi$ space with effects on training dynamics"
    # )
    # print("Saved to:", imgdir / "grad_path_length_in_phi_space_do.pdf")


    # plot_grad_path_length(
    #     GRAD=GRAD_DONOT,
    #     learning_rates=learning_rates,
    #     savepath=imgdir / "grad_path_length_donot.pdf",
    #     title="Gradient path length without effects on training dynamics"
    # )
    # print("Saved to:", imgdir / "grad_path_length_donot.pdf")

    # plot_grad_path_length(
    #     GRAD=GRAD_DONOT,
    #     learning_rates=learning_rates,
    #     diag_g=DIAG_G_DONOT,
    #     savepath=imgdir / "grad_path_length_in_phi_space_donot.pdf",
    #     title=r"Gradient path length in $\Phi$ space without effects on training dynamics"
    # )
    # print("Saved to:", imgdir / "grad_path_length_in_phi_space_donot.pdf")


    # nb_iter = 10

    # COS_DONOT = torch.zeros(nb_iter)
    # COS_DO = torch.zeros(nb_iter)

    # for it in range(nb_iter):
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     # --- construction des variantes de modèles/optimiseurs de façon déclarative


    #     model_donot = make_model_donot(seed=it, device=device)
    #     grad_donot = GRAD_DONOT[0, it, 0, 0, :].to(device)
    #     grad_ref_donot = GRAD_DONOT[0, it, 0, 1, :].to(device)

    #     model_do = make_model_do(seed=it, device=device)
    #     grad_do = GRAD_DO[0, it, 0, 0, :].to(device)
    #     grad_ref_do = GRAD_DO[0, it, 0, 1, :].to(device)



    #     rescaling_donot = find_rescaling(model_donot, verbose=False,  nb_iter=50, device=device, data="moons")

    #     rescaling_do = find_rescaling(model_do, verbose=False, nb_iter=50, device=device, data="moons")

    #     scaled_grad_donot = grad_ref_donot * rescaling_donot
    #     scaled_grad_do = grad_ref_do * rescaling_do

    #     cos_sim_donot = torch.nn.functional.cosine_similarity(grad_donot.unsqueeze(0), scaled_grad_donot.unsqueeze(0)).item()
    #     cos_sim_do = torch.nn.functional.cosine_similarity(grad_do.unsqueeze(0), scaled_grad_do.unsqueeze(0)).item()
    #     COS_DO[it] = cos_sim_do
    #     COS_DONOT[it] = cos_sim_donot
    
    # data = pd.DataFrame({
    #     'value': torch.cat([COS_DONOT, COS_DO]).numpy(),
    #     'init': ['default'] * len(COS_DONOT) + ['special'] * len(COS_DO)
    # })

    # # Style global
    # sns.set(style="whitegrid", context="talk")

    # plt.figure(figsize=(7, 5))

    # # Points individuels (en arrière-plan)
    # sns.stripplot(
    #     data=data, x='init', y='value',
    #     color='gray', alpha=0.4, jitter=0.25, zorder=1
    # )

    # # Boxplot (en avant-plan)
    # sns.boxplot(
    #     data=data, x='init', y='value',
    #     width=0.5, fliersize=0, boxprops=dict(alpha=0.8),
    #     medianprops=dict(color='red', linewidth=2),
    #     zorder=2
    # )

    # plt.title('Similarity between original and rescaled gradient')
    # plt.ylabel('Cosine Similarity')
    # plt.ylim(0, None)
    # plt.grid(True, linestyle='--', alpha=0.6)
    # # plt.tight_layout()
    # # plus de place pour le x label
    # plt.subplots_adjust(bottom=0.15)
    # plt.savefig(imgdir / "cosine_similarity_boxplot.pdf")
        
    

    
