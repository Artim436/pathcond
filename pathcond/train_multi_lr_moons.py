from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from pathcond.models import MNISTMLP, Moons_MLP, Moons_MLP_unbalanced, MLP, apply_init
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model, compute_diag_G, optimize_neuron_rescaling_polynomial_jitted
from pathcond.plot import plot_rescaling_analysis
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np


import time



def fit_with_telportation(  
    epochs: int = 5,
    hidden=(2, 2),
    ep_teleport: int = 0,
    nb_lr: int = 10,
    nb_iter_optim_rescaling: int = 1,
    nb_iter: int = 1,
    frac: float = 1.0,
    data="moons",
    balanced=True,
) -> Tuple[
    torch.Tensor, torch.Tensor
]:
    """
    Entraîne 4 variantes du même MLP sur MNIST :
      - sgd        : modèle principal entraîné en SGD + teleportation
      - ref_sgd    : référence en SGD (sans téléportation)
      - adam       : modèle entraîné avec Adam + teleportation
      - ref_adam   : référence avec Adam

    Renvoie :
      model_sgd, loss_sgd, loss_ref_sgd, acc_sgd, acc_ref_sgd,
      loss_adam, loss_ref_adam, acc_adam, acc_ref_adam
    """
    architectures = [
        [2, 8, 16, 8, 2],
        [2, 16, 32, 32, 16, 2],
        [2, 16, 32, 64, 32, 16, 2],
        [2, 32, 64, 128, 64, 32, 2],
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adam = False
    test_model = Moons_MLP(hidden[0], hidden[1], seed=0)
    nb_params = sum(p.numel() for p in test_model.parameters())
    learning_rates = [1]
    # archi_learning_rates = [0.1, 0.03, 0.02, 0.015]
    archi_learning_rates = [0.01]*4
    ACC_TEST = torch.zeros((len(architectures), nb_lr, nb_iter, epochs, 2))  # sgd, ref_sgd
    ACC_TRAIN = torch.zeros((len(architectures), nb_lr, nb_iter, epochs, 2))
    LOSS = torch.zeros((len(architectures), nb_lr, nb_iter, epochs, 2))
    TIME = torch.zeros((len(architectures), 1, nb_iter, 1, 2))
    EPOCHS  = torch.zeros((len(architectures), 1, nb_iter, 1, 2))
    # GRAD = torch.zeros((nb_lr, nb_iter, epochs, 2, nb_params))
    # DIAG_G = torch.zeros((nb_lr, nb_iter, epochs, 2, nb_params))

    # ep_teleport = [k*100 for k in range(epochs//100)]
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    for archi_index, architecture in tqdm(enumerate(architectures)):
        for lr_index, lr in enumerate(learning_rates):
            lr = archi_learning_rates[archi_index]
            for it in range(nb_iter):

                torch.manual_seed(it)
                

                # --- construction des variantes de modèles/optimiseurs de façon déclarative
                def make_model(seed: int = 0, device=None):
                    model = MLP(architecture, seed=seed)
                    # apply_init(model, scheme="kaiming_uniform")
                    return model.to(device) if device is not None else model

                
                criterion = nn.CrossEntropyLoss()

                variants = {
                    "sgd": {
                        "model": make_model(seed=it, device=device),
                        "optimizer": None,  # défini juste après
                        "trainer": torch.optim.SGD,
                        "hist_loss": [],
                        "hist_acc_train": [],
                        "hist_acc_test": [],
                        "label": "sgd",
                        "hist_grad": [],
                    },
                    "ref_sgd": {
                        "model": make_model(seed=it, device=device),
                        "optimizer": None,
                        "trainer": torch.optim.SGD,
                        "hist_loss": [],
                        "hist_acc_train": [],
                        "hist_acc_test": [],
                        "label": "ref sgd",
                        "hist_grad": [],
                    } 
                }

                # initialise les optimiseurs
                for v in variants.values():
                    v["optimizer"] = v["trainer"](v["model"].parameters(), lr=lr)


                ep_teleport = [ep_teleport] if isinstance(ep_teleport, int) else ep_teleport


                start_sgd = time.time()
                first_sgd = False
                

                # --- boucle d'entraînement
                for ep in range(epochs):
                    # téléportation uniquement pour le modèle principal 'sgd'
                    if ep in ep_teleport:
                        variants["sgd"]["model"] = rescaling_path_dynamics_with_jit(
                            variants["sgd"]["model"], nb_iter=nb_iter_optim_rescaling, device=device, data=data
                        )
                        # print(f"Rescaling applied in {end_teleport - start_teleport:.2f} seconds.")
                        variants["sgd"]["optimizer"] = torch.optim.SGD(
                            variants["sgd"]["model"].parameters(), lr=lr
                        )


                    # entraînement d'un epoch pour chaque variante
                    key = "sgd"
                    v = variants[key]
                    v["model"].train()
                    v["model"].to(device)
                    v["optimizer"].zero_grad()
                    logits = v["model"](X_train)
                    train_loss = criterion(logits, y_train.to(device))
                    train_loss.backward()
                    v["hist_grad"] = torch.cat([p.grad.flatten() for p in v["model"].parameters()]).cpu()
                    v["optimizer"].step()
                    v["hist_loss"].append(train_loss.item())
                    preds = torch.argmax(logits, dim=1)
                    accuracy = (preds == y_train.to(device)).float().mean()
                    v["hist_acc_train"].append(accuracy)

                    # évaluation
                    v = variants[key]
                    v["model"].eval()
                    v["model"].to(device)
                    with torch.no_grad():
                        logits = v["model"](X_test)
                        preds = torch.argmax(logits, dim=1)
                        accuracy = (preds == y_test.to(device)).float().mean()
                        v["hist_acc_test"].append(accuracy)
                        if (accuracy >= 0.99 and not first_sgd) or (ep == epochs - 1 and not first_sgd):
                            first_sgd = True
                            end_sgd = time.time()
                            TIME[archi_index, 0, it, 0, 0] = end_sgd - start_sgd
                            EPOCHS[archi_index, 0, it, 0, 0] = ep + 1
                            print("it: ", it, key, " archi / lr: ", architecture, " / ", lr, " epochs to 99%: ", ep, " time: ", end_sgd - start_sgd)
                            break
                        

                start_ref_sgd = time.time()
                first_ref_sgd = False
                for ep in range(epochs):
                    key = "ref_sgd"
                    v = variants[key]
                    v["model"].train()
                    v["model"].to(device)
                    v["optimizer"].zero_grad()
                    logits = v["model"](X_train)
                    train_loss = criterion(logits, y_train.to(device))
                    train_loss.backward()
                    v["hist_grad"] = torch.cat([p.grad.flatten() for p in v["model"].parameters()]).cpu()
                    v["optimizer"].step()
                    v["hist_loss"].append(train_loss.item())
                    preds = torch.argmax(logits, dim=1)
                    accuracy = (preds == y_train.to(device)).float().mean()
                    v["hist_acc_train"].append(accuracy)

                    # évaluation
                    v = variants[key]
                    v["model"].eval()
                    v["model"].to(device)
                    with torch.no_grad():
                        logits = v["model"](X_test)
                        preds = torch.argmax(logits, dim=1)
                        accuracy = (preds == y_test.to(device)).float().mean()
                        v["hist_acc_test"].append(accuracy)
                        if (accuracy >= 0.99 and not first_ref_sgd) or (ep == epochs - 1 and not first_ref_sgd):
                            first_ref_sgd = True
                            end_ref_sgd = time.time()
                            TIME[archi_index, 0, it, 0, 1] = end_ref_sgd - start_ref_sgd
                            EPOCHS[archi_index, 0, it, 0, 1] = ep + 1
                            print("it: ", it, key, " archi / lr: ", architecture, " / ", lr, " epochs to 99%: ", ep, " time: ", end_ref_sgd - start_ref_sgd)
                            break
                # loss_history = variants["sgd"]["hist_loss"]
                # loss_history_ref = variants["ref_sgd"]["hist_loss"]

                # acc_train_history = variants["sgd"]["hist_acc_train"]
                # acc_train_history_ref = variants["ref_sgd"]["hist_acc_train"]

                # acc_test_history = variants["sgd"]["hist_acc_test"]
                # acc_test_history_ref = variants["ref_sgd"]["hist_acc_test"]


                # LOSS[archi_index, lr_index, it, :, 0] = torch.tensor(loss_history)
                # LOSS[archi_index, lr_index, it, :, 1] = torch.tensor(loss_history_ref)

                # ACC_TRAIN[archi_index, lr_index, it, :, 0] = torch.tensor(acc_train_history)
                # ACC_TRAIN[archi_index, lr_index, it, :, 1] = torch.tensor(acc_train_history_ref)

                # ACC_TEST[archi_index, lr_index, it, :, 0] = torch.tensor(acc_test_history)
                # ACC_TEST[archi_index, lr_index, it, :, 1] = torch.tensor(acc_test_history_ref)


    return (
        TIME, EPOCHS
    )



def rescaling_path_dynamics(model, verbose: bool = False, soft: bool = True, nb_iter=1, name: str = "sgd", device="cpu", data="moons") -> MNISTMLP:
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    inputs = torch.randn(3, 2).to(device)
    model = model.to(device)
    original_output = model(inputs)
    if verbose:
        print(f"   Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
        # print(f"   Architecture: {model}")


    # 3. Optimisation séquentielle
    if verbose:
        print("\n1. Optimisation séquentielle neurone par neurone...")
        print("-" * 50)

    if verbose:
        BZ_opt, Z_opt, alpha, OBJ_hist = optimize_neuron_rescaling_polynomial(model=model, n_iter=nb_iter, verbose=verbose, tol=1e-6)
        lambdas_history = torch.exp(-(Z_opt.clone().detach())/2).cpu().numpy()
        OBJ_hist = torch.tensor(OBJ_hist).to("cpu")
        plot_rescaling_analysis(final_model=final_model, lambdas_history=lambdas_history, norms_history=OBJ_hist, nb_iter_optim=nb_iter, name=name)
    else:
        BZ_opt = optimize_neuron_rescaling_polynomial(model=model, n_iter=nb_iter, verbose=verbose, tol=1e-6)
    final_model = reweight_model(model, BZ_opt).to(dtype=torch.float32, device=device)


    


    # 4. Vérification de la sortie finale
    if verbose:
        print("\n4. Vérification de la préservation de la sortie finale...")
    final_output = final_model(inputs)
    assert torch.allclose(original_output, final_output, atol=1e-5)
    # print("✅ Sortie finale préservée après rescaling.")

    return final_model


def rescaling_path_dynamics_with_jit(model,  nb_iter=1, device="cpu", data="moons") -> MNISTMLP:
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    inputs = torch.randn(3, 2).to(device)
    model = model.to(device)
    original_output = model(inputs)
    BZ_opt = optimize_neuron_rescaling_polynomial_jitted(model=model, n_iter=nb_iter, tol=1e-6)
    final_model = reweight_model(model, BZ_opt).to(dtype=torch.float32, device=device)
    final_output = final_model(inputs)
    assert torch.allclose(original_output, final_output, atol=1e-5)
    # print("✅ Sortie finale préservée après rescaling.")

    return final_model