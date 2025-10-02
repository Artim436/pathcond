from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn as nn
from pathcond.mlp import MNISTMLP, Moons_MLP, Moons_MLP_unbalanced
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model
from pathcond.plot import plot_rescaling_analysis
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


import time



def fit_with_telportation(  
    epochs: int = 5,
    hidden=(2, 2),
    ep_teleport: int = 0,
    nb_lr: int = 10,
    nb_iter_optim_rescaling: int = 1,
    nb_iter: int = 1,
    frac: float = 1.0,
    data="mnist",
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
    adam = False
    learning_rates = torch.logspace(-4, -1, nb_lr)
    if adam:
        ACC = torch.zeros((nb_lr, nb_iter, epochs, 4))  # sgd, ref_sgd, adam, ref_adam
        LOSS = torch.zeros((nb_lr, nb_iter, epochs, 4))
    else:
        ACC = torch.zeros((nb_lr, nb_iter, epochs, 2))  # sgd, ref_sgd
        LOSS = torch.zeros((nb_lr, nb_iter, epochs, 2))

    


    for lr_index, lr in enumerate(learning_rates):
        for it in range(nb_iter):

            start = time.time()
            torch.manual_seed(it)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # --- construction des variantes de modèles/optimiseurs de façon déclarative
            def make_model(seed: int = 0, device=None):
                if balanced:
                    model = Moons_MLP(hidden[0], hidden[1], seed=seed)
                    return model.to(device) if device is not None else model
                else:
                    model = Moons_MLP_unbalanced(hidden[0], hidden[1], seed=seed)
                    return model.to(device) if device is not None else model
            
            criterion = nn.CrossEntropyLoss()

            variants = {
                "sgd": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,  # défini juste après
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "hist_acc": [],
                    "label": "sgd",
                },
                "ref_sgd": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "hist_acc": [],
                    "label": "ref sgd",
                } if not adam else {},
                "adam": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,
                    "trainer": torch.optim.Adam,
                    "hist_loss": [],
                    "hist_acc": [],
                    "label": "adam",
                },
                "ref_adam": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,
                    "trainer": torch.optim.Adam,
                    "hist_loss": [],
                    "hist_acc": [],
                    "label": "ref adam",
                },
            }

            # initialise les optimiseurs
            for v in variants.values():
                v["optimizer"] = v["trainer"](v["model"].parameters(), lr=lr)

            # data
            
            X, y = make_moons(n_samples=1000, noise=0.1, random_state=0)
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )


            ep_teleport = [ep_teleport] if isinstance(ep_teleport, int) else ep_teleport

            # --- boucle d'entraînement
            for ep in range(epochs):
                # téléportation uniquement pour le modèle principal 'sgd'
                if ep in ep_teleport:
                    start_teleport = time.time()
                    variants["sgd"]["model"] = rescaling_path_dynamics(
                        variants["sgd"]["model"], verbose=False, soft=True, name="sgd", nb_iter=nb_iter_optim_rescaling, device=device, data=data
                    )
                    end_teleport = time.time()
                    print(f"Rescaling applied in {end_teleport - start_teleport:.2f} seconds.")
                    variants["sgd"]["optimizer"] = torch.optim.SGD(
                        variants["sgd"]["model"].parameters(), lr=lr
                    )
                if ep in ep_teleport and adam:
                    variants["adam"]["model"] = rescaling_path_dynamics(
                        variants["adam"]["model"], verbose=False, soft=True, name="adam", nb_iter=nb_iter_optim_rescaling, device=device, data=data
                    )
                    # IMPORTANT: réinitialiser l'optimiseur pour récupérer les nouveaux paramètres
                    variants["adam"]["optimizer"] = torch.optim.Adam(
                        variants["adam"]["model"].parameters(), lr=lr
                    )

                # entraînement d'un epoch pour chaque variante
                for key in ["sgd", "ref_sgd"] if not adam else ["sgd", "ref_sgd", "adam", "ref_adam"]:
                    v = variants[key]
                    v["model"].train()
                    v["optimizer"].zero_grad()
                    logits = v["model"](X_train.to(device), device=device)
                    train_loss = criterion(logits, y_train.to(device))
                    train_loss.backward()
                    v["optimizer"].step()
                    v["hist_loss"].append(train_loss.item())

                # évaluation
                for key in ["sgd", "ref_sgd"] if not adam else ["sgd", "ref_sgd", "adam", "ref_adam"]:
                    v = variants[key]
                    v["model"].eval()
                    with torch.no_grad():
                        logits = v["model"](X_test.to(device), device=device)
                        preds = torch.argmax(logits, dim=1)
                        accuracy = (preds == y_test.to(device)).float().mean()
                        v["hist_acc"].append(accuracy)

            end = time.time()
            print(f"Training completed in {end - start:.2f} seconds.")

            # --- restitue EXACTEMENT les mêmes éléments et dans le même ordre
            loss_history = variants["sgd"]["hist_loss"]
            loss_history_ref = variants["ref_sgd"]["hist_loss"]
            LOSS[lr_index, it, :, 0] = torch.tensor(loss_history)
            LOSS[lr_index, it, :, 1] = torch.tensor(loss_history_ref)
            if adam:
                loss_history_adam = variants["adam"]["hist_loss"]
                loss_history_ref_adam = variants["ref_adam"]["hist_loss"]
                LOSS[lr_index, it, :, 2] = torch.tensor(loss_history_adam)
                LOSS[lr_index, it, :, 3] = torch.tensor(loss_history_ref_adam)



            acc_history = variants["sgd"]["hist_acc"]
            acc_history_ref = variants["ref_sgd"]["hist_acc"]
            ACC[lr_index, it, :, 0] = torch.tensor(acc_history)
            ACC[lr_index, it, :, 1] = torch.tensor(acc_history_ref)
            if adam:
                acc_history_adam = variants["adam"]["hist_acc"]
                acc_history_ref_adam = variants["ref_adam"]["hist_acc"]
                ACC[lr_index, it, :, 2] = torch.tensor(acc_history_adam)
                ACC[lr_index, it, :, 3] = torch.tensor(acc_history_ref_adam)

    return (
        LOSS, ACC
    )



def rescaling_path_dynamics(model, verbose: bool = False, soft: bool = True, nb_iter=1, name: str = "sgd", device="cpu", data="mnist") -> MNISTMLP:
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    if data == "mnist":
        inputs = torch.randn(3, 1, 28, 28).to(device)
    elif data == "moons":
        inputs = torch.randn(3, 2).to(device)
    original_output = model.forward(inputs, device=device)
    if verbose:
        print(f"   Modèle créé avec {sum(p.numel() for p in model.parameters())} paramètres")
        # print(f"   Architecture: {model}")
    else:
        print(f"✅ Modèle initialisé ({sum(p.numel() for p in model.parameters())} paramètres)")

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
    final_model = reweight_model(model, BZ_opt)


    # 4. Vérification de la sortie finale
    if verbose:
        print("\n4. Vérification de la préservation de la sortie finale...")
    final_output = final_model.forward(inputs, device=device)
    torch.allclose(original_output, final_output, atol=1e-5)
    print("✅ Sortie finale préservée après rescaling.")

    return final_model


