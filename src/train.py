from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn as nn
from mlp import MNISTMLP
from data import mnist_loaders
from rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model
from plot import plot_rescaling_analysis

import time



def train_one_epoch(model, loader, criterion, optimizer, device, fraction: float = 1.0) -> float:
    """
    Entraîne le modèle sur une fraction du dataset.

    Args:
        model: nn.Module
        loader: DataLoader
        criterion: fonction de perte
        optimizer: Optimizer
        device: cuda ou cpu
        fraction: fraction du dataset à utiliser (0 < fraction <= 1.0)

    Returns:
        float: perte moyenne sur les échantillons vus
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    # nombre de batches maximum à parcourir
    max_batches = int(len(loader) * fraction)

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model.forward(x, device=device), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model.forward(x, device=device).argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def fit_with_telportation(  
    epochs: int = 5,
    lr: float = 1e-3,
    hidden=(2, 2),
    ep_teleport: int = 0,
    nb_iter_optim_rescaling: int = 1,
    nb_iter: int = 1,
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
    if adam:
        ACC = torch.zeros((nb_iter, epochs, 4))  # sgd, ref_sgd, adam, ref_adam
        LOSS = torch.zeros((nb_iter, epochs, 4))
    else:
        ACC = torch.zeros((nb_iter, epochs, 2))  # sgd, ref_sgd
        LOSS = torch.zeros((nb_iter, epochs, 2))

    for it in range(nb_iter):

        start = time.time()
        torch.manual_seed(it)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # --- construction des variantes de modèles/optimiseurs de façon déclarative
        def make_model(seed: int = 0, device=None) -> MNISTMLP:
            model = MNISTMLP(hidden[0], hidden[1], p_drop=0.0, seed=seed)
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
        train_dl, test_dl = mnist_loaders()

        ep_teleport = [ep_teleport] if isinstance(ep_teleport, int) else ep_teleport

        # --- boucle d'entraînement
        for ep in range(epochs):
            # téléportation uniquement pour le modèle principal 'sgd'
            if ep in ep_teleport:
                start_teleport = time.time()
                variants["sgd"]["model"], alpha = rescaling_path_dynamics(
                    variants["sgd"]["model"], verbose=True, soft=True, name="sgd", nb_iter=nb_iter_optim_rescaling, device=device
                )
                end_teleport = time.time()
                print(f"Rescaling applied in {end_teleport - start_teleport:.2f} seconds.")
                variants["sgd"]["optimizer"] = torch.optim.SGD(
                    variants["sgd"]["model"].parameters(), lr=lr*alpha
                )
            if ep in ep_teleport and adam:
                variants["adam"]["model"] = rescaling_path_dynamics(
                    variants["adam"]["model"], verbose=True, soft=True, name="adam", nb_iter=nb_iter_optim_rescaling, device=device
                )
                # IMPORTANT: réinitialiser l'optimiseur pour récupérer les nouveaux paramètres
                variants["adam"]["optimizer"] = torch.optim.Adam(
                    variants["adam"]["model"].parameters(), lr=lr
                )

            # entraînement d'un epoch pour chaque variante
            for key in ["sgd", "ref_sgd"] if not adam else ["sgd", "ref_sgd", "adam", "ref_adam"]:
                v = variants[key]
                train_loss = train_one_epoch(v["model"], train_dl, criterion, v["optimizer"], device, fraction=0.01)
                v["hist_loss"].append(train_loss)

            # évaluation
            for key in ["sgd", "ref_sgd"] if not adam else ["sgd", "ref_sgd", "adam", "ref_adam"]:
                v = variants[key]
                acc = evaluate(v["model"], test_dl, device)
                v["hist_acc"].append(acc)

            # logs identiques (mais générés de façon compacte)
            print(
                f"Epoch {ep}/{epochs} - "
                f"loss: {variants['sgd']['hist_loss'][-1]:.4f} - acc: {variants['sgd']['hist_acc'][-1]:.4f}"
            )
            print(
                f"Epoch {ep}/{epochs} - ref loss: {variants['ref_sgd']['hist_loss'][-1]:.4f} - "
                f"ref acc: {variants['ref_sgd']['hist_acc'][-1]:.4f}"
            )
            if adam:
                print(
                    f"Epoch {ep}/{epochs} - adam loss: {variants['adam']['hist_loss'][-1]:.4f} - "
                    f"adam acc: {variants['adam']['hist_acc'][-1]:.4f}"
                )
                print(
                    f"Epoch {ep}/{epochs} - ref adam loss: {variants['ref_adam']['hist_loss'][-1]:.4f} - "
                    f"ref adam acc: {variants['ref_adam']['hist_acc'][-1]:.4f}"
                )

        end = time.time()
        print(f"Training completed in {end - start:.2f} seconds.")

        # --- restitue EXACTEMENT les mêmes éléments et dans le même ordre
        loss_history = variants["sgd"]["hist_loss"]
        loss_history_ref = variants["ref_sgd"]["hist_loss"]
        LOSS[it, :, 0] = torch.tensor(loss_history)
        LOSS[it, :, 1] = torch.tensor(loss_history_ref)
        if adam:
            loss_history_adam = variants["adam"]["hist_loss"]
            loss_history_ref_adam = variants["ref_adam"]["hist_loss"]
            LOSS[it, :, 2] = torch.tensor(loss_history_adam)
            LOSS[it, :, 3] = torch.tensor(loss_history_ref_adam)



        acc_history = variants["sgd"]["hist_acc"]
        acc_history_ref = variants["ref_sgd"]["hist_acc"]
        ACC[it, :, 0] = torch.tensor(acc_history)
        ACC[it, :, 1] = torch.tensor(acc_history_ref)
        if adam:
            acc_history_adam = variants["adam"]["hist_acc"]
            acc_history_ref_adam = variants["ref_adam"]["hist_acc"]
            ACC[it, :, 2] = torch.tensor(acc_history_adam)
            ACC[it, :, 3] = torch.tensor(acc_history_ref_adam)

    return (
        LOSS, ACC
    )



def rescaling_path_dynamics(model, verbose: bool = False, soft: bool = True, nb_iter=1, name: str = "sgd", device="cpu") -> MNISTMLP:
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    inputs = torch.randn(3, 1, 28, 28).to(device)
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

    BZ_opt, alpha, Z_opt, OBJ_hist = optimize_neuron_rescaling_polynomial(model=model, n_iter=nb_iter, verbose=verbose, tol=1e-6)
    final_model = reweight_model(model, BZ_opt)
    lambdas_history = torch.exp(-(Z_opt.clone().detach())/2).to("cpu")
    OBJ_hist = torch.tensor(OBJ_hist).to("cpu")
    plot_rescaling_analysis(final_model=final_model, lambdas_history=lambdas_history, norms_history=OBJ_hist, nb_iter_optim=nb_iter, name=name)

    # 4. Vérification de la sortie finale
    if verbose:
        print("\n4. Vérification de la préservation de la sortie finale...")
    final_output = final_model.forward(inputs, device=device)
    torch.allclose(original_output, final_output, atol=1e-5)
    print("✅ Sortie finale préservée après rescaling, alpha =", alpha.item())

    return final_model, alpha


