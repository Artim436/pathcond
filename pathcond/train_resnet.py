from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from pathcond.data import mnist_loaders, cifar10_loaders
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial_jitted_sparse, reweight_model_cnn
from tqdm import tqdm
from pathcond.models import resnet18_mnist, resnet18_cifar10

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
    model = model.to(device)

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model = model.to(device)
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def fit_with_telportation(
    epochs: int = 5,
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
    ACC_TRAIN = torch.zeros((nb_lr, nb_iter, epochs, 2))  # sgd, ref_sgd
    LOSS = torch.zeros((nb_lr, nb_iter, epochs, 2))
    ACC_TEST = torch.zeros((nb_lr, nb_iter, epochs, 2))

    # ep_teleport = [k*100 for k in range(epochs//100)]s
    print(torch.__version__)
    print(torch.version.cuda)
    print("cuda available:", torch.cuda.is_available())
    for lr_index, lr in tqdm(enumerate(learning_rates)):
        for it in range(nb_iter):

            torch.manual_seed(it)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # --- construction des variantes de modèles/optimiseurs de façon déclarative
            def make_model(seed: int = 0, device=None):
                if data == "mnist":
                    model = resnet18_mnist(num_classes=10, seed=seed).to(device)
                else:  # cifar10
                    model = resnet18_cifar10(num_classes=10, seed=seed).to(device)
                return model

            criterion = nn.CrossEntropyLoss()

            variants = {
                "sgd": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,  # défini juste après
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "hist_acc_tr": [],
                    "hist_acc": [],
                    "label": "sgd",
                },
                "ref_sgd": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "hist_acc_tr": [],
                    "hist_acc": [],
                    "label": "ref sgd",
                }
            }

            # initialise les optimiseurs
            for v in variants.values():
                v["optimizer"] = v["trainer"](v["model"].parameters(), lr=lr)

            ep_teleport = [ep_teleport] if isinstance(ep_teleport, int) else ep_teleport
            if data == "mnist":
                train_dl, test_dl = mnist_loaders(batch_size=128)
            else:  # cifar10
                train_dl, test_dl = cifar10_loaders(batch_size=128)

            # --- boucle d'entraînement
            for ep in range(epochs):
                # évaluation
                for key in ["sgd", "ref_sgd"]:
                    v = variants[key]
                    acc_tr = evaluate(v["model"], train_dl, device)
                    v["hist_acc_tr"].append(acc_tr)
                    acc = evaluate(v["model"], test_dl, device)
                    v["hist_acc"].append(acc)

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

                # entraînement d'un epoch pour chaque variante
                for key in ["sgd", "ref_sgd"]:
                    v = variants[key]
                    train_loss = train_one_epoch(v["model"], train_dl, criterion, v["optimizer"], device, fraction=frac)
                    v["hist_loss"].append(train_loss)

            # --- restitue EXACTEMENT les mêmes éléments et dans le même ordre
            loss_history = variants["sgd"]["hist_loss"]
            loss_history_ref = variants["ref_sgd"]["hist_loss"]
            LOSS[lr_index, it, :, 0] = torch.tensor(loss_history)
            LOSS[lr_index, it, :, 1] = torch.tensor(loss_history_ref)

            acc_history_tr = variants["sgd"]["hist_acc_tr"]
            acc_history_ref_tr = variants["ref_sgd"]["hist_acc_tr"]
            ACC_TRAIN[lr_index, it, :, 0] = torch.tensor(acc_history_tr)
            ACC_TRAIN[lr_index, it, :, 1] = torch.tensor(acc_history_ref_tr)

            acc_history = variants["sgd"]["hist_acc"]
            acc_history_ref = variants["ref_sgd"]["hist_acc"]
            ACC_TEST[lr_index, it, :, 0] = torch.tensor(acc_history)
            ACC_TEST[lr_index, it, :, 1] = torch.tensor(acc_history_ref)

    return (
        LOSS, ACC_TRAIN, ACC_TEST
    )


def rescaling_path_dynamics(model, verbose: bool = False, soft: bool = True, nb_iter=1, name: str = "sgd", device="cpu", data="mnist"):
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    if data == "mnist":
        inputs = torch.randn(16, 1, 28, 28).to(device)
    else:
        inputs = torch.randn(16, 3, 32, 32).to(device)
    BZ_opt, Z_opt = optimize_neuron_rescaling_polynomial_jitted_sparse(model, n_iter=1, tol=1e-6)
    final_model = reweight_model_cnn(model, BZ_opt, Z_opt).to(dtype=torch.float32, device=device)
    final_model = final_model.to(device).eval()
    model.eval()

    final_output = final_model(inputs)
    original_output = model(inputs)
    assert torch.allclose(original_output, final_output, atol=1e-5)
    # print("✅ Sortie finale préservée après rescaling.")

    return final_model
