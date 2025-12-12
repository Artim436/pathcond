from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn
from pathcond.data import mnist_loaders, cifar10_loaders
from pathcond.rescaling_polyn import optimize_rescaling_polynomial, reweight_model
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
    TIME = torch.zeros((nb_lr, nb_iter, 1, 2))
    EPOCHS = torch.zeros((nb_lr, nb_iter, 1, 2))

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

            ep_teleport = [ep_teleport] if isinstance(ep_teleport, int) else ep_teleport
            if data == "mnist":
                train_dl, test_dl = mnist_loaders(batch_size=128)
            else:  # cifar10
                train_dl, test_dl = cifar10_loaders(batch_size=128)


            # --- Baseline ---

            model = make_model(seed=it, device=device)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            hist_acc_tr = []
            hist_acc_te = []
            hist_loss = []

            # --- boucle d'entraînement
            start = time.time()
            for ep in range(epochs):
                acc_tr = evaluate(model, train_dl, device)
                hist_acc_tr.append(acc_tr)
                acc = evaluate(model, test_dl, device)
                hist_acc_te.append(acc)
                train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, fraction=frac)
                hist_loss.append(train_loss)
                if acc >= 0.99:
                    break
            end = time.time()
            
            LOSS[lr_index, it, :, 1] = torch.tensor(hist_loss)
            ACC_TRAIN[lr_index, it, :, 1] = torch.tensor(hist_acc_tr)
            ACC_TEST[lr_index, it, :, 1] = torch.tensor(hist_acc_te)

            TIME[lr_index, it, 0, 1] = end - start
            EPOCHS[lr_index, it, 0, 1] = ep + 1

            print(f"Finished lr_index={lr_index}, it={it}.")
            print(f" Baseline : Final train acc: {hist_acc_tr[-1]:.4f}, test acc: {hist_acc_te[-1]:.4f}, loss: {hist_loss[-1]:.4f}, epochs: {ep+1}, total time: {end - start:.2f} seconds.")

            # --- Path-Cond SGD with Teleport ---

            model = make_model(seed=it, device=device)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)

            hist_acc_tr = []
            hist_acc_te = []
            hist_loss = []

            start = time.time()
            for ep in range(epochs):
                if ep in ep_teleport:
                    start_teleport = time.time()
                    model = rescaling_path_dynamics(model, device=device, data=data)
                    end_teleport = time.time()
                    optimizer = torch.optim.SGD(model.parameters(), lr=lr) # tres important
                acc_tr = evaluate(model, train_dl, device)
                hist_acc_tr.append(acc_tr)
                acc = evaluate(model, test_dl, device)
                hist_acc_te.append(acc)
                train_loss = train_one_epoch(model, train_dl, criterion, optimizer, device, fraction=frac)
                hist_loss.append(train_loss) 
                if acc >= 0.99:
                    break
            end = time.time()

            

            
            print(f"Teleportation time: {end_teleport - start_teleport:.2f} seconds")
            print(f" PathCond : Final train acc: {hist_acc_tr[-1]:.4f}, test acc: {hist_acc_te[-1]:.4f}, loss: {hist_loss[-1]:.4f}, epochs: {ep+1}, total time: {end - start:.2f} seconds.")
            print("-----------------------------------------------------")
                    
            LOSS[lr_index, it, :, 0] = torch.tensor(hist_loss)
            ACC_TRAIN[lr_index, it, :, 0] = torch.tensor(hist_acc_tr)
            ACC_TEST[lr_index, it, :, 0] = torch.tensor(hist_acc_te)

            TIME[lr_index, it, 0, 0] = end - start
            EPOCHS[lr_index, it, 0, 0] = ep + 1

    return (
        LOSS, ACC_TRAIN, ACC_TEST, TIME, EPOCHS
    )


def rescaling_path_dynamics(model, device="cpu", data="mnist"):
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    if data == "mnist":
        inputs = torch.randn(16, 1, 28, 28).to(device)
    else:
        inputs = torch.randn(16, 3, 32, 32).to(device)
    BZ_opt, Z_opt = optimize_rescaling_polynomial(model, n_iter=1, tol=1e-6)
    final_model = reweight_model(model, BZ_opt, Z_opt).to(dtype=torch.float32, device=device)
    final_model = final_model.to(device).eval()
    model.eval()

    final_output = final_model(inputs)
    original_output = model(inputs)
    assert torch.allclose(original_output, final_output, atol=1e-5)
    # print("✅ Sortie finale préservée après rescaling.")

    return final_model
