from __future__ import annotations
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import time
import os

from pathcond.models import resnet18_cifar10, resnet18_mnist, MLP, UNet
from pathcond.data import moons_loaders, mnist_loaders, cifar10_loaders
from enorm.enorm import ENorm
from training_utils import (
    train_one_epoch, evaluate, rescaling_path_cond, 
    train_one_epoch_full_batch, evaluate_full_batch
)

def train_loop( 
    architecture = [8, 8], 
    method = "baseline", 
    learning_rate = 0.01, 
    epochs = 10000, 
    seed = 0, 
    data = "moons",
    optimizer = "sgd"
    ):
    
    # --- Configuration du Device ---
    # Sur Moons (petit dataset), le CPU est souvent plus rapide car on évite la latence PCIe
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- MLflow Configuration ---
    EXPERIMENT_NAME = f"PathCond_{data}_experiments"
    mlflow.set_experiment(EXPERIMENT_NAME)
    # os.environ["MLFLOW_TRACKING_ASYNC"] = "true" # Mode asynchrone pour la vitesse

    torch.manual_seed(seed)

    # --- Sélection du Modèle ---
    if isinstance(architecture, list):
        if data == "moons":
            model = MLP([2] + architecture + [2], seed=seed+123)
        elif data == "mnist":
            model = MLP([28*28] + architecture + [10], seed=seed+1234)
        elif data == "fashionmnist":
            model = MLP([28*28] + architecture + [10], seed=seed+1234)
    elif architecture in ["resnet", "resnet18"]:
        if data == "mnist": model = resnet18_mnist(seed=seed+1234)
        elif data == "cifar10": model = resnet18_cifar10(seed=seed+1234)
    else:
        raise NotImplementedError("Architecture not implemented") 

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # --- Chargement des Données ---
    if data == "moons":
        # On déplace les tenseurs sur le device UNE SEULE FOIS ici
        X_train, X_test, y_train, y_test = moons_loaders(seed=seed)
        X_train, X_test = X_train.to(device), X_test.to(device)
        y_train, y_test = y_train.to(device), y_test.to(device)
    elif data == "mnist":
        train_dl, test_dl = mnist_loaders(batch_size=128, seed=seed)
    elif data == "cifar10":
        train_dl, test_dl = cifar10_loaders(batch_size=128, seed=seed)
    
    # --- Initialisation de l'Optimiseur ---
    def get_optimizer(model_params):
        if optimizer == "sgd":
            return torch.optim.SGD(model_params, lr=learning_rate)
        elif optimizer == "adam":
            return torch.optim.Adam(model_params, lr=learning_rate)
        raise NotImplementedError()

    optim = get_optimizer(model.parameters())
    
    # --- Rescaling Initial ---
    time_teleport_at_init = 0.0
    if method in ["pathcond", "dag_enorm"]:
        start_init = time.time()
        is_enorm = (method == "dag_enorm")
        model = rescaling_path_cond(model, nb_iter_optim_rescaling=10, device=device, data=data, enorm=is_enorm)
        optim = get_optimizer(model.parameters())
        time_teleport_at_init = time.time() - start_init
    elif method == "enorm":
        enorm_obj = ENorm(model.named_parameters(), optim, c=1, model_type="linear")
    
    timer_method_during_training = 0.0

    with mlflow.start_run():
        mlflow.log_params({"architecture": str(architecture), "method": method, "lr": learning_rate, "epochs": epochs, "data": data})
        mlflow.set_tag("mlflow.runName", f'{method}_{data}_{seed}_lr{learning_rate}')

        start_training_total = time.time()
        for epoch in range(epochs):
            # Training
            if data == "moons":
                model.train()
                optim.zero_grad()
                loss = criterion(model(X_train), y_train)
                loss.backward()
                optim.step()
            else:
                loss = train_one_epoch(model, train_dl, criterion, optim, device)

            # Méthodes spécifiques (Schedule)
            if method == "enorm":
                start_step = time.time()
                enorm_obj.step()
                timer_method_during_training += time.time() - start_step
            elif method == "pathcond_telep_schedule":
                start_step = time.time()
                model = rescaling_path_cond(model, nb_iter_optim_rescaling=10, device=device, data=data, enorm=False)
                optim = get_optimizer(model.parameters())
                timer_method_during_training += time.time() - start_step

            # Metrics (tous les 50 epochs pour garder de la fluidité dans l'UI)
            if epoch % 50 == 0 or epoch == epochs - 1:
                if data == "moons":
                    model.eval()
                    pred = model(X_train).argmax(dim=1)
                    acc = (pred == y_train).sum().item() / y_train.size(0)
                    pred = model(X_test).argmax(dim=1)
                    acc_test = (pred == y_test).sum().item() / y_test.size(0)
                else:
                    acc = evaluate(model, train_dl, device)
                    acc_test = evaluate(model, test_dl, device)
                
                mlflow.log_metrics({
                    "train_loss": loss,
                    "train_acc": acc,
                    "test_acc": acc_test,
                    "overhead_time": timer_method_during_training,
                    "teleport_time_init": time_teleport_at_init
                }, step=epoch)

        mlflow.log_metric("total_time", time.time() - start_training_total)
        # Log du modèle uniquement à la fin
        # mlflow.pytorch.log_model(model, name="model")

        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("n_params", n_params)
    return model

if __name__ == "__main__":
    train_loop()