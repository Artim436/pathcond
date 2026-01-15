from __future__ import annotations
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import time
import os

from pathcond.models import resnet18_cifar10, resnet18_mnist, MLP, UNet
from pathcond.data import moons_loaders, mnist_loaders, cifar10_loaders
from expes.enorm import ENorm
from training_utils import train_one_epoch, evaluate, rescaling_path_cond



def train_loop( 
    architecture = [8, 8], # List[int] or resnet or Unet
    method = "baseline", # baseline, pathcond, pathcond_telep_schedule, enorm, dag_enorm
    learning_rate = 0.01, # float,
    epochs = 1, # int
    seed = 0, #int
    data = "moons",
    optimizer = "sgd" # moons, mnist, fashion mnist, cifar10, cifar100, imagenet
    ):
    """
    Train 4 variants of the same model:
      - baseline SGD
      - Pathcond SGD rescaling at init
      - Pathcond rescaling every ... epochs
      - Enorm SGD
      - DAG Enorm
    """
    # --- MLflow Configuration ---
    EXPERIMENT_NAME = f"PathCond_{data}_experiments"
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Enable system metrics monitoring
    mlflow.config.enable_system_metrics_logging()
    mlflow.config.set_system_metrics_sampling_interval(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available())
    
    torch.manual_seed(seed)

    # Model selection
    # Fix: use isinstance(architecture, list) to check type correctly
    if isinstance(architecture, list):
        if data == "moons":
            model = MLP([2] + architecture + [2], seed=seed)
        elif data == "mnist":
            model = MLP([28*28] + architecture + [10], seed=seed)
        elif data == "fashionmnist":
            model = MLP([28*28] + architecture + [10], seed=seed)
    elif (architecture == "resnet" or architecture == "resnet18") and data == "mnist":
        model = resnet18_mnist(seed=seed)
    elif (architecture == "resnet" or architecture == "resnet18") and data == "cifar10":
        model = resnet18_cifar10(seed=seed)
    else:
        raise NotImplementedError("Architecture not implemented") 

    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer selection
    if optimizer == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise NotImplementedError("Optimizer not implemented")
    
    # Data Loading
    if data == "moons":
        train_dl, test_dl = moons_loaders(batch_size=128, seed=seed)
    elif data == "mnist":
        train_dl, test_dl = mnist_loaders(batch_size=128, seed=seed)
    elif data == "cifar10":
        train_dl, test_dl = cifar10_loaders(batch_size=128, seed=seed)
    else:
        raise NotImplementedError("Data not implemented")
    
    # Initial Rescaling Logic
    time_teleport_at_init = 0.0
    if method == "pathcond":
        start_init = time.time()
        model = rescaling_path_cond(model, nb_iter_optim_rescaling=10, device=device, data=data, enorm=False)
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
        time_teleport_at_init = time.time() - start_init
    elif method == "dag_enorm":
        start_init = time.time()
        model = rescaling_path_cond(model, nb_iter_optim_rescaling=10, device=device, data=data, enorm=True)
        optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
        time_teleport_at_init = time.time() - start_init
    elif method == "enorm":
        enorm = ENorm(model.named_parameters(), optim, c=1, model_type="linear")
    
    timer_method_during_training = 0.0

    params = {
        "architecture": str(architecture),
        "method": method,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "seed": seed,
        "data": data,
        "optimizer": optimizer,
    }

    # Start MLflow Tracking
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.set_tag("mlflow.runName", f'{method}_{data}_{seed}_lr{learning_rate}')

        start_training_total = time.time()
        for epoch in range(epochs):
            # Training phase
            loss = train_one_epoch(model, train_dl, criterion, optim, device)
            
            # Evaluation phase
            acc = evaluate(model, train_dl, device)
            acc_test = evaluate(model, test_dl, device)
            
            # Method specific steps during training
            if method == "enorm":
                start_step = time.time()
                enorm.step()
                timer_method_during_training += time.time() - start_step
            elif method == "pathcond_telep_schedule":
                start_step = time.time()
                model = rescaling_path_cond(model, nb_iter_optim_rescaling=10, device=device, data=data, enorm=False)
                optim = torch.optim.SGD(model.parameters(), lr=learning_rate)
                timer_method_during_training += time.time() - start_step

            # Metrics logging
            mlflow.log_metrics({
                "train_loss": loss,
                "train_acc": acc,
                "test_acc": acc_test,
                "cumulative_time": time.time() - start_training_total,
                "init_teleport_time": time_teleport_at_init,
                "training_method_overhead": timer_method_during_training
            }, step=epoch)

        # Final logs
        mlflow.log_metric("total_time", time.time() - start_training_total)
        mlflow.pytorch.log_model(model, name="model")

    return model

if __name__ == "__main__":
    train_loop()