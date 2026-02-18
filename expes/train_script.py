from __future__ import annotations
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import time
import os

from pathcond.models import resnet18_cifar10, resnet18_mnist, MLP, UNet, MLP_BN, FullyConvolutional, FullyConvolutional_without_bn
from pathcond.data import moons_loaders, mnist_loaders, cifar10_loaders
from enorm.enorm import ENorm
from training_utils import (
    train_one_epoch, evaluate, rescaling_path_cond, train_one_epoch_enc_dec, evaluate_enc_dec
)

def train_loop( 
    architecture = [8, 8], 
    method = "baseline", 
    learning_rate = 0.01, 
    epochs = 10000, 
    seed = 0, 
    data = "moons",
    optimizer = "sgd",
    experiment_name = "default_experiment",
    momentum = False,
    init_kai_normal = False,
    ):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- MLflow Configuration ---
    EXPERIMENT_NAME = f"PC_{data}_{experiment_name}"
    mlflow.set_experiment(EXPERIMENT_NAME)

    torch.manual_seed(seed)
    if isinstance(architecture, list):
        if data == "moons":
            if method.startswith("bn"):
                model = MLP_BN([2] + architecture + [2], seed=seed+123)
                if init_kai_normal:
                    model.reset_parameters()
            else:
                model = MLP([2] + architecture + [2], seed=seed+123)
                if init_kai_normal:
                    model.reset_parameters()
        elif data == "mnist":
            if method.startswith("bn"):
                model = MLP_BN([28*28] + architecture + [10], seed=seed+1234)
                if init_kai_normal:
                    model.reset_parameters()
            else:
                model = MLP([28*28] + architecture + [10], seed=seed+1234)
                if init_kai_normal:
                    model.reset_parameters()
        elif data == "mnist_enc_dec":
            if method.startswith("bn"):
                model = MLP_BN([28*28] + architecture + [28*28], seed=seed+1234)
                if init_kai_normal:
                    model.reset_parameters()
            else:
                model = MLP([28*28] + architecture + [28*28], seed=seed+1234)
                if init_kai_normal:
                    model.reset_parameters()
        elif data == "cifar10":
            if method.startswith("bn"):
                model = MLP_BN([32*32*3] + architecture + [10], seed=seed+1234)
                if init_kai_normal:
                    model.reset_parameters()
            else:
                model = MLP([32*32*3] + architecture + [10], seed=seed+1234)
                if init_kai_normal:
                    model.reset_parameters()
    elif architecture in ["resnet", "resnet18"]:
        if data == "mnist": model = resnet18_mnist(seed=seed+1234)
        elif data == "cifar10": model = resnet18_cifar10(seed=seed+1234)
    elif architecture == "full_conv" or architecture == "full_conv_big_batch":
        model = FullyConvolutional(bias=True)
        assert data == "cifar10", "Full Conv only implemented for CIFAR-10"
    elif architecture == "full_conv_no_bn":
        model = FullyConvolutional_without_bn(bias=True)
        assert data == "cifar10", "Full Conv without BN only implemented for CIFAR-10"
    elif architecture == "resnet18_c":
        from pathcond.models import resnet18_c_cifar10
        model = resnet18_c_cifar10(seed=seed+1234)
        assert data == "cifar10", "ResNet18 without BN only implemented for CIFAR-10"
    else:
        raise NotImplementedError("Architecture not implemented")
    
    model.to(device)
    if data == "mnist_enc_dec":
        criterion = nn.MSELoss()
    else:
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
        if architecture == "full_conv_big_batch":
            train_dl, test_dl = cifar10_loaders(batch_size=256, seed=seed)
        else:
            train_dl, test_dl = cifar10_loaders(batch_size=128, seed=seed)
    elif data == "mnist_enc_dec":
        train_dl, test_dl = mnist_loaders(batch_size=128, seed=seed)
    
    # --- Initialisation de l'Optimiseur ---
    def get_optimizer(model_params):
        if optimizer == "sgd":
            if momentum:
                return torch.optim.SGD(model_params, lr=learning_rate, momentum=0.9)
            return torch.optim.SGD(model_params, lr=learning_rate)
        elif optimizer == "adam":
            return torch.optim.Adam(model_params, lr=learning_rate)
        elif optimizer == "muon":
            return torch.optim.Muon(model_params, lr=learning_rate)
        raise NotImplementedError()

    optim = get_optimizer(model.parameters())

    

    with mlflow.start_run():
        mlflow.log_params({"architecture": str(architecture), "method": method, "lr": learning_rate, "epochs": epochs, "data": data})
        mlflow.set_tag("mlflow.runName", f'{method}_{data}_{seed}_lr{learning_rate}')
        mlflow.pytorch.log_model(model, name="model_initial")

        if method.startswith("bn_"):
            method = method[3:]
    
        start_training_total = time.time()
        time_teleport_at_init = 0.0
        # --- Rescaling Initial ---
        if method in ["pathcond", "dag_enorm", "pathcond_telep_schedule"]:
            start_init = time.time()
            is_enorm = (method == "dag_enorm")
            is_resnet_c = (architecture == "resnet18_c")
            model, rescaling_on_hidden_neurons = rescaling_path_cond(model, nb_iter_optim_rescaling=10, device=device, data=data, enorm=is_enorm, is_resnet_c=is_resnet_c)
            time_teleport_at_init = time.time() - start_init
            optim = get_optimizer(model.parameters())
            mlflow.log_param("rescaling_on_hidden_neurons_init", rescaling_on_hidden_neurons)
            mlflow.log_param("max_rescaling_on_hidden_neurons_init", torch.max(torch.abs(rescaling_on_hidden_neurons)).item())
        elif method == "enorm":
            if architecture in ["resnet", "resnet18", "full_conv", "full_conv_no_bn", "full_conv_no_bn_bias", "full_conv_big_batch"]:
                enorm_obj = ENorm(model.named_parameters(), optim, model_type="conv")
            else:
                enorm_obj = ENorm(model.named_parameters(), optim, model_type="linear")
        
        timer_method_during_training = 0.0
        timer_forward = 0.0
        timer_backward = 0.0

    
        
        start_epoch = time.time()
        # Avant la boucle, on initialise les événements (uniquement si cuda)
        # --- Initialisation (avant la boucle) ---
        if device == "cuda":
            # On crée les événements une seule fois
            start_fwd = torch.cuda.Event(enable_timing=True)
            end_fwd = torch.cuda.Event(enable_timing=True)
            start_bwd = torch.cuda.Event(enable_timing=True)
            end_bwd = torch.cuda.Event(enable_timing=True)

        # Compteurs pour la moyenne
        nb_measures = 0

        for epoch in range(epochs):
            # On ne mesure le timing que lors des phases de log pour ne pas ralentir le reste
            should_measure = (epoch+1) % 1000 == 0 and (device == "cuda") and (data == "moons")

            if data == "moons":
                model.train()
                optim.zero_grad()

                if should_measure:
                    # --- Forward avec mesure ---
                    start_fwd.record()
                    loss = criterion(model(X_train), y_train)
                    end_fwd.record()

                    # --- Backward avec mesure ---
                    start_bwd.record()
                    loss.backward()
                    end_bwd.record()
                    
                    # On synchronise uniquement quand on mesure
                    torch.cuda.synchronize()
                    timer_forward += start_fwd.elapsed_time(end_fwd) / 1000.0
                    timer_backward += start_bwd.elapsed_time(end_bwd) / 1000.0
                    nb_measures += 1
                else:
                    # --- Entraînement normal (vitesse maximale) ---
                    loss = criterion(model(X_train), y_train)
                    loss.backward()
                
                optim.step()
            elif data in ["mnist", "cifar10"]:
                loss = train_one_epoch(model, train_dl, criterion, optim, device)
            elif data == "mnist_enc_dec":
                loss, timer_forward, timer_backward = train_one_epoch_enc_dec(model, train_dl, criterion, optim, device)

            # Méthodes spécifiques (Schedule)
            if method == "enorm":
                start_step = time.time()
                enorm_obj.step()
                timer_method_during_training += time.time() - start_step
            
            if data == "mnist_enc_dec" or data == "mnist":
                telep_schedule_epoch = 50 #500 epochs in total
            elif data == "cifar10":
                telep_schedule_epoch = 20 #60 epochs in total
            if method == "pathcond_telep_schedule" and epoch % telep_schedule_epoch == 0 and epoch > 0:
                start_step = time.time()
                model, resc_during_training = rescaling_path_cond(model, nb_iter_optim_rescaling=1, device=device, data=data, enorm=False)
                optim = get_optimizer(model.parameters())
                timer_method_during_training += time.time() - start_step
                mlflow.log_param(f"rescaling_on_hidden_neurons_epoch_{epoch}", torch.max(torch.abs(resc_during_training)))


            # Metrics (tous les 50 epochs pour garder de la fluidité dans l'UI)
            if data == "moons":
                if epoch % 10 == 0 or epoch == epochs - 1:
                    model.eval()
                    pred = model(X_train).argmax(dim=1)
                    acc = (pred == y_train).sum().item() / y_train.size(0)
                    pred = model(X_test).argmax(dim=1)
                    acc_test = (pred == y_test).sum().item() / y_test.size(0)
                    current_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
                    mlflow.log_metrics({
                        "train_loss": current_loss,
                        "train_acc": acc,
                        "test_acc": acc_test,
                    }, step=epoch)
            else:
                current_loss = loss.item() if isinstance(loss, torch.Tensor) else loss
                if data in ["mnist", "cifar10"]:
                    acc = evaluate(model, train_dl, device)
                    acc_test = evaluate(model, test_dl, device)
                    mlflow.log_metrics({
                        "train_acc": acc,
                        "test_acc": acc_test,
                        "train_loss": current_loss
                    }, step=epoch)
                elif data == "mnist_enc_dec":
                    rec_loss_test = evaluate_enc_dec(model, test_dl, device)
                    mlflow.log_metrics({
                        "test_rec_loss": rec_loss_test,
                        "train_loss": current_loss
                    }, step=epoch)


        mlflow.log_metric("total_time", time.time() - start_training_total)
        mlflow.log_metric("time_per_epoch", (time.time() - start_epoch) / epochs)
        mlflow.log_metric("overhead_time", timer_method_during_training)
        avg_fwd = timer_forward / nb_measures if nb_measures > 0 else timer_forward / epochs
        avg_bwd = timer_backward / nb_measures if nb_measures > 0 else timer_backward / epochs
        mlflow.log_metric("mean_forward_time_cuda", avg_fwd)
        mlflow.log_metric("mean_backward_time_cuda", avg_bwd)
        mlflow.log_metric("time_teleport_at_init", time_teleport_at_init)
        

        # Log du modèle uniquement à la fin
        mlflow.pytorch.log_model(model, name="model_final")

        n_params = sum(p.numel() for p in model.parameters())
        mlflow.log_param("n_params", n_params)
    return model

if __name__ == "__main__":
    train_loop(method="pathcond", architecture=[32,32,32], learning_rate=0.001, epochs=60, seed=0, data="moons", experiment_name="resnet18c_pathcond_cifar10")