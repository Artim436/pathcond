from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn as nn
from pathcond.mlp import toy_MLP
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model
from pathcond.plot import plot_rescaling_analysis
from tqdm import tqdm



import time



def fit_with_telportation(  
    epochs: int = 5,
    hidden=2,
    ep_teleport: int = 0,
    nb_lr: int = 10,
    nb_iter_optim_rescaling: int = 1,
    nb_iter: int = 1,
    frac: float = 1.0
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
    LOSS = torch.zeros((nb_lr, nb_iter, epochs, 4))  # sgd, ref_sgd, equinorm, extrem

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.tensor([[i] for i in torch.arange(-5, 5, 1)], dtype=torch.float32).to(device)


    def f_label(x):
        return -0.3*nn.ReLU()(x +0.4) + 0.5*nn.ReLU()(x - 0.4) + 0.1*nn.ReLU()(-0.3*x + +0.2) 

    labels = f_label(inputs).to(device)

    for lr_index, lr in tqdm(enumerate(learning_rates)):
        for it in range(nb_iter):

            start = time.time()
            torch.manual_seed(it)

            # --- construction des variantes de modèles/optimiseurs de façon déclarative
            def make_model(seed: int = 0, device=None):
                model = toy_MLP(d_input=1, d_hidden1=hidden, seed=seed, teacher_init=False)
                return model.to(device)
            
            criterion = nn.MSELoss()

            variants = {
                "pathcond": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,  # défini juste après
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "label": "sgd",
                },
                "baseline": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "label": "baseline",
                },
                "equinorm": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "label": "equinorm",        
            },
                "extreme": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "label": "extreme",        
            }
            }

            # initialise les optimiseurs
            for v in variants.values():
                v["optimizer"] = v["trainer"](v["model"].parameters(), lr=lr)

            ep_teleport = [ep_teleport] if isinstance(ep_teleport, int) else ep_teleport

            # --- boucle d'entraînement
            for ep in range(epochs):
                # téléportation uniquement pour le modèle principal 'sgd'
                if ep in ep_teleport:
                    start_teleport = time.time()
                    variants["pathcond"]["model"] = rescaling_path_dynamics(
                        variants["pathcond"]["model"], verbose=False, soft=True, name="sgd", nb_iter=nb_iter_optim_rescaling, device=device
                    )
                    end_teleport = time.time()
                    # print(f"Rescaling applied in {end_teleport - start_teleport:.2f} seconds.")
                    variants["pathcond"]["optimizer"] = torch.optim.SGD(
                        variants["pathcond"]["model"].parameters(), lr=lr
                    )
                    variants["extreme"]["model"], alpha = rescaling_extreme(
                        variants["extreme"]["model"], device=device
                    )
                    variants["extreme"]["optimizer"] = torch.optim.SGD(
                        variants["extreme"]["model"].parameters(), lr=lr/alpha
                    )
                    variants["equinorm"]["model"] = equinorm_rescaling(variants["equinorm"]["model"], device=device)
                    variants["equinorm"]["optimizer"] = torch.optim.SGD(
                        variants["equinorm"]["model"].parameters(), lr=lr
                    )


                # entraînement d'un epoch pour chaque variante
                for key in ["pathcond", "baseline", "equinorm", "extreme"]:
                    v = variants[key]
                    v["model"].train()
                    v["optimizer"].zero_grad()
                    logits = v["model"](inputs, device=device)
                    train_loss = criterion(logits, labels)
                    train_loss.backward()
                    v["optimizer"].step()
                    v["hist_loss"].append(train_loss.item())

            end = time.time()
            # print(f"Training completed in {end - start:.2f} seconds. (lr={lr:.4f}, iter={it})")

            # --- restitue EXACTEMENT les mêmes éléments et dans le même ordre
            loss_history_pathcond = variants["pathcond"]["hist_loss"]
            loss_history_baseline = variants["baseline"]["hist_loss"]
            loss_history_equinorm = variants["equinorm"]["hist_loss"]
            loss_history_extreme = variants["extreme"]["hist_loss"]
            LOSS[lr_index, it, :, 0] = torch.tensor(loss_history_pathcond)
            LOSS[lr_index, it, :, 1] = torch.tensor(loss_history_baseline)
            LOSS[lr_index, it, :, 2] = torch.tensor(loss_history_equinorm)
            LOSS[lr_index, it, :, 3] = torch.tensor(loss_history_extreme)
    return LOSS



def rescaling_path_dynamics(model, verbose: bool = False, soft: bool = True, nb_iter=1, name: str = "sgd", device="cpu") -> MNISTMLP:
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    inputs = torch.randn(3, 1).to(device)

    original_output = model.forward(inputs, device=device)
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
    final_model = reweight_model(model, BZ_opt).to(dtype=torch.float32)


    # 4. Vérification de la sortie finale
    if verbose:
        print("\n4. Vérification de la préservation de la sortie finale...")
    final_output = final_model.forward(inputs, device=device)
    assert torch.allclose(original_output, final_output, atol=1e-5)
    # print("✅ Sortie finale préservée après rescaling.")

    return final_model

def rescaling_extreme(model, device="cpu") -> MNISTMLP:
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    inputs = torch.randn(3, 1).to(device)
    original_output = model.forward(inputs, device=device)
    rescale_factor = 1
    # x = torch.max(torch.abs(model.model[2].weight.data[0,:]))/rescale_factor
    # x = torch.abs(model.model[2].weight.data[0,0])/rescale_factor
    x = torch.median(torch.abs(model.model[2].weight.data[0,:]))/rescale_factor
    L = []
    hidden_dim = model.model[0].weight.shape[0]
    for l in range(hidden_dim):
        L.append(torch.abs(model.model[2].weight.data[0,l]/x))
    rescale = torch.asarray(L).to(device)
    model.model[0].weight.data *= rescale.view(model.model[0].weight.shape)
    model.model[2].weight.data *= 1 / rescale.view(model.model[2].weight.shape)
    model.model[0].bias.data *= rescale.view(model.model[0].bias.shape)
    # print("v squared", model.model[2].weight.data**2)
    alpha = model.model[2].weight.data[0,0]**2
     #print("alpha", alpha)
    final_output = model.forward(inputs, device=device)
    assert torch.allclose(original_output, final_output, atol=1e-5)
    # print("✅ Sortie finale préservée après rescaling.")
    return model, alpha


def equinorm_rescaling(model, device="cpu") -> MNISTMLP:
    # print("norm of weights before", torch.norm(model.get_weights_as_vector()))
    inputs = torch.randn(3, 1).to(device)
    original_output = model.forward(inputs, device=device)
     #print("weights squared", model.model[2].weight.data**2)
    num = model.model[2].weight.data.pow(2)
    den = model.model[0].weight.data.view(-1).pow(2) + model.model[0].bias.data.view(-1).pow(2)
    res = (num/den).pow(0.25)
    rescale = res.to(device)
    model.model[0].weight.data *= rescale.view_as(model.model[0].weight.data)
    model.model[2].weight.data *= 1 / rescale.view(model.model[2].weight.shape)
    model.model[0].bias.data *= rescale.view(model.model[0].bias.shape)
    # print("norm of weights after", torch.norm(model.get_weights_as_vector()))
    final_output = model.forward(inputs, device=device)
    assert torch.allclose(original_output, final_output, atol=1e-5)
    return model
