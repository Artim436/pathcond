from __future__ import annotations
from typing import Tuple, List
import torch
import torch.nn as nn
from pathcond.models import toy_MLP, MLP, apply_init
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model
from pathcond.plot import plot_rescaling_analysis
from tqdm import tqdm



import time



def fit_with_telportation(  
    epochs: int = 5,
    hidden=2,
    ep_teleport: int = 0,
    nb_init: int = 10,
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
    LOSS = torch.zeros((nb_init, nb_iter, epochs, 2))  # sgd, ref_sgd, equinorm, extrem
    LOSS_TEST = torch.zeros((nb_init, nb_iter, epochs, 2))

    model = MLP([8]*6)
    nb_params = sum(p.numel() for p in model.parameters())
    GRAD = torch.zeros((nb_init, nb_iter, 2, nb_params))  # sgd, ref_sgd


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.randn(256, 8).to(device)

    inits = torch.logspace(-2, -1, nb_init).tolist()

    lr = 0.1




    def random_rotation_matrix(dim=8):
        # Décomposition QR d'une matrice normale
        A = torch.randn(dim, dim)
        Q, R = torch.linalg.qr(A)

        det_Q = torch.det(Q)

        d = torch.ones(dim, device=Q.device, dtype=Q.dtype)

        if det_Q < 0:
            d[-1] = -1.0
            
        Q = Q @ torch.diag(d)


        return Q
    

    A = random_rotation_matrix(dim=8).to(device)
    
    assert torch.isclose(torch.det(A), torch.tensor(1.0, device=device))

    y = inputs @ A.T




    for init_index, init in tqdm(enumerate(inits)):
        for it in range(nb_iter):

            start = time.time()
            torch.manual_seed(it)

            # --- construction des variantes de modèles/optimiseurs de façon déclarative
            def make_model(seed: int = 0, device=None, init_scale: float = init) -> MLP:
                model = MLP([8]*6, seed=seed)
                apply_init(model, scheme="normal", std=init_scale)
                return model.to(device)
            
            criterion = nn.MSELoss()

            variants = {
                "pathcond": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,  # défini juste après
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "loss_test": [],
                    "label": "sgd",
                },
                "baseline": {
                    "model": make_model(seed=it, device=device),
                    "optimizer": None,
                    "trainer": torch.optim.SGD,
                    "hist_loss": [],
                    "loss_test": [],
                    "label": "baseline",
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
                    variants["pathcond"]["model"], rescaling = rescaling_path_dynamics(
                        variants["pathcond"]["model"], verbose=False, soft=True, name="sgd", nb_iter=nb_iter_optim_rescaling, device=device
                    )
                    end_teleport = time.time()
                    # print(f"Rescaling applied in {end_teleport - start_teleport:.2f} seconds.")
                    variants["pathcond"]["optimizer"] = torch.optim.SGD(
                        variants["pathcond"]["model"].parameters(), lr=lr
                    )

                # entraînement d'un epoch pour chaque variante
                for key in ["pathcond", "baseline"]:
                    v = variants[key]
                    v["model"].to(device)
                    v["model"].train()
                    v["optimizer"].zero_grad()
                    y_pred = v["model"](inputs)
                    train_loss = criterion(y_pred, y)
                    train_loss.backward()
                    if ep == 0:
                        with torch.no_grad():
                            grad = torch.cat([param.grad.view(-1) for param in v["model"].parameters()])
                            if key == "pathcond":
                                one_on_rescaling = 1 / rescaling
                                GRAD[init_index, it, 0, :] = grad*one_on_rescaling
                            else:
                                GRAD[init_index, it, 1, :] = grad
                    v["optimizer"].step()
                    v["hist_loss"].append(train_loss.item())
                    v["model"].eval()
                    # with torch.no_grad():
                    #     X_test = torch.randn(256, 8).to(device)
                    #     Y_test = X_test @ A.T
                    #     y_test_pred = v["model"](X_test)
                    #     test_loss = criterion(y_test_pred, Y_test)
                    #     v["loss_test"].append(test_loss.item())



            # --- restitue EXACTEMENT les mêmes éléments et dans le même ordre
            loss_history_pathcond = variants["pathcond"]["hist_loss"]
            loss_history_baseline = variants["baseline"]["hist_loss"]
            LOSS[init_index, it, :, 0] = torch.tensor(loss_history_pathcond)
            LOSS[init_index, it, :, 1] = torch.tensor(loss_history_baseline)

    return LOSS, GRAD



def rescaling_path_dynamics(model, verbose: bool = False, soft: bool = True, nb_iter=1, name: str = "sgd", device="cpu") -> MNISTMLP:
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    inputs = torch.randn(8, 8).to(device)

    model.to(device)
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
        BZ_opt = optimize_neuron_rescaling_polynomial(model=model, n_iter=nb_iter, verbose=verbose, tol=1e-6).to(dtype=torch.float32)
    final_model = reweight_model(model, BZ_opt)

    rescaling = torch.exp(-BZ_opt / 2)

    # 4. Vérification de la sortie finale
    if verbose:
        print("\n4. Vérification de la préservation de la sortie finale...")
    final_output = final_model(inputs)
    torch.allclose(original_output, final_output, atol=1e-5)
    # print("✅ Sortie finale préservée après rescaling.")

    return final_model, rescaling

def rescaling_extreme(model, device="cpu") -> MNISTMLP:
    """Test de validation de la fonctionnalité de rescaling par neurone."""

    rescale_factor = 0.01
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
    return model, alpha


def equinorm_rescaling(model, device="cpu") -> MNISTMLP:
    # print("norm of weights before", torch.norm(model.get_weights_as_vector()))
    num = model.model[2].weight.data.pow(2)
    den = model.model[0].weight.data.view(-1).pow(2) + model.model[0].bias.data.view(-1).pow(2)
    res = (num/den).pow(0.25)
    rescale = res.to(device)
    model.model[0].weight.data *= rescale.view_as(model.model[0].weight.data)
    model.model[2].weight.data *= 1 / rescale.view(model.model[2].weight.shape)
    model.model[0].bias.data *= rescale.view(model.model[0].bias.shape)
    # print("norm of weights after", torch.norm(model.get_weights_as_vector()))
    return model
