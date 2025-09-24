import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from utils import _param_start_offsets



def grad_path_norm(model, device="cpu") -> torch.Tensor:
    inputs = torch.ones(1, 1, model.model[0].in_features)  # Dummy input for G computation
    def fct(model, inputs, device="cpu"):
        return model.forward(inputs, device=device).sum()
    grad = torch.autograd.grad(fct(model, inputs, device=device), model.parameters())
    grad = [g.view(-1) for g in grad]  # Aplatir les gradients
    return torch.cat(grad)  # Concaténer les gradients en un seul tenseur




def set_weights_for_path_norm(
    model, exponent=1, provide_original_weights=False
):
    """
    Applies $w\\mapsto |w|^{exponent}$ to all weights $w$ of the model
    for path-norm computation. It handles cases where the weights of the
    convolutional and linear layers are pruned with the torch.nn.utils.prune
    library (but not when their biases are pruned).

    Args:
        model (torch.nn.Module): Input model.
        exponent (float, optional): Exponent for weight transformation.
        Defaults to 1.
        provide_original_weights (bool, optional): If True, provide the
        original weights for resetting later. Defaults to True.

    Returns:
        dict: Original weights of the model if
        `provide_original_weights` is True; otherwise, empty dict.
    """
    # If a module is pruned, its original weights
    # are in weight_orig instead of weight.
    orig_weights = {}
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
            if provide_original_weights:
                orig_weights[n + ".weight"] = m.weight.detach().clone()
            m.weight.data = torch.abs(m.weight.detach())
            if exponent != 1:
                m.weight.data = torch.pow(m.weight.detach(), exponent)
            if m.bias is not None:
                if provide_original_weights:
                    orig_weights[n + ".bias"] = m.bias.detach().clone()
                m.bias.data = torch.abs(m.bias.detach())
                if exponent != 1:
                    m.bias.data = torch.pow(m.bias.detach(), exponent)
        elif isinstance(m, torch.nn.BatchNorm2d):
            if provide_original_weights:
                orig_weights[n + ".weight"] = m.weight.detach().clone()
                orig_weights[n + ".bias"] = m.bias.detach().clone()
                orig_weights[n + ".running_mean"] = (
                    m.running_mean.detach().clone()
                )
                orig_weights[n + ".running_var"] = (
                    m.running_var.detach().clone()
                )
            m.weight.data = torch.abs(m.weight.detach())
            m.bias.data = torch.abs(m.bias.detach())
            m.running_mean.data = torch.abs(m.running_mean.detach())
            # Running_var already non-negative,
            # no need to put it in absolute value

            if exponent != 1:
                m.weight.data = torch.pow(m.weight.detach(), exponent)
                m.bias.data = torch.pow(m.bias.detach(), exponent)
                m.running_mean.data = torch.pow(
                    m.running_mean.detach(), exponent
                )
                m.running_var.data = torch.pow(
                    m.running_var.detach(), exponent
                )
    return orig_weights

def reset_model(model, orig_weights):
    """
    Reset weights and maxpool layer of a model.

    Args:
        name (str): Name of the model.
        model (torch.nn.Module): Input model.
        orig_weights (dict): Original weights of the model.
    """
    for n, m in model.named_modules():
        if (
                isinstance(m, torch.nn.Conv2d) or
                isinstance(m, torch.nn.Linear)
        ):
            m.weight.data = orig_weights[n + ".weight"]
            if m.bias is not None:
                m.bias.data = orig_weights[n + ".bias"]
        elif isinstance(m, torch.nn.BatchNorm2d):
            m.weight.data = orig_weights[n + ".weight"]
            m.bias.data = orig_weights[n + ".bias"]
            m.running_mean.data = orig_weights[n + ".running_mean"]
            m.running_var.data = orig_weights[n + ".running_var"]


def compute_diag_G(model, eps: float = 1e-12):
    orig_w = set_weights_for_path_norm(model, exponent=2, provide_original_weights=True)
    res = grad_path_norm(model, device=next(model.parameters()).device)
    reset_model(model, orig_w)
    return res

def apply_neuron_rescaling_mlp(model, layer_idx, neuron_idx, lamda) -> nn.Module:
    """
    Applique un rescaling par neurone à une couche spécifique.
    
    Args:
        model: Le modèle PyTorch
        layer_idx: Index de la couche à rescaler
        neuron_idx: Index du neurone dans la couche
        lamda: Facteur de rescaling (float)"""
    model_copy = copy.deepcopy(model)
    
    # Indices des couches linéaires dans le Sequential
    linear_indices = [i for i, layer in enumerate(model.model) if isinstance(layer, nn.Linear)]
    
    if layer_idx >= len(linear_indices):
        raise ValueError(f"layer_idx doit être < {len(linear_indices)}")
    
    actual_idx = linear_indices[layer_idx]
    layer = model_copy.model[actual_idx]
    
    lamda = float(lamda)
    
    # Rescaling des poids et biais par neurone
    with torch.no_grad():
        # Chaque ligne de weight correspond à un neurone de sortie
        # On multiplie chaque ligne par le lambda correspondant
        layer.weight.data[neuron_idx, :] *= lamda
        
        # Rescaling du biais si présent
        if layer.bias is not None:
            layer.bias.data[neuron_idx] *= lamda
    
    # Si ce n'est pas la dernière couche, ajuster la couche suivante
    if layer_idx < len(linear_indices) - 1:
        next_idx = linear_indices[layer_idx + 1]
        next_layer = model_copy.model[next_idx]
        with torch.no_grad():
            # Chaque colonne de la couche suivante correspond à un neurone d'entrée
            # On divise chaque colonne par le lambda correspondant
            next_layer.weight.data[:, neuron_idx] /= lamda
    
    return model_copy


def optimize_neuron_rescaling_polynomial(model, n_iter=10, tol=1e-6, verbose=False) -> torch.Tensor:
    """
    Optimize per-hidden-neuron log-rescalings Z via a per-neuron quadratic, using
    incremental updates to avoid recomputing B @ Z from scratch at every step.

    Returns
    -------
    Z : torch.Tensor [n_hidden_neurons], dtype=float32 on model's device
        Log-rescaling values per hidden neuron.
    """
    # --- Setup: device/dtype and network structure ---
    device = next(model.parameters()).device
    dtype = torch.float32

    # Collect linear layers; exclude the final (output) layer from hidden count
    linear_indices = [i for i, layer in enumerate(model.model) if isinstance(layer, nn.Linear)]
    n_params = sum(p.numel() for p in model.parameters())
    n_params_tensor = torch.tensor(n_params, dtype=dtype, device=device)
    n_hidden_neurons = sum(model.model[i].out_features for i in linear_indices[:-1])

    # Parameters to optimize
    Z = torch.zeros(n_hidden_neurons, dtype=dtype, device=device)

    # Problem-specific matrices/vectors
    B = compute_matrix_B(model).to(device=device, dtype=dtype)     # shape: [m, n_hidden_neurons]
    diag_G = compute_diag_G(model).to(device=device, dtype=dtype)  # shape: [m], elementwise factor

    # Maintain BZ incrementally: BZ = B @ Z
    BZ = torch.zeros(n_params, dtype=dtype, device=device)    
    OBJ = [function_F(n_params, BZ, diag_G).item()]
    print(f"Initial obj: {OBJ[0]:.6f}")
    for k in range(n_iter):
        delta_total = 0.0
        y_bar = 0.0  # Track max for numerical stability (not strictly necessary)
        for h in range(n_hidden_neurons):
            # Column for neuron h
            b_h = B[:, h]  # shape: [m]

            # Partition indices (must return index sets for rows of B/diag_G)
            in_h, out_h, other_h = compute_in_out_other_h(b_h)

            # Ensure torch index tensors on the right device
            in_h_t    = torch.as_tensor(in_h,    device=device, dtype=torch.long)
            out_h_t   = torch.as_tensor(out_h,   device=device, dtype=torch.long)
            other_h_t = torch.as_tensor(other_h, device=device, dtype=torch.long)

            card_in_h  = int(in_h_t.numel())
            card_out_h = int(out_h_t.numel())
            # Leave-one-out energy vector: exp( (B @ Z) - b_h * Z[h] ) * diag_G
            # Using the maintained BZ avoids a full matmul here.
            Y_h = BZ - b_h * Z[h]  # shape: [m]
            E = torch.exp(Y_h - y_bar) * diag_G  # shape: [m]

            # Polynomial coefficients components
            # A_h is scalar (int), others are sums over selected rows of E
            A_h = (card_in_h - card_out_h)
            B_h = E[out_h_t].sum()
            
            C_h = E[in_h_t].sum()
            D_h = E[other_h_t].sum()


            # Polynomial: P(X) = a*X^2 + b*X + c where
            a = B_h * (A_h + n_params_tensor)
            b = D_h * A_h
            c = C_h * (A_h - n_params_tensor)

            if a <= 0.0:
                raise ValueError(f"Non-positive a={a} in quadratic for neuron {h} at iter {k}, A_h={A_h}, B_h={B_h}, C_h={C_h}, D_h={D_h}")
            if c >= 0.0:
                raise ValueError(f"Non-negative c={c} in quadratic for neuron {h} at iter {k}, A_h={A_h}, B_h={B_h}, C_h={C_h}, D_h={D_h}")


            # Degenerate to linear if a ~ 0
            if abs(a) < 1e-20:
                if abs(b) >= 1e-20:
                    x = -c / b
                    if x > 0.0:
                        z_new = torch.log(x)
                    else:
                        raise ValueError(f"Non-positive root {x} in linear case for neuron {h} at iter {k}, a={a}, b={b}, c={c}")
                else:
                    if abs(c) < 1e-20:
                        raise ValueError(f"a = {a}, b = {b}, c = {c} all ~ 0 for neuron {h} at iter {k}")
                    else:
                        raise ValueError(f"a = {a}, b = {b} both ~ 0 but c = {c} != 0 for neuron {h} at iter {k}")
            else:
                disc = torch.square(b) - 4.0 * a * c
                if disc > 0.0:
                    sqrt_disc = torch.sqrt(disc)
                    x1 = (-b + sqrt_disc) / (2.0 * a)
                    x2 = (-b - sqrt_disc) / (2.0 * a)
                    candidates = [x for x in (x1, x2) if x > 0.0]
                    if len(candidates) != 1:
                        print("candidates:", candidates, x1, x2, a, b, c)
                        raise ValueError(f"Unexpected number of positive roots {len(candidates)} for neuron {h} at iter {k}")
                    z_new = torch.log(candidates[0])
                else:
                    raise ValueError(f"Negative or infinit discriminant {disc} in quadratic for neuron {h} at iter {k}")
            # Update Z[h] and incrementally refresh BZ
            delta = z_new - float(Z[h])
            if delta != 0.0:
                delta_total += abs(delta)
                BZ = BZ + b_h * delta  # rank-1 update instead of recomputing B @ Z
                if abs(z_new) > abs(Z[h]):
                    if z_new > y_bar:
                        y_bar = z_new
                Z[h] = z_new
                if verbose:
                    obj = function_F(n_params, BZ, diag_G).item()
                    OBJ.append(obj)
        if delta_total < tol:
            print(f"Converged after {k+1} iterations (delta_total={delta_total:.6e} < tol={tol})")
            break
    alpha = n_params/torch.sum(torch.exp(BZ) * diag_G).item()
    obj = function_F(n_params, BZ, diag_G).item()
    print(f"Final obj: {obj:.6f}, alpha: {alpha:.6f}")
    if verbose:
        return BZ, Z, alpha, OBJ
    return BZ

def function_F(n, BZ, dG):
    first = n*(torch.logsumexp(BZ + torch.log(dG), axis=0) - math.log(n))
    second = torch.sum(BZ) 
    return first - second





def compute_matrix_B(model: nn.Module) -> torch.Tensor:
    """
    Calcule la matrice B pour tous les neurones cachés du modèle.
    B = [b_1, b_2, ..., b_H] où b_i est le vecteur b pour le i-ème neurone caché.

    Args:
        model: Le modèle PyTorch (avec des nn.Linear chaînés)

    Returns:
        B: torch.int8 de taille [n_params, n_hidden_neurons]
           (+1 sur les poids sortants vers la couche suivante, -1 sur les poids/biais entrants,
            0 ailleurs)
    """
    # Récupération des couches linéaires dans l'ordre d'application
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) < 2:
        raise ValueError("Le modèle doit contenir au moins deux nn.Linear pour avoir des neurones cachés.")

    # Totaux
    starts, n_params = _param_start_offsets(model)
    n_hidden = sum(layer.out_features for layer in linear_layers[:-1])  # exclut la dernière couche (non cachée)
    device = next(model.parameters()).device

    # Matrice résultat
    B = torch.zeros((n_params, n_hidden), dtype=torch.int8, device=device)

    # Remplit B couche par couche (vectorisé)
    col_start = 0
    for l in range(len(linear_layers) - 1):
        layer = linear_layers[l]
        next_layer = linear_layers[l + 1]

        o = layer.out_features
        i = layer.in_features
        on, inn = next_layer.weight.shape  # [out_next, in_next]

        if inn != o:
            raise ValueError(
                f"Mismatch structure: next_layer.in_features={inn} != layer.out_features={o}."
            )

        col_end = col_start + o  # cette couche contribue 'o' colonnes (une par neurone)

        # --- (1) Poids entrants de la couche courante : -1 sur la ligne 'neuron' de W0 ---
        w0 = layer.weight  # [o, i]
        w0_start, w0_end = starts[id(w0)]
        # bloc (o*i, o) correspondant à tout W0 (aplati) x colonnes de cette couche dans B
        block_in = B[w0_start:w0_start + o * i, col_start:col_end]  # shape (o*i, o)
        # Construire un motif (o, i, o) où pour chaque neurone k: T[k, :, k] = -1
        T = torch.zeros((o, i, o), dtype=torch.int8, device=device)
        idx = torch.arange(o, device=device)
        T[idx, :, idx] = -1
        block_in.copy_(T.view(o * i, o))

        # --- (2) Biais de la couche courante : -1 sur la diagonale ---
        if layer.bias is not None:
            b0 = layer.bias  # [o]
            b0_start, b0_end = starts[id(b0)]
            block_bias = B[b0_start:b0_start + o, col_start:col_end]  # shape (o, o)
            block_bias.copy_(-torch.eye(o, dtype=torch.int8, device=device))

        # --- (3) Poids sortants vers la couche suivante : +1 sur la colonne 'neuron' de W1 ---
        w1 = next_layer.weight  # [on, inn] avec inn == o
        w1_start, w1_end = starts[id(w1)]
        block_out = B[w1_start:w1_start + on * inn, col_start:col_end]  # shape (on*o, o)
        # Motif: répéter une matrice identité (o x o) 'on' fois verticalement
        M = torch.eye(o, dtype=torch.int8, device=device).repeat(on, 1)  # (on*o, o)
        block_out.copy_(M)

        # Avancer le curseur de colonnes
        col_start = col_end

    return B


def compute_in_out_other_h(vect_b):
    """
    Calcule les indices des poids entrants et sortants
    d'un neurone caché donné.
    Args:
        vect_b: vecteur b (torch.Tensor de taille [n_params])
    Returns:
        in_h: indices des poids entrants (torch.Tensor de taille [n_in_h])
        out_h: indices des poids sortants (torch.Tensor de taille [n_out_h])
        other_h: indices des autres poids (torch.Tensor de taille [n_other])
    """
    in_h = torch.where(vect_b == -1)[0]
    out_h = torch.where(vect_b == 1)[0]
    other_h = torch.where(vect_b == 0)[0]
    return in_h, out_h, other_h


def reweight_model(model: nn.Module, BZ: torch.Tensor) -> nn.Module:
    """
    Reweight a model according to a log-rescaling vector BZ.

    Args:
        model (nn.Module): Pytorch model.
        BZ (torch.Tensor): log-rescaling vector of size [n_params].
    
    Returns:
        nn.Module: Reweighted model.
    """
    # Copie du modèle (mêmes poids, pas de lien mémoire)
    new_model = copy.deepcopy(model)

    # Vérification
    total_params = sum(p.numel() for p in model.parameters())
    assert BZ.numel() == total_params, \
        f"Taille de BZ {BZ.numel()} incompatible avec {total_params} paramètres"

    # On va parcourir les paramètres
    idx = 0
    for p in new_model.parameters():
        numel = p.numel()
        # On reshape la portion correspondante de BZ
        bz_chunk = BZ[idx:idx+numel].view_as(p.data)
        idx += numel
        # Multiplication
        p.data = p.data * torch.exp(-0.5 * bz_chunk)
    return new_model




def hessian_2(model, inputs):
    """
    Calcule la Hessienne complète de la fonction scalaire
    f(model) = model.forward_2(inputs).sum() 
    par rapport à tous les paramètres du modèle.
    """
    # Étape 1 : fonction scalaire
    def f(model, inputs):
        return model.forward_squared(inputs).sum()

    # Étape 2 : premier gradient ∇f
    grad = torch.autograd.grad(f(model, inputs), model.parameters(), create_graph=True)
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad])

    # Étape 3 : calcul ligne par ligne de la Hessienne
    hessian_rows = []
    print("Computing Hessian...")
    for i in tqdm(range(grad_vec.numel())):
        grad2 = torch.autograd.grad(grad_vec[i], model.parameters(), retain_graph=True)
        row = torch.cat([g.contiguous().view(-1) for g in grad2])
        hessian_rows.append(row)

    # Étape 4 : empilement en matrice
    hessian = torch.stack(hessian_rows)

    return hessian


def compute_G_matrix(model) -> torch.Tensor:
    """
    Calcule la matrice G pour le modèle donné et les entrées.
    G_{ij} = H_{ij}/4 si i ≠ j
    G_{ii} = H_{ii}/2 si i = j
    """
    inputs = torch.ones(1, 1, model.model[0].in_features)  # Dummy input for G computation
    hessian = hessian_2(model, inputs)  # supposé renvoyer un tenseur carré (H)
    
    # Copie pour ne pas modifier H
    G = hessian.clone()

    # Division des diagonales par 2
    diag_indices = torch.arange(G.shape[0], device=G.device)
    G[diag_indices, diag_indices] = G[diag_indices, diag_indices] / 2.0

    # Division du reste par 4
    off_diag_mask = ~torch.eye(G.shape[0], dtype=bool, device=G.device) # True hors daig et Flase sur la diag
    G[off_diag_mask] = G[off_diag_mask] / 4.0

    return G
