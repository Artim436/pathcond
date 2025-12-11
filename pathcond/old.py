import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from pathcond.utils import _param_start_offsets
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from pathcond.utils import split_sorted_by_column


def optimize_neuron_rescaling_polynomial(model, n_iter=10, tol=1e-6, verbose=False, reg=None) -> torch.Tensor:
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
    dtype = torch.double
    dtype = torch.double

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
    if reg is not None:
        diag_G += reg

    # Maintain BZ incrementally: BZ = B @ Z
    BZ = torch.zeros(n_params, dtype=dtype, device=device)
    OBJ = [function_F(n_params, BZ, diag_G).item()]
    # print(f"Initial obj: {OBJ[0]:.6f}")
    for k in range(n_iter):
        delta_total = 0.0
        y_bar = 0.0  # Track max for numerical stability (not strictly necessary)
        for h in range(n_hidden_neurons):
            # Column for neuron h
            b_h = B[:, h]  # shape: [m]

            # Partition indices (must return index sets for rows of B/diag_G)
            in_h, out_h, other_h = compute_in_out_other_h(b_h)

            # Ensure torch index tensors on the right device
            in_h_t = torch.as_tensor(in_h,    device=device, dtype=torch.long)
            out_h_t = torch.as_tensor(out_h,   device=device, dtype=torch.long)
            other_h_t = torch.as_tensor(other_h, device=device, dtype=torch.long)

            card_in_h = int(in_h_t.numel())
            card_out_h = int(out_h_t.numel())
            # Leave-one-out energy vector: exp( (B @ Z) - b_h * Z[h] ) * diag_G
            # Using the maintained BZ avoids a full matmul here.
            Y_h = BZ - b_h * Z[h]  # shape: [m]
            y_bar = Y_h.max()
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
                raise ValueError(
                    f"Non-positive a={a} in quadratic for neuron {h} at iter {k}, A_h={A_h}, B_h={B_h}, C_h={C_h}, D_h={D_h}")
            if c >= 0.0:
                raise ValueError(
                    f"Non-negative c={c} in quadratic for neuron {h} at iter {k}, A_h={A_h}, B_h={B_h}, C_h={C_h}, D_h={D_h}")

            # Degenerate to linear if a ~ 0
            if abs(a) < 1e-30:
                if abs(b) >= 1e-30:
                    x = -c / b
                    if x > 0.0:
                        z_new = torch.log(x)
                    else:
                        raise ValueError(
                            f"Non-positive root {x} in linear case for neuron {h} at iter {k}, a={a}, b={b}, c={c}")
                else:
                    if abs(c) < 1e-30:
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
                        raise ValueError(
                            f"Unexpected number of positive roots {len(candidates)} for neuron {h} at iter {k}")
                    z_new = torch.log(candidates[0])
                else:
                    raise ValueError(f"Negative or infinit discriminant {disc} in quadratic for neuron {h} at iter {k}")
            # Update Z[h] and incrementally refresh BZ
            delta = z_new - float(Z[h])
            if delta != 0.0:
                delta_total += abs(delta)
                BZ = BZ + b_h * delta  # rank-1 update instead of recomputing B @ Z
                # if abs(z_new) > abs(Z[h]):
                #     if z_new > y_bar:
                #         y_bar = z_new
                Z[h] = z_new
                if verbose:
                    obj = function_F(n_params, BZ, diag_G).item()
                    OBJ.append(obj)
        if delta_total < tol:
            # print(f"Converged after {k+1} iterations (delta_total={delta_total:.6e} < tol={tol})")
            break
    alpha = n_params/torch.sum(torch.exp(BZ) * diag_G).item()
    obj = function_F(n_params, BZ, diag_G).item()
    # print(f"Final obj: {obj:.6f}, alpha: {alpha:.6f}")
    if verbose:
        return BZ, Z, alpha, OBJ
    return BZ


def optimize_rescaling_gd(model,
                          lr=1e-2,
                          n_iter=100,
                          optimizer='SGD',
                          tol=1e-6,
                          verbose=False,
                          to_log=False):

    OPTIMIZERS = {'SGD': torch.optim.SGD,
                  'Adam': torch.optim.Adam}
    if to_log:
        log = {}
        losses = []

    def loss(z, g, B):
        # B is n times H
        Bz = B @ z
        n = B.shape[0]
        if torch.all(g > 0):
            return n*torch.logsumexp(torch.log(g) + Bz, 0) - Bz.sum()
        else:
            v = g*torch.exp(Bz)
            return n*torch.log(v.sum()) - Bz.sum()
            return n*torch.log(v.sum()) - Bz.sum()

    device = next(model.parameters()).device
    dtype = torch.float32

    # Collect linear layers; exclude the final (output) layer from hidden count
    linear_indices = [i for i, layer in enumerate(model.model) if isinstance(layer, nn.Linear)]
    n_params = sum(p.numel() for p in model.parameters())
    n_params_tensor = torch.tensor(n_params, dtype=dtype, device=device)
    n_hidden_neurons = sum(model.model[i].out_features for i in linear_indices[:-1])

    # Parameters to optimize
    Z = torch.zeros(n_hidden_neurons, dtype=dtype, device=device, requires_grad=True)
    if to_log:
        with torch.no_grad():
            losses.append(loss(Z, diag_G, B).item())

    optimizer = OPTIMIZERS[optimizer]([Z], lr=lr)

    # Problem-specific matrices/vectors
    B = compute_matrix_B(model).to(device=device, dtype=dtype)     # shape: [m, n_hidden_neurons]
    diag_G = compute_diag_G(model).to(device=device, dtype=dtype)  # shape: [m], elementwise factor

    for i in range(n_iter):
        # do gd pass
        optimizer.zero_grad()
        output = loss(Z, diag_G, B)
        output.backward()
        optimizer.step()
        if to_log:
            with torch.no_grad():
                losses.append(loss(Z, diag_G, B).item())

    BZ = B @ Z.clone().detach()
    alpha = n_params/torch.sum(torch.exp(BZ) * diag_G).item()
    if to_log:
        log['loss'] = losses
        log['Z'] = Z.clone().detach()
        log['alpha'] = alpha

    if to_log:
        return BZ, log
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


# def reweight_model(model: nn.Module, BZ: torch.Tensor) -> nn.Module:
#     """
#     Reweight a model according to a log-rescaling vector BZ.

#     Args:
#         model (nn.Module): Pytorch model.
#         BZ (torch.Tensor): log-rescaling vector of size [n_params].

#     Returns:
#         nn.Module: Reweighted model.
#     """
#     # Copie du modèle (mêmes poids, pas de lien mémoire)
#     new_model = copy.deepcopy(model)

#     # Vérification
#     total_params = sum(p.numel() for p in model.parameters())
#     assert BZ.numel() == total_params, \
#         f"Taille de BZ {BZ.numel()} incompatible avec {total_params} paramètres"

#     # On va parcourir les paramètres
#     idx = 0
#     for p in new_model.parameters():
#         numel = p.numel()
#         # On reshape la portion correspondante de BZ
#         bz_chunk = BZ[idx:idx+numel].view_as(p.data)
#         idx += numel
#         # Multiplication
#         p.data = p.data * torch.exp(-0.5 * bz_chunk)
#     return new_model
