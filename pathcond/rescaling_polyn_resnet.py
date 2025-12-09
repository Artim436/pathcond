import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from pathcond.utils import _param_start_offsets
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def grad_path_norm(model, device="cpu", data="mnist") -> torch.Tensor:
    if data == "mnist":
        inputs = torch.ones(1, 1, 32, 32)
    else:  # cifar10
        inputs = torch.ones(1, 3, 32, 32)
    inputs = inputs.to(device)
    def fct(model, inputs, device="cpu"):
        model = model.to(device)
        return model(inputs).sum().to(device)
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


def _iter_basicblocks_resnet18(model: nn.Module):
    """
    Itère (dans l'ordre forward) sur tous les BasicBlock de layer1..layer4.
    Renvoie des tuples (block, layer_name, block_idx).
    """
    for lname in ["layer1", "layer2", "layer3", "layer4"]:
        layer = getattr(model, lname)
        for i, block in enumerate(layer):  # torchvision.models.resnet.BasicBlock
            yield block, lname, i


# ---------- Comptage des "neurones" cachés (canaux conv1) ----------
def _count_hidden_channels_resnet18(model: nn.Module) -> int:
    total = 0
    for block, _, _ in _iter_basicblocks_resnet18(model):
        total += block.conv1.out_channels
    return total


# ---------- Construction de B au niveau canal (conv1 -> conv2) ----------
def compute_matrix_B_resnet18(model: nn.Module) -> torch.Tensor:
    """
    Matrice B (int8) pour le rescaling par canal des sorties de conv1 dans chaque BasicBlock.
    Pour chaque canal k de conv1 d'un bloc:
      -1 sur conv1.weight[k, :, :, :]  (poids entrants du canal)
      -1 sur bn1.bias[k]
      +1 sur conv2.weight[:, k, :, :]  (poids sortants vers la couche suivante)
    """
    # Offsets vectoriels stables
    starts, n_params = _param_start_offsets(model)
    device = next(model.parameters()).device

    # Nombre total de "neurones" cachés (tous les canaux conv1)
    n_hidden = _count_hidden_channels_resnet18(model)
    print(f"Number of hidden neurons (conv1 channels): {n_hidden}")
    print(f"Number of parameters: {n_params}")
    B = torch.zeros((n_params, n_hidden), dtype=torch.int8, device=device)

    col = 0  # curseur de colonne dans B

    for block, lname, bi in _iter_basicblocks_resnet18(model):
        conv1, bn1, conv2 = block.conv1, block.bn1, block.conv2

        # Sanity checks usuels
        assert isinstance(conv1, nn.Conv2d) and isinstance(conv2, nn.Conv2d)
        assert isinstance(bn1, nn.BatchNorm2d)
        assert conv2.in_channels == conv1.out_channels, \
            f"Mismatch in {lname}[{bi}]: conv2.in_channels != conv1.out_channels"

        o = conv1.out_channels
        i = conv1.in_channels
        kH, kW = conv1.kernel_size

        # ---- (1) conv1.weight : -1 sur tous les noyaux du canal k ----
        w1 = conv1.weight  # [o, i, kH, kW]
        w1_start, _ = starts[id(w1)]
        # Bloc de B correspondant à toute la (flatten) de conv1.weight
        block_in = B[w1_start : w1_start + o*i*kH*kW, col : col + o]  # shape = (o*i*kH*kW, o)

        # Motif: pour chaque colonne k, on met -1 sur le "plan" w1[k, :, :, :]
        # Construisons un tenseur (o, i, kH, kW, o) avec -1 sur [:, :, :, :, diag]
        T = torch.zeros((o, i, kH, kW, o), dtype=torch.int8, device=device)
        idx = torch.arange(o, device=device)
        T[idx, :, :, :, idx] = -1
        block_in.copy_(T.view(o * i * kH * kW, o))

        # ---- (2) bn1.weight & bn1.bias : -1 sur la diagonale ----
        b1 = bn1.bias    # [o]
        b1_start, _ = starts[id(b1)]

        B[b1_start : b1_start + o, col : col + o].copy_(
            -torch.eye(o, dtype=torch.int8, device=device)
        )

        # Si conv1 a un biais (rare dans ResNet), on le compte aussi comme "entrant"
        if conv1.bias is not None:
            cb = conv1.bias  # [o]
            cb_start, _ = starts[id(cb)]
            B[cb_start : cb_start + o, col : col + o].copy_(
                -torch.eye(o, dtype=torch.int8, device=device)
            )

        # ---- (3) conv2.weight : +1 sur la colonne d'entrée k ----
        w2 = conv2.weight  # [o2, o, kH2, kW2]  (o en dimension "in_channels")
        o2, o_in, kH2, kW2 = w2.shape
        assert o_in == o

        w2_start, _ = starts[id(w2)]
        block_out = B[w2_start : w2_start + o2*o*kH2*kW2, col : col + o]  # (o2*o*kH2*kW2, o)

        T_out = torch.zeros((o2, o, kH2, kW2, o), dtype=torch.int8, device=device)
        idx = torch.arange(o, device=device)
        T_out[:, idx, :, :, idx] = 1
        block_out.copy_(T_out.view(o2 * o * kH2 * kW2, o))
        # Avancer le curseur (une colonne par canal)
        col += o

    return B


def reweight_model_resnet(model: nn.Module, BZ: torch.Tensor, Z: torch.Tensor) -> nn.Module:
    """
    Reweight a model according to a log-rescaling vector BZ.

    Args:
        model (nn.Module): Pytorch model.
        BZ (torch.Tensor): log-rescaling vector of size [n_params].

    Returns:
        nn.Module: Reweighted model (on same device as input model).
    """
    # Deep copy to avoid modifying the original
    new_model = copy.deepcopy(model)

    # Detect device of model
    device = next(model.parameters()).device
    new_model.to(device)

    # Flatten parameters into one vector
    param_vec = parameters_to_vector(new_model.parameters())

    # Ensure BZ is on same device and shape is correct
    BZ = BZ.to(device)
    assert BZ.shape == param_vec.shape, \
        f"Taille de BZ {BZ.shape} incompatible avec {param_vec.shape}"

    # Vectorized reweighting
    reweighted_vec = param_vec * torch.exp(-0.5 * BZ)

    #modifie running mean and variance of batchnorm layers
    start_rmean = 0
    start_rbias = 0
    start_weights = 0
    with torch.no_grad():
        for block, lname, bi in _iter_basicblocks_resnet18(model):
            bn1 = block.bn1
            o = block.conv1.out_channels

            # running mean
            rmean = bn1.running_mean  # [o]
            rmean_start = start_rmean
            rmean_end = start_rmean + o
            bzrmean = Z[rmean_start:rmean_end]
            rmean *= torch.exp(-0.5 * bzrmean)
            start_rmean += o

            # running var
            rbias = bn1.bias  # [o]
            rbias_start = start_rbias
            rbias_end = start_rbias + o
            bzrbias = Z[rbias_start:rbias_end]
            rbias *= torch.exp(-0.5 * bzrbias)
            start_rbias += o

            rweight = bn1.weight  # [o]
            rweight_start = start_weights
            rweight_end = start_weights + o
            bzrw = Z[rweight_start:rweight_end]
            rweight *= torch.exp(-0.5 * bzrw)
            start_weights += o
    print(start_rmean, start_rbias, start_weights)

    # Copy back into model
    vector_to_parameters(reweighted_vec, new_model.parameters())

    return new_model


def optimize_neuron_rescaling_polynomial_jitted_resnet(model, n_iter=10, tol=1e-6) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = torch.double
    print("B computed")
    diag_G = compute_diag_G(model).to(device=device, dtype=dtype)  # shape: [m], elementwise factor
    print("diag_G computed")
    B = compute_matrix_B_resnet18(model).to(device=device)
    BZ, Z = update_z_polynomial_jit_resnet(diag_G, B, n_iter)
    # free memory
    del B
    del diag_G
    return BZ, Z

@torch.jit.script
def update_z_polynomial_jit_resnet(g, B, nb_iter: int, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.zeros(B.shape[1], dtype=torch.float64, device=B.device)
    # Do one pass on every z_h
    # Maintain BZ incrementally: BZ = B @ Z
    BZ = torch.zeros(B.shape[0], dtype=z.dtype, device=z.device)

    n_params_tensor = B.shape[0]
    H = B.shape[1]

    for k in range(nb_iter):
        delta_total = 0.0
        for h in range(H):
            b_h = B[:, h]

            mask_in_h = (b_h == -1)
            mask_out_h = (b_h == 1)
            mask_other_h = (b_h == 0)
            card_in_h = mask_in_h.sum()
            card_out_h = mask_out_h.sum()

            # directly use precomputed card
            A_h = int(card_in_h.item()) - int(card_out_h.item())

            b_h = b_h.to(dtype=z.dtype)  # shape: [m]

            # Leave-one-out energy vector
            Y_h = BZ - b_h * z[h]
            y_bar = Y_h.max()
            E = torch.exp(Y_h - y_bar) * g

            # sums using masks
            B_h = (E * mask_out_h).sum()
            C_h = (E * mask_in_h).sum()
            D_h = (E * mask_other_h).sum()

            # Polynomial coefficients
            a = B_h * (A_h + n_params_tensor)
            b = D_h * A_h
            c = C_h * (A_h - n_params_tensor)

            disc = b * b - 4.0 * a * c
            sqrt_disc = torch.sqrt(disc)
            x1 = (-b + sqrt_disc) / (2.0 * a)
            x2 = (-b - sqrt_disc) / (2.0 * a)

            z_new = torch.log(torch.maximum(x1, x2))

            # Update Z[h] and incrementally refresh BZ
            delta = z_new - z[h]
            if delta != 0.0:
                delta_total += abs(delta)
                BZ = BZ + b_h * delta
                z[h] = z_new
        if delta_total < tol:
            break
    return BZ, z


def reweight_model_resnet(model: nn.Module, BZ: torch.Tensor, Z: torch.Tensor) -> nn.Module:
    """
    Reweight a model according to a log-rescaling vector BZ.

    Args:
        model (nn.Module): Pytorch model.
        BZ (torch.Tensor): log-rescaling vector of size [n_params].

    Returns:
        nn.Module: Reweighted model (on same device as input model).
    """
    # Deep copy to avoid modifying the original
    new_model = copy.deepcopy(model)
    new_model.eval()

    # Detect device of model
    device = next(model.parameters()).device
    new_model.to(device)

    # Flatten parameters into one vector
    param_vec = parameters_to_vector(new_model.parameters())

    # Ensure BZ is on same device and shape is correct
    BZ = BZ.to(device)
    assert BZ.shape == param_vec.shape, \
        f"Taille de BZ {BZ.shape} incompatible avec {param_vec.shape}"

    # Vectorized reweighting
    reweighted_vec = param_vec * torch.exp(-0.5 * BZ)

    #modifie running mean and variance of batchnorm layers
    start_rmean = 0

    with torch.no_grad():
        for block, lname, bi in _iter_basicblocks_resnet18(new_model):
            bn1 = block.bn1
            o = block.conv1.out_channels
            # running mean
            rmean = bn1.running_mean  # [o]
            rmean_start = start_rmean
            rmean_end = start_rmean + o
            bzrmean = Z[rmean_start:rmean_end]
            rmean *= torch.exp(0.5 * bzrmean)
            bias = bn1.bias  # [o]
            bias *= torch.exp(0.5 * bzrmean)
            start_rmean += o
    # Copy back into model
    vector_to_parameters(reweighted_vec, new_model.parameters())

    return new_model


def compute_matrix_B_resnet18_it(model: nn.Module) -> torch.Tensor:
    """
    Matrice B (int8) pour le rescaling par canal des sorties de conv1 dans chaque BasicBlock.
    Pour chaque canal k de conv1 d'un bloc:
      -1 sur conv1.weight[k, :, :, :]  (poids entrants du canal)
      -1 sur bn1.bias[k]
      +1 sur conv2.weight[:, k, :, :]  (poids sortants vers la couche suivante)
    """
    # Offsets vectoriels stables
    starts, n_params = _param_start_offsets(model)
    device = "cpu" #next(model.parameters()).device

    for block, lname, bi in _iter_basicblocks_resnet18(model):
        conv1, bn1, conv2 = block.conv1, block.bn1, block.conv2

        # Sanity checks usuels
        assert isinstance(conv1, nn.Conv2d) and isinstance(conv2, nn.Conv2d)
        assert isinstance(bn1, nn.BatchNorm2d)
        assert conv2.in_channels == conv1.out_channels, \
            f"Mismatch in {lname}[{bi}]: conv2.in_channels != conv1.out_channels"

        o = conv1.out_channels
        i = conv1.in_channels
        kH, kW = conv1.kernel_size

        B = torch.zeros((n_params, o), dtype=torch.int8, device=device)

        # ---- (1) conv1.weight : -1 sur tous les noyaux du canal k ----
        w1 = conv1.weight  # [o, i, kH, kW]
        w1_start, _ = starts[id(w1)]
        # Bloc de B correspondant à toute la (flatten) de conv1.weight
        block_in = B[w1_start : w1_start + o*i*kH*kW, :o]  # shape = (o*i*kH*kW, o)

        # Motif: pour chaque colonne k, on met -1 sur le "plan" w1[k, :, :, :]
        # Construisons un tenseur (o, i, kH, kW, o) avec -1 sur [:, :, :, :, diag]
        T = torch.zeros((o, i, kH, kW, o), dtype=torch.int8, device=device)
        idx = torch.arange(o, device=device)
        T[idx, :, :, :, idx] = -1
        block_in.copy_(T.view(o * i * kH * kW, o))

        # ---- (2) bn1.weight & bn1.bias : -1 sur la diagonale ----
        b1 = bn1.bias    # [o]
        b1_start, _ = starts[id(b1)]

        B[b1_start : b1_start + o, :o].copy_(
            -torch.eye(o, dtype=torch.int8, device=device)
        )

        # Si conv1 a un biais (rare dans ResNet), on le compte aussi comme "entrant"
        if conv1.bias is not None:
            cb = conv1.bias  # [o]
            cb_start, _ = starts[id(cb)]
            B[cb_start : cb_start + o, :o].copy_(
                -torch.eye(o, dtype=torch.int8, device=device)
            )

        # ---- (3) conv2.weight : +1 sur la colonne d'entrée k ----
        w2 = conv2.weight  # [o2, o, kH2, kW2]  (o en dimension "in_channels")
        o2, o_in, kH2, kW2 = w2.shape
        assert o_in == o

        w2_start, _ = starts[id(w2)]
        block_out = B[w2_start : w2_start + o2*o*kH2*kW2, :o]  # (o2*o*kH2*kW2, o)

        T_out = torch.zeros((o2, o, kH2, kW2, o), dtype=torch.int8, device=device)
        idx = torch.arange(o, device=device)
        T_out[:, idx, :, :, idx] = 1
        block_out.copy_(T_out.view(o2 * o * kH2 * kW2, o))
        # Avancer le curseur (une colonne par canal)

        yield B


def optimize_neuron_rescaling_polynomial_jitted_resnet_iter(model, n_iter=10, tol=1e-6) -> torch.Tensor:
    device = next(model.parameters()).device
    dtype = torch.double
    diag_G = compute_diag_G(model).to(device=device, dtype=dtype)  # shape: [m], elementwise factor
    BZ, Z = update_z_polynomial_jit_resnet_iterable(model, n_hidden=_count_hidden_channels_resnet18(model), g=diag_G, nb_iter=n_iter, tol=tol)
    del diag_G
    return BZ, Z



# Pas de jit possible avec yield
def update_z_polynomial_jit_resnet_iterable(model: nn.Module, n_hidden: int, g: torch.Tensor,  nb_iter: int, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    n_params = g.shape[0]
    z = torch.zeros(n_hidden, dtype=torch.float64, device=g.device)
    BZ = torch.zeros(n_params, dtype=z.dtype, device=z.device)
    H = z.shape[0]
    for k in range(nb_iter):
        delta_total = 0.0
        for B in compute_matrix_B_resnet18_it(model):
            BZ, Z, sub_tot_delta = sub_update_z_polynomial_jit_resnet_iterable(n_params, g, B, BZ, z)
            delta_total += sub_tot_delta
            del B
            if delta_total < tol:
                break
    return BZ, z


@torch.jit.script
def sub_update_z_polynomial_jit_resnet_iterable(n_params: int, g: torch.Tensor, B: torch.Tensor, BZ: torch.Tensor,  z: torch.Tensor, block_size: int = 32) -> tuple[torch.Tensor, torch.Tensor, float]:
    delta_total = 0.0
    H = B.shape[1]
    for h0 in range(0, H, block_size): # on ne peut pas mettre toute la matrice B sur gpu donc on on la met sur cpu et on la met sur gpu par bloc
        h1 = min(h0 + block_size, H)
        k = h1 - h0
        B_block = B[:, h0:h1].to(g.device)
        for h in range(k):
            b_h = B_block[:, h].to(device=g.device)

            mask_in_h = (b_h == -1)
            mask_out_h = (b_h == 1)
            mask_other_h = (b_h == 0)
            card_in_h = mask_in_h.sum()
            card_out_h = mask_out_h.sum()

            # directly use precomputed card
            A_h = int(card_in_h.item()) - int(card_out_h.item())

            b_h = b_h.to(dtype=z.dtype)  # shape: [m]

            # Leave-one-out energy vector
            Y_h = BZ - b_h * z[h]
            y_bar = Y_h.max()
            E = torch.exp(Y_h - y_bar) * g

            # sums using masks
            B_h = (E * mask_out_h).sum()
            C_h = (E * mask_in_h).sum()
            D_h = (E * mask_other_h).sum()

            # Polynomial coefficients
            a = B_h * (A_h + n_params)
            b = D_h * A_h
            c = C_h * (A_h - n_params)

            disc = b * b - 4.0 * a * c
            sqrt_disc = torch.sqrt(disc)
            x1 = (-b + sqrt_disc) / (2.0 * a)
            x2 = (-b - sqrt_disc) / (2.0 * a)

            z_new = torch.log(torch.maximum(x1, x2))

            # Update Z[h] and incrementally refresh BZ
            delta = z_new - z[h]
            if delta != 0.0:
                delta_total += abs(delta)
                BZ = BZ + b_h * delta
                z[h] = z_new
    del B
    return BZ, z, delta_total