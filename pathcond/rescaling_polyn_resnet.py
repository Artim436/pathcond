import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from pathcond.utils import _param_start_offsets, split_sorted_by_column
from torch.nn.utils import parameters_to_vector, vector_to_parameters


def grad_path_norm(model, device="cpu", data="mnist") -> torch.Tensor:
    model.eval()
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
            start_rmean += o
    # Copy back into model
    vector_to_parameters(reweighted_vec, new_model.parameters())

    return new_model


@torch.jit.script
def update_z_polynomial_jit_sparse(g, pos_cols: list[torch.Tensor], neg_cols: list[torch.Tensor], nb_iter: int, n_hidden: int, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    device = g.device
    dtype = torch.double
    n_params = g.shape[0]
    H = n_hidden    
    Z = torch.zeros(H, dtype=torch.float64, device=device)
    BZ = torch.zeros(n_params, dtype=dtype, device=device) 
    for k in range(nb_iter):
        delta_total = 0.0
        for h in range(H):
            out_h, in_h = pos_cols[h], neg_cols[h]

            mask_in = torch.zeros(n_params, dtype=torch.bool, device=device)
            mask_out = torch.zeros(n_params, dtype=torch.bool, device=device)
            mask_in[in_h] = True
            mask_out[out_h] = True
            remaining = torch.logical_not(torch.logical_or(mask_in, mask_out))
            other_h = torch.nonzero(remaining).flatten()

            # other_h = (~(mask_in | mask_out)).nonzero(as_tuple=True)[0]

            card_in_h  = int(in_h.numel())
            card_out_h = int(out_h.numel())

            bhzh = Z[h]*(mask_out.long()- mask_in.long())

            
            Y_h = BZ - bhzh  # shape: [m]
            y_bar = Y_h.max()
            E = torch.exp(Y_h - y_bar) * g  # shape: [m]

            A_h = (card_in_h - card_out_h)
            B_h = E[out_h].sum()
            
            C_h = E[in_h].sum()
            D_h = E[other_h].sum()

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
            delta = z_new - Z[h]
            if delta != 0.0:
                delta_total += abs(delta)
                bh_delta = delta*(mask_out.long()- mask_in.long())
                BZ = BZ + bh_delta
                Z[h] = z_new
        if delta_total < tol:
            break
    return BZ, Z

def optimize_neuron_rescaling_polynomial_jitted_sparse(model, n_iter=10, tol=1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = torch.double
    diag_G = compute_diag_G(model).to(device=device, dtype=dtype)  # shape: [m], elementwise factor
    pos_cols, neg_cols = compute_matrix_B_resnet18_sparse_fast(model)
    n_hidden_neurons = _count_hidden_channels_resnet18(model)
    BZ, Z = update_z_polynomial_jit_sparse(diag_G,pos_cols, neg_cols, n_iter, n_hidden_neurons, tol)
    return BZ, Z

def compute_matrix_B_resnet18_sparse_fast(model: nn.Module):
    """
    Version sparse ultra-optimisée:
    Retourne pour chaque canal k de conv1 de chaque BasicBlock :

        - in_list[h]   = indices où B[:,h] = -1
        - out_list[h]  = indices où B[:,h] = +1
        - other_list[h]= indices où B[:,h] = 0

    Sans jamais construire la matrice B dense.
    """
    starts, n_params = _param_start_offsets(model)
    device = next(model.parameters()).device

    n_hidden = _count_hidden_channels_resnet18(model)

    # On stocke tous les +1 et -1 sous forme (rows, cols)
    pos_rows_all = []
    pos_cols_all = []
    neg_rows_all = []
    neg_cols_all = []

    col = 0  # colonne courante

    for block, lname, bi in _iter_basicblocks_resnet18(model):
        conv1, bn1, conv2 = block.conv1, block.bn1, block.conv2
        o = conv1.out_channels
        i = conv1.in_channels
        kH, kW = conv1.kernel_size

        # ----------- (1) conv1.weight -> -1 entrants -----------
        w1 = conv1.weight
        w1_start, _ = starts[id(w1)]

        # indices des poids entrants
        base = torch.arange(o * i * kH * kW, device=device)
        base = base.view(o, -1)          # (o, i*kH*kW)
        rows_in = (w1_start + base).reshape(-1)   # concat
        cols_in = torch.repeat_interleave(torch.arange(o, device=device), i * kH * kW)

        # Décalage colonne
        cols_in = cols_in + col

        neg_rows_all.append(rows_in)
        neg_cols_all.append(cols_in)

        # ----------- (2) bn1.bias -> -1 -----------
        b1 = bn1.bias
        b1_start, _ = starts[id(b1)]
        rows_bn = torch.arange(b1_start, b1_start + o, device=device)
        cols_bn = torch.arange(col, col + o, device=device)

        neg_rows_all.append(rows_bn)
        neg_cols_all.append(cols_bn)

        # ----------- (3) conv1.bias si existe -----------
        if conv1.bias is not None:
            cb = conv1.bias
            cb_start, _ = starts[id(cb)]
            rows_cb = torch.arange(cb_start, cb_start + o, device=device)
            cols_cb = torch.arange(col, col + o, device=device)

            neg_rows_all.append(rows_cb)
            neg_cols_all.append(cols_cb)

        # ----------- (4) conv2.weight -> +1 sortants -----------
        w2 = conv2.weight
        o2, o_in, kH2, kW2 = w2.shape
        assert o_in == o
        w2_start, _ = starts[id(w2)]

        base2 = torch.arange(o2 * o * kH2 * kW2, device=device)
        base2 = base2.view(o2, o, -1)         # (o2, o, kH2*kW2)
        rows_out = (w2_start + base2).reshape(-1)

        # cols: chaque canal k répété kH2*kW2 fois, et répété pour chaque out_channel
        cols_out = torch.arange(o, device=device).repeat_interleave(kH2 * kW2)
        cols_out = cols_out.repeat(o2) + col

        pos_rows_all.append(rows_out)
        pos_cols_all.append(cols_out)

        col += o  # on avance après ce bloc

    # Fusionne tous les +1 et -1
    pos_rows_all = torch.cat(pos_rows_all)
    pos_cols_all = torch.cat(pos_cols_all)
    neg_rows_all = torch.cat(neg_rows_all)
    neg_cols_all = torch.cat(neg_cols_all)
    neg_cols_all, idx = torch.sort(neg_cols_all) # trier les colonnes négatives
    neg_rows_all = neg_rows_all[idx]
    pos_cols_all, idx = torch.sort(pos_cols_all) # trier les colonnes positives
    pos_rows_all = pos_rows_all[idx]


    pos_cols = split_sorted_by_column(pos_cols_all, pos_rows_all, n_hidden)
    neg_cols = split_sorted_by_column(neg_cols_all, neg_rows_all, n_hidden)

    return pos_cols, neg_cols