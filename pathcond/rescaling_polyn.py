import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision.models.resnet import BasicBlock
from pathcond.utils import iter_modules_by_type, count_hidden_channels_generic, _detect_model_type
from pathcond.network_to_optim import compute_diag_G, compute_B_mlp, compute_B_cnn


@torch.jit.script
def update_z_polynomial(g, pos_cols: list[torch.Tensor], neg_cols: list[torch.Tensor], nb_iter: int, n_hidden: int, tol: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
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

            # other_h = (~(mask_in | mask_out)).nonzero(as_tuple=True)[0] unused in jitted code

            card_in_h = int(in_h.numel())
            card_out_h = int(out_h.numel())

            Z_h = Z[h].item()

            # Y_h = BZ - bhzh  # O(n_params)
            Y_h = BZ.clone()  # O(n_params)
            Y_h[out_h] = BZ[out_h] - Z_h
            Y_h[in_h] = BZ[in_h] + Z_h

            # y_bar = Y_h.max()  # O(n_params) in case of overflow problems
            # E = torch.exp(Y_h - y_bar) * g  # O(n_params
            E = torch.exp(Y_h) * g  # O(n_params)

            A_h = (card_in_h - card_out_h)
            B_h = E[out_h].sum()

            C_h = E[in_h].sum()
            E_sum = E.sum()
            D_h = E_sum - B_h - C_h  # avoid computing other_h

            # Polynomial coefficients
            a = B_h * (A_h + n_params)
            b = D_h * A_h
            c = C_h * (A_h - n_params)

            if a <= 0.0:
                raise ValueError(
                    f"Non-positive a={a} in quadratic for neuron {h} at iter {k}, A_h={A_h}, B_h={B_h}, C_h={C_h}, D_h={D_h}")
            if c >= 0.0:
                raise ValueError(
                    f"Non-negative c={c} in quadratic for neuron {h} at iter {k}, A_h={A_h}, B_h={B_h}, C_h={C_h}, D_h={D_h}")
            else:
                disc = torch.square(b) - 4.0 * a * c
                if disc > 0.0:
                    sqrt_disc = torch.sqrt(disc)
                    x1 = (-b + sqrt_disc) / (2.0 * a)
                    if x1 <= 0.0:
                        raise ValueError(
                            f"Unexpected number of positive roots for neuron {h} at iter {k}")
                    z_new = torch.log(x1)
                else:
                    raise ValueError(f"Negative or infinit discriminant {disc} in quadratic for neuron {h} at iter {k}")
            # Update Z[h] and incrementally refresh BZ
            delta = z_new - Z[h]
            delta_total += abs(delta)
            BZ[out_h] += delta
            BZ[in_h] -= delta
            Z[h] = z_new
        if delta_total < tol:
            break
    return BZ, Z


def optimize_rescaling_polynomial(model, n_iter=10, tol=1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = torch.double
    diag_G = compute_diag_G(model).to(device=device, dtype=dtype)  # shape: [m], elementwise factor
    model_type = _detect_model_type(model)
    if model_type == "MLP":
        linear_indices = [i for i, layer in enumerate(model.model) if isinstance(layer, nn.Linear)]
        n_hidden_neurons = sum(model.model[i].out_features for i in linear_indices[:-1])
        pos_cols, neg_cols = compute_B_mlp(model)
    elif model_type == "CNN":
        n_hidden_neurons = count_hidden_channels_generic(model)
        pos_cols, neg_cols = compute_B_cnn(model)
    BZ, Z = update_z_polynomial(diag_G, pos_cols, neg_cols, n_iter, n_hidden_neurons, tol)
    return BZ, Z


def reweight_model(model: nn.Module, BZ: torch.Tensor, Z: torch.Tensor = None) -> nn.Module:
    """
    Reweight a Pytorch model automatically.
    - Always rescales parameters using BZ.
    - If BatchNorm is detected, also rescales running_mean using Z.

    Args:
        model (nn.Module): Pytorch model.
        BZ (torch.Tensor): log-rescaling vector of size [n_params].
        Z (torch.Tensor, optional): rescaling vector for BatchNorm running statistics.

    Returns:
        nn.Module: Reweighted model (deep copy).
    """

    # Deep copy to avoid modifying the original
    new_model = copy.deepcopy(model)
    new_model.eval()

    # Detect device
    device = next(model.parameters()).device
    new_model.to(device)

    # Flatten parameters
    param_vec = parameters_to_vector(new_model.parameters())

    BZ = BZ.to(device)
    assert BZ.shape == param_vec.shape, \
        f"Size of BZ {BZ.shape} incompatible with {param_vec.shape}"

    # Apply standard parameter reweighting
    reweighted_vec = param_vec * torch.exp(-0.5 * BZ)

    Z = Z.to(device)
    start_rmean = 0

    with torch.no_grad():
        for _, block in iter_modules_by_type(new_model, BasicBlock):
            bn1 = block.bn1
            o = block.conv1.out_channels
            # running mean
            rmean = bn1.running_mean  # [o]
            rmean_start = start_rmean
            rmean_end = start_rmean + o
            bzrmean = Z[rmean_start:rmean_end]
            rmean *= torch.exp(0.5 * bzrmean)
            start_rmean += o

    # Copy back parameters
    vector_to_parameters(reweighted_vec, new_model.parameters())

    return new_model


