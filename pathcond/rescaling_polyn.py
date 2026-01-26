import copy
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision.models.resnet import BasicBlock
from pathcond.utils import iter_modules_by_type, count_hidden_channels_generic, _detect_model_type, count_hidden_channels_full_conv, count_hidden_channels_resnet_c
from pathcond.network_to_optim import compute_diag_G, compute_B_mlp, compute_B_resnet, compute_B_full_conv, compute_B_resnet_c
from torch import Tensor
from typing import List, Tuple


@torch.jit.script
def update_z_polynomial(
    g: torch.Tensor, 
    pos_cols: list[torch.Tensor], 
    neg_cols: list[torch.Tensor], 
    nb_iter: int, 
    n_hidden: int, 
    tol: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    device = g.device
    dtype = torch.double
    n_params = g.shape[0]
    H = n_hidden    
    Z = torch.zeros(H, dtype=dtype, device=device) 
    BZ = torch.zeros(n_params, dtype=dtype, device=device)
    E_sum = 0.0
    for k in range(nb_iter):
        delta_total = 0.0
        for h in range(H):
            out_h, in_h = pos_cols[h], neg_cols[h]
            card_in_h  = float(in_h.numel()) 
            card_out_h = float(out_h.numel())

            E_h_old = torch.sum(torch.exp(BZ[out_h]) * g[out_h]) + torch.sum(torch.exp(BZ[in_h]) * g[in_h])
            
            if h == 0 and k == 0:
                E_sum = g.sum() 
            
            D_h = E_sum - E_h_old
            Z_h = Z[h].item()
            
            E_out_h = torch.exp(BZ[out_h] - Z_h) * g[out_h] 
            B_h = E_out_h.sum()
            
            E_in_h = torch.exp(BZ[in_h] + Z_h) * g[in_h]
            C_h = E_in_h.sum()

            A_h = card_in_h - card_out_h
            
            a = B_h * (A_h + n_params)
            b = D_h * A_h
            c = C_h * (A_h - n_params)

            disc = b * b - 4.0 * a * c
            if disc < 0.0:
                raise ValueError(f"Negative discriminant encountered: {disc.item()}")
            sqrt_disc = torch.sqrt(disc)
            x1 = (-b + sqrt_disc) / (2.0 * a)
            if x1 <= 0.0:
                raise ValueError(f"Non-positive root encountered: {x1.item()}")
            z_new = torch.log(x1)

            delta = z_new - Z[h]
            delta_total += abs(delta)
            
            BZ[out_h] += delta
            BZ[in_h] -= delta
            Z[h] = z_new
            
            E_h_new = torch.sum(torch.exp(BZ[out_h]) * g[out_h]) + torch.sum(torch.exp(BZ[in_h]) * g[in_h])
            E_sum = D_h + E_h_new
            
        if delta_total < tol:
            break
            
    return BZ, Z


@torch.jit.script
def update_z_polynomial_enorm(
    param_vector: Tensor,
    pos_cols: List[Tensor], 
    neg_cols: List[Tensor], 
    nb_iter: int, 
    n_hidden: int, 
    tol: float = 1e-6
) -> Tuple[Tensor, Tensor]:
    # Use double precision as requested
    dtype = torch.double
    device = param_vector.device
    n_params = param_vector.numel()
    
    Z = torch.zeros(n_hidden, dtype=dtype, device=device) 
    BZ = torch.zeros(n_params, dtype=dtype, device=device)
    
    # Pre-calculate squared parameters
    g = param_vector.pow(2).to(dtype)
    
    for k in range(nb_iter):
        delta_total = 0.0
        for h in range(n_hidden):
            # Extract indices for current hidden unit
            out_idx = pos_cols[h]
            in_idx = neg_cols[h]

            # Current Z value for this hidden unit
            z_h = Z[h]
            
            # Use indexing to get relevant g values
            g_out = g[out_idx]
            g_in = g[in_idx]
            
            # Vectorized exponential and sum operations
            # Note: We keep things as tensors to avoid .item() inside the loop
            e_out_h = torch.exp(2.0 * (BZ[out_idx] - z_h)) * g_out
            b_h = e_out_h.sum()
            
            e_in_h = torch.exp(2.0 * (BZ[in_idx] + z_h)) * g_in
            c_h = e_in_h.sum()
            
            # Calculate new Z
            x = c_h / b_h
            z_new = 0.25 * torch.log(x)

            delta = z_new - z_h
            delta_total += torch.abs(delta).item()
            
            # Update BZ and Z
            BZ[out_idx] += delta
            BZ[in_idx] -= delta
            Z[h] = z_new
            
        if delta_total < tol:
            break
            
    return BZ, Z


def optimize_rescaling_polynomial(model, n_iter=10, tol=1e-6, resnet=False, enorm=False) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = torch.double
    if enorm:
        param_vector = parameters_to_vector(model.parameters()).to(device=device, dtype=dtype)
    else:
        diag_G = compute_diag_G(model).to(device=device, dtype=dtype)  # shape: [m], elementwise factor
    model_type = _detect_model_type(model)
    if model_type == "MLP":
        linear_indices = [i for i, layer in enumerate(model.model) if isinstance(layer, nn.Linear)]
        n_hidden_neurons = sum(model.model[i].out_features for i in linear_indices[:-1])
        pos_cols, neg_cols = compute_B_mlp(model)
    elif model_type == "CNN":
        if not resnet:
            n_hidden_neurons = count_hidden_channels_full_conv(model)
            pos_cols, neg_cols = compute_B_full_conv(model)
        else:
            n_hidden_neurons = count_hidden_channels_resnet_c(model)
            pos_cols, neg_cols = compute_B_resnet_c(model)
    if pos_cols is None or neg_cols is None:
        raise ValueError("pos_cols and neg_cols could not be computed.")
    if enorm:
        BZ, Z = update_z_polynomial_enorm(param_vector, pos_cols, neg_cols, n_iter, n_hidden_neurons, tol)
    else:
        BZ, Z = update_z_polynomial(diag_G, pos_cols, neg_cols, n_iter, n_hidden_neurons, tol)
    large_value = 1e1
    # print if there is some inf or nan values
    if torch.isinf(BZ).any() or torch.isinf(Z).any():
        print("Warning: Inf values encountered in BZ or Z during optimization.")
    if torch.isnan(BZ).any() or torch.isnan(Z).any():
        print("Warning: NaN values encountered in BZ or Z during optimization.")
    BZ = torch.where(torch.isinf(BZ), torch.full_like(BZ, large_value), BZ)
    Z = torch.where(torch.isinf(Z), torch.full_like(Z, large_value), Z)
    BZ = torch.where(torch.isnan(BZ), torch.zeros_like(BZ), BZ)
    Z = torch.where(torch.isnan(Z), torch.zeros_like(Z), Z)
    BZ = BZ.to(dtype=torch.float32)
    Z = Z.to(dtype=torch.float32)
    
    return BZ, Z




def reweight_model(model: nn.Module, BZ: torch.Tensor, Z: torch.Tensor = None, module_type=None, enorm=False) -> nn.Module:
    """
    Reweight a Pytorch model automatically.
    - Always rescales parameters using BZ.
    - If BatchNorm is detected, also rescales running_mean using Z.

    Args:
        model (nn.Module): Pytorch model.
        BZ (torch.Tensor): log-rescaling vector of size [n_params].
        Z (torch.Tensor, optional): rescaling vector for BatchNorm running statistics.
        enorm (bool, optional): Whether to use equinorm reweighting.

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
    if enorm:
        reweighted_vec = param_vec * torch.exp(BZ)
    else:
        reweighted_vec = param_vec * torch.exp(-0.5 * BZ)

    Z = Z.to(device)
    start_rmean = 0

    with torch.no_grad():
        for _, block in iter_modules_by_type(new_model, module_type or BasicBlock):
            if hasattr(block, 'bn1'):
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


def reweight_model_inplace(
    model: nn.Module, 
    BZ: torch.Tensor, 
    enorm: bool = False
) -> None:
    """
    Reweight a Pytorch model IN-PLACE .
    Rescale les paramètres en utilisant BZ.
    Ne prend pas en charge la rescaling des BatchNorm.
    
    Args:
        model (nn.Module): Pytorch model.
        BZ (torch.Tensor): log-rescaling vector de taille [n_params].
        enorm (bool, optional): Whether to use equinorm reweighting.
    
    Returns:
        None (modifications in-place)
    """
    
    model.eval()
    
    # Detect device
    device = next(model.parameters()).device
    BZ = BZ.to(device)
    
    # Flatten parameters
    param_vec = parameters_to_vector(model.parameters())
    
    assert BZ.shape == param_vec.shape, \
        f"Size of BZ {BZ.shape} incompatible with {param_vec.shape}"
    
    # Apply standard parameter reweighting
    if enorm:
        reweighted_vec = param_vec * torch.exp(BZ)
    else:
        reweighted_vec = param_vec * torch.exp(-0.5 * BZ)
    
    # Copy back parameters IN-PLACE
    with torch.no_grad():
        vector_to_parameters(reweighted_vec, model.parameters())
