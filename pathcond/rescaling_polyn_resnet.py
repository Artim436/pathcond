import copy
import torch
import torch.nn as nn
from tqdm import tqdm
import math
from pathcond.utils import _param_start_offsets, split_sorted_by_column
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torchvision.models.resnet import BasicBlock


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

def iter_modules_by_type(model, module_type):
    """
    Itère sur tous les sous-modules d'un modèle PyTorch qui sont
    des instances du type de module spécifié.

    Args:
        model (nn.Module): Le modèle CNN (ex: ResNet, VGG, CustomModel).
        module_type (type/tuple[type]): Le type de module à rechercher (ex: nn.Conv2d, BasicBlock).

    Yields:
        (name, module): Le nom et l'instance du module trouvé.
    """
    # named_modules() itère sur TOUS les modules du plus haut niveau au plus bas niveau
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            # Évite d'itérer sur le modèle lui-même s'il est du type recherché
            if module is not model:
                yield name, module

def count_hidden_channels_generic(model):
    """
    Compte les canaux de sortie (out_channels) de la première convolution
    de TOUS les BasicBlock trouvés dans un modèle.
    Ceci fonctionnera pour ResNet-18, 34, ou tout modèle utilisant BasicBlock.
    """
    total_channels = 0

    # 1. Utilisation de la fonction générique pour itérer sur tous les BasicBlock
    for name, block in iter_modules_by_type(model, BasicBlock):

        # 2. Le reste de votre logique reste la même
        #    'block' est ici l'instance de BasicBlock
        if hasattr(block, 'conv1') and isinstance(block.conv1, nn.Conv2d):
            total_channels += block.conv1.out_channels
        else:
            # Sécurité si un bloc BasicBlock n'a pas l'attribut 'conv1'
            print(f"Avertissement: Le bloc {name} ne possède pas l'attribut 'conv1' de type Conv2d.")

    return total_channels


def reweight_model_cnn(model: nn.Module, BZ: torch.Tensor, Z: torch.Tensor) -> nn.Module:
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
        for lname, block in iter_modules_by_type(model, BasicBlock):
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

            # other_h = (~(mask_in | mask_out)).nonzero(as_tuple=True)[0] unused in jitted code

            card_in_h  = int(in_h.numel())
            card_out_h = int(out_h.numel())

            Z_h = Z[h].item()
      
            # Y_h = BZ - bhzh  # O(n_params)
            Y_h = BZ.clone() # O(n_params)
            Y_h[out_h] = BZ[out_h] - Z_h
            Y_h[in_h] = BZ[in_h] + Z_h
            
            y_bar = Y_h.max() # O(n_params)
            E = torch.exp(Y_h - y_bar) * g  # O(n_params)

            A_h = (card_in_h - card_out_h)
            B_h = E[out_h].sum()
            
            C_h = E[in_h].sum()
            E_sum = E.sum()
            D_h = E_sum - B_h - C_h # avoid computing other_h

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
            delta_total += abs(delta)
            BZ[out_h] += delta
            BZ[in_h] -= delta
            Z[h] = z_new
        if delta_total < tol:
            break
    return BZ, Z

def optimize_neuron_rescaling_polynomial_jitted_sparse(model, n_iter=10, tol=1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    device = next(model.parameters()).device
    dtype = torch.double
    diag_G = compute_diag_G(model).to(device=device, dtype=dtype)  # shape: [m], elementwise factor
    pos_cols, neg_cols = compute_matrix_B_cnn_sparse(model)
    n_hidden_neurons = count_hidden_channels_generic(model)
    BZ, Z = update_z_polynomial_jit_sparse(diag_G,pos_cols, neg_cols, n_iter, n_hidden_neurons, tol)
    return BZ, Z

def compute_matrix_B_cnn_sparse(model: nn.Module):
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

    n_hidden = count_hidden_channels_generic(model)

    # On stocke tous les +1 et -1 sous forme (rows, cols)
    pos_rows_all = []
    pos_cols_all = []
    neg_rows_all = []
    neg_cols_all = []

    col = 0  # colonne courante

    for lname, block in iter_modules_by_type(model, BasicBlock):
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