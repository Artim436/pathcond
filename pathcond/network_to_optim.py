import torch
import torch.nn as nn
from tqdm import tqdm
from pathcond.utils import _param_start_offsets, split_sorted_by_column, get_model_input_size
from torchvision.models.resnet import BasicBlock
from pathcond.utils import count_hidden_channels_generic, iter_modules_by_type




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


def grad_path_norm(model, device="cpu") -> torch.Tensor:
    """
    Compute the gradient of the path-norm with respect to all parameters
    of the model.
    Args:
        model (torch.nn.Module): Input model.
        input_size (tuple): Size of the input tensor (C, H, W).
        device (str, optional): Device to perform computations on.
        Defaults to "cpu".
    """
    input_size = get_model_input_size(model)
    inputs = torch.ones(input_size, device=device)
    model.eval()
    inputs = inputs.to(device)

    def fct(model, inputs, device="cpu"):
        model = model.to(device)
        return model(inputs).sum().to(device)
    grad = torch.autograd.grad(fct(model, inputs, device=device), model.parameters())
    grad = [g.view(-1) for g in grad]  # Aplatir les gradients
    return torch.cat(grad)  # Concaténer les gradients en un seul tenseur


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


def compute_diag_G(model):
    """
    Compute the diagonal of the G matrix for the given model.
    G_{ii} = H_{ii}/2
    where H is the Hessian of the function path-norm 2.
    Args:
        model (torch.nn.Module): Input model.
    Returns:
        torch.Tensor: Diagonal of the G matrix.
    """
    orig_w = set_weights_for_path_norm(model, exponent=2, provide_original_weights=True)
    res = grad_path_norm(model, device=next(model.parameters()).device)
    reset_model(model, orig_w)
    return res


def hessian_2(model, inputs):
    """
    Computes the Hessian of the path-norm 2 of the given model using
    automatic differentiation.
    Args:
        model (torch.nn.Module): Input model.
        inputs (torch.Tensor): Input tensor for the model.
    Returns:
        torch.Tensor: Hessienne de f.
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
    off_diag_mask = ~torch.eye(G.shape[0], dtype=bool, device=G.device)  # True hors daig et Flase sur la diag
    G[off_diag_mask] = G[off_diag_mask] / 4.0

    return G


def compute_B_mlp(model: nn.Module):
    linear_layers = [m for m in model.modules() if isinstance(m, nn.Linear)]
    if len(linear_layers) < 2:
        raise ValueError("Au moins deux nn.Linear sont nécessaires.")

    starts, n_params = _param_start_offsets(model)
    n_hidden = sum(layer.out_features for layer in linear_layers[:-1])
    device = next(model.parameters()).device

    # On construit d'abord 2 grands vecteurs
    pos_rows_all = []
    pos_cols_all = []
    neg_rows_all = []
    neg_cols_all = []

    col_start = 0
    for l, (layer, next_layer) in enumerate(zip(linear_layers[:-1], linear_layers[1:])):
        o = layer.out_features
        i = layer.in_features
        on, inn = next_layer.weight.shape
        assert inn == o

        col_end = col_start + o

        w0 = layer.weight        # [o, i]
        w0_start, _ = starts[id(w0)]

        # Lignes du bloc W0 : shape (o, i)
        rows_w0 = w0_start + (torch.arange(o, device=device)[:, None] * i
                              + torch.arange(i, device=device)[None, :])

        # Colonnes du bloc : shape (o, i)
        cols_w0 = col_start + torch.arange(o, device=device)[:, None].expand(o, i)

        neg_rows_all.append(rows_w0.reshape(-1))
        neg_cols_all.append(cols_w0.reshape(-1))

        if layer.bias is not None:
            b0 = layer.bias
            b0_start, _ = starts[id(b0)]

            rows_b0 = b0_start + torch.arange(o, device=device)
            cols_b0 = col_start + torch.arange(o, device=device)

            neg_rows_all.append(rows_b0)
            neg_cols_all.append(cols_b0)

        w1 = next_layer.weight
        w1_start, _ = starts[id(w1)]

        # Lignes du bloc W1 : shape (o, on)
        rows_w1 = w1_start + torch.arange(o, device=device)[:, None] \
            + torch.arange(on, device=device)[None, :] * o

        # Colonnes du bloc : shape (o, on)
        cols_w1 = col_start + torch.arange(o, device=device)[:, None].expand(o, on)

        pos_rows_all.append(rows_w1.reshape(-1))
        pos_cols_all.append(cols_w1.reshape(-1))

        col_start = col_end

    # Concat uniques
    pos_rows_all = torch.cat(pos_rows_all)
    pos_cols_all = torch.cat(pos_cols_all)
    neg_rows_all = torch.cat(neg_rows_all)
    neg_cols_all = torch.cat(neg_cols_all)
    neg_cols_all, idx = torch.sort(neg_cols_all)  # trier les colonnes négatives
    neg_rows_all = neg_rows_all[idx]

    # Maintenant on sépare par colonne (H colonnes)
    # On veut : liste[t] = indices des rows où col == t
    # pos_rows_all, pos_cols_all sont déjà triés !
    pos_cols = split_sorted_by_column(pos_cols_all, pos_rows_all, n_hidden)
    neg_cols = split_sorted_by_column(neg_cols_all, neg_rows_all, n_hidden)

    return pos_cols, neg_cols


def compute_B_cnn(model: nn.Module):
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
    neg_cols_all, idx = torch.sort(neg_cols_all)  # trier les colonnes négatives
    neg_rows_all = neg_rows_all[idx]
    pos_cols_all, idx = torch.sort(pos_cols_all)  # trier les colonnes positives
    pos_rows_all = pos_rows_all[idx]
    pos_cols = split_sorted_by_column(pos_cols_all, pos_rows_all, n_hidden)
    neg_cols = split_sorted_by_column(neg_cols_all, neg_rows_all, n_hidden)

    return pos_cols, neg_cols
