import torch
import torch.nn as nn
from tqdm import tqdm
from pathcond.utils import _param_start_offsets, split_sorted_by_column, get_model_input_size
from torchvision.models.resnet import BasicBlock
from pathcond.utils import count_hidden_channels_generic, iter_modules_by_type, count_hidden_channels_resnet_c, count_hidden_channels_full_conv


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x



# the next function is copied from https://github.com/agonon/pathnorm_toolkit, with the agreement of Antoine Gonon 
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
    def f(model, inputs):
        return model.forward_squared(inputs).sum()

    grad = torch.autograd.grad(f(model, inputs), model.parameters(), create_graph=True)
    grad_vec = torch.cat([g.contiguous().view(-1) for g in grad])

    hessian_rows = []
    print("Computing Hessian...")
    for i in tqdm(range(grad_vec.numel())):
        grad2 = torch.autograd.grad(grad_vec[i], model.parameters(), retain_graph=True)
        row = torch.cat([g.contiguous().view(-1) for g in grad2])
        hessian_rows.append(row)

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

        w0 = layer.weight 
        w0_start, _ = starts[id(w0)]

        rows_w0 = w0_start + (torch.arange(o, device=device)[:, None] * i
                              + torch.arange(i, device=device)[None, :])

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

        rows_w1 = w1_start + torch.arange(o, device=device)[:, None] \
            + torch.arange(on, device=device)[None, :] * o

        cols_w1 = col_start + torch.arange(o, device=device)[:, None].expand(o, on)

        pos_rows_all.append(rows_w1.reshape(-1))
        pos_cols_all.append(cols_w1.reshape(-1))

        col_start = col_end

    pos_rows_all = torch.cat(pos_rows_all)
    pos_cols_all = torch.cat(pos_cols_all)
    neg_rows_all = torch.cat(neg_rows_all)
    neg_cols_all = torch.cat(neg_cols_all)
    neg_cols_all, idx = torch.sort(neg_cols_all) 
    neg_rows_all = neg_rows_all[idx]


    pos_cols = split_sorted_by_column(pos_cols_all, pos_rows_all, n_hidden)
    neg_cols = split_sorted_by_column(neg_cols_all, neg_rows_all, n_hidden)

    return pos_cols, neg_cols


def compute_B_resnet(model: nn.Module, module_type=None):
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

    n_hidden = count_hidden_channels_generic(model, module_type=module_type)

    pos_rows_all = []
    pos_cols_all = []
    neg_rows_all = []
    neg_cols_all = []

    col = 0  

    for lname, block in iter_modules_by_type(model, module_type or BasicBlock):
        if not hasattr(block, 'bn1'):
            conv1, conv2 = block.conv1, block.conv2
            bn1 = None
        else:
            conv1, bn1, conv2 = block.conv1, block.bn1, block.conv2
        o = conv1.out_channels
        i = conv1.in_channels
        kH, kW = conv1.kernel_size

        w1 = conv1.weight
        w1_start, _ = starts[id(w1)]

        base = torch.arange(o * i * kH * kW, device=device)
        base = base.view(o, -1)   
        rows_in = (w1_start + base).reshape(-1)
        cols_in = torch.repeat_interleave(torch.arange(o, device=device), i * kH * kW)

        cols_in = cols_in + col

        neg_rows_all.append(rows_in)
        neg_cols_all.append(cols_in)

        if bn1 is not None:
            b1 = bn1.bias
            b1_start, _ = starts[id(b1)]
            rows_bn = torch.arange(b1_start, b1_start + o, device=device)
            cols_bn = torch.arange(col, col + o, device=device)

            neg_rows_all.append(rows_bn)
            neg_cols_all.append(cols_bn)

        if conv1.bias is not None:
            cb = conv1.bias
            cb_start, _ = starts[id(cb)]
            rows_cb = torch.arange(cb_start, cb_start + o, device=device)
            cols_cb = torch.arange(col, col + o, device=device)

            neg_rows_all.append(rows_cb)
            neg_cols_all.append(cols_cb)

        w2 = conv2.weight
        o2, o_in, kH2, kW2 = w2.shape
        assert o_in == o
        w2_start, _ = starts[id(w2)]

        base2 = torch.arange(o2 * o * kH2 * kW2, device=device)
        base2 = base2.view(o2, o, -1)        
        rows_out = (w2_start + base2).reshape(-1)

        cols_out = torch.arange(o, device=device).repeat_interleave(kH2 * kW2)
        cols_out = cols_out.repeat(o2) + col

        pos_rows_all.append(rows_out)
        pos_cols_all.append(cols_out)

        col += o  
    pos_rows_all = torch.cat(pos_rows_all)
    pos_cols_all = torch.cat(pos_cols_all)
    neg_rows_all = torch.cat(neg_rows_all)
    neg_cols_all = torch.cat(neg_cols_all)
    neg_cols_all, idx = torch.sort(neg_cols_all) 
    neg_rows_all = neg_rows_all[idx]
    pos_cols_all, idx = torch.sort(pos_cols_all)  
    pos_rows_all = pos_rows_all[idx]
    pos_cols = split_sorted_by_column(pos_cols_all, pos_rows_all, n_hidden)
    neg_cols = split_sorted_by_column(neg_cols_all, neg_rows_all, n_hidden)

    return pos_cols, neg_cols


def compute_B_full_conv(model: nn.Module):
    """
    Version sparse pour CNN générique:

    Retourne:
        pos_cols: liste de tenseurs, pos_cols[h] = indices où B[:,h] = +1
        neg_cols: liste de tenseurs, neg_cols[h] = indices où B[:,h] = -1
    """
    starts, n_params = _param_start_offsets(model)
    device = next(model.parameters()).device
    
    n_hidden = count_hidden_channels_full_conv(model)
    
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(module)
    
    pos_rows_all = []
    pos_cols_all = []
    neg_rows_all = []
    neg_cols_all = []
    
    col = 0  
    
    for idx in range(len(conv_layers) - 1):
        conv1 = conv_layers[idx]
        conv2 = conv_layers[idx + 1]
        
        o = conv1.out_channels  
        i = conv1.in_channels
        kH, kW = conv1.kernel_size
        
        w1 = conv1.weight
        w1_start, _ = starts[id(w1)]
        

        base = torch.arange(o * i * kH * kW, device=device)
        base = base.view(o, -1)  # (o, i*kH*kW)
        rows_in = (w1_start + base).reshape(-1)
        
        cols_in = torch.repeat_interleave(
            torch.arange(o, device=device), 
            i * kH * kW
        )
        cols_in = cols_in + col
        
        neg_rows_all.append(rows_in)
        neg_cols_all.append(cols_in)
        
        if conv1.bias is not None:
            b1 = conv1.bias
            b1_start, _ = starts[id(b1)]
            rows_b1 = torch.arange(b1_start, b1_start + o, device=device)
            cols_b1 = torch.arange(col, col + o, device=device)
            
            neg_rows_all.append(rows_b1)
            neg_cols_all.append(cols_b1)
        

        w2 = conv2.weight
        o2, o_check, kH2, kW2 = w2.shape
        assert o_check == o, f"Mismatch: conv1.out_channels={o} != conv2.in_channels={o_check}"
        
        w2_start, _ = starts[id(w2)]
        
        base2 = torch.arange(o2 * o * kH2 * kW2, device=device)
        base2 = base2.view(o2, o, -1) 
        rows_out = (w2_start + base2).reshape(-1)
        
        cols_out = torch.arange(o, device=device).repeat_interleave(kH2 * kW2)
        cols_out = cols_out.repeat(o2) + col
        
        pos_rows_all.append(rows_out)
        pos_cols_all.append(cols_out)
        
        col += o  
    
    pos_rows_all = torch.cat(pos_rows_all)
    pos_cols_all = torch.cat(pos_cols_all)
    neg_rows_all = torch.cat(neg_rows_all)
    neg_cols_all = torch.cat(neg_cols_all)
    
    neg_cols_all, idx = torch.sort(neg_cols_all)
    neg_rows_all = neg_rows_all[idx]
    pos_cols_all, idx = torch.sort(pos_cols_all)
    pos_rows_all = pos_rows_all[idx]
    
    pos_cols = split_sorted_by_column(pos_cols_all, pos_rows_all, n_hidden)
    neg_cols = split_sorted_by_column(neg_cols_all, neg_rows_all, n_hidden)
    
    return pos_cols, neg_cols


def compute_B_resnet_c(model):
    """
    Version optimisée pour ResNet type C selon la structure de rescaling:
    
    Retourne:
        pos_cols: liste de tenseurs, pos_cols[h] = indices où B[:,h] = +1
        neg_cols: liste de tenseurs, neg_cols[h] = indices où B[:,h] = -1
    """
    starts, n_params = _param_start_offsets(model)
    device = next(model.parameters()).device
    
    n_hidden = count_hidden_channels_resnet_c(model)
    
    pos_rows_all = []
    pos_cols_all = []
    neg_rows_all = []
    neg_cols_all = []
    
    col = 0 
    
    def add_conv_weights(conv, sign, col_offset, starts, device):
        """Ajoute les poids d'une conv avec le signe donné"""
        rows_list = []
        cols_list = []
        
        w = conv.weight
        o, i, kH, kW = w.shape
        w_start, _ = starts[id(w)]
        
        if sign == -1:  
            base = torch.arange(o * i * kH * kW, device=device).view(o, -1)
            rows = (w_start + base).reshape(-1)
            cols = torch.repeat_interleave(torch.arange(o, device=device), i * kH * kW) + col_offset
        else:  
            base = torch.arange(o * i * kH * kW, device=device).view(o, i, -1)
            rows = (w_start + base).reshape(-1)
            cols = torch.arange(i, device=device).repeat_interleave(kH * kW).repeat(o) + col_offset
        
        rows_list.append(rows)
        cols_list.append(cols)
        
        if conv.bias is not None and sign == -1:  
            b_start, _ = starts[id(conv.bias)]
            rows_list.append(torch.arange(b_start, b_start + o, device=device))
            cols_list.append(torch.arange(col_offset, col_offset + o, device=device))
        
        return rows_list, cols_list
    
    conv1_initial = model.conv1
    o_init = conv1_initial.out_channels
    
    rows_list, cols_list = add_conv_weights(conv1_initial, -1, col, starts, device)
    neg_rows_all.extend(rows_list)
    neg_cols_all.extend(cols_list)
    
    first_block = model.layer1[0]
    rows_list, cols_list = add_conv_weights(first_block.conv1, +1, col, starts, device)
    pos_rows_all.extend(rows_list)
    pos_cols_all.extend(cols_list)
    
    rows_list, cols_list = add_conv_weights(first_block.shortcut[0], +1, col, starts, device)
    pos_rows_all.extend(rows_list)
    pos_cols_all.extend(cols_list)
    
    col += o_init
    
    all_blocks = []
    for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
        layer = getattr(model, layer_name)
        for block in layer:
            all_blocks.append(block)
    
    for block_idx, block in enumerate(all_blocks):
        conv1 = block.conv1
        conv2 = block.conv2
        shortcut = block.shortcut[0]
        
        o_internal = conv1.out_channels
        
        w1 = conv1.weight
        o1, i1, kH1, kW1 = w1.shape
        w1_start, _ = starts[id(w1)]
        base = torch.arange(o1 * i1 * kH1 * kW1, device=device).view(o1, -1)
        rows = (w1_start + base).reshape(-1)
        cols = torch.repeat_interleave(torch.arange(o1, device=device), i1 * kH1 * kW1) + col
        neg_rows_all.append(rows)
        neg_cols_all.append(cols)
        
        if conv1.bias is not None:
            b1_start, _ = starts[id(conv1.bias)]
            neg_rows_all.append(torch.arange(b1_start, b1_start + o1, device=device))
            neg_cols_all.append(torch.arange(col, col + o1, device=device))
        
        rows_list, cols_list = add_conv_weights(conv2, +1, col, starts, device)
        pos_rows_all.extend(rows_list)
        pos_cols_all.extend(cols_list)
        
        col += o_internal
        
        if block_idx < len(all_blocks) - 1:
            o_right = conv2.out_channels
            
            w2 = conv2.weight
            o2, i2, kH2, kW2 = w2.shape
            w2_start, _ = starts[id(w2)]
            base = torch.arange(o2 * i2 * kH2 * kW2, device=device).view(o2, -1)
            rows = (w2_start + base).reshape(-1)
            cols = torch.repeat_interleave(torch.arange(o2, device=device), i2 * kH2 * kW2) + col
            neg_rows_all.append(rows)
            neg_cols_all.append(cols)
            
            if conv2.bias is not None:
                b2_start, _ = starts[id(conv2.bias)]
                neg_rows_all.append(torch.arange(b2_start, b2_start + o2, device=device))
                neg_cols_all.append(torch.arange(col, col + o2, device=device))
            
            w_skip = shortcut.weight
            o_skip, i_skip, kH_skip, kW_skip = w_skip.shape
            w_skip_start, _ = starts[id(w_skip)]
            base = torch.arange(o_skip * i_skip * kH_skip * kW_skip, device=device).view(o_skip, -1)
            rows = (w_skip_start + base).reshape(-1)
            cols = torch.repeat_interleave(torch.arange(o_skip, device=device), i_skip * kH_skip * kW_skip) + col
            neg_rows_all.append(rows)
            neg_cols_all.append(cols)
            
            if shortcut.bias is not None:
                b_skip_start, _ = starts[id(shortcut.bias)]
                neg_rows_all.append(torch.arange(b_skip_start, b_skip_start + o_skip, device=device))
                neg_cols_all.append(torch.arange(col, col + o_skip, device=device))
            
            next_block = all_blocks[block_idx + 1]
            rows_list, cols_list = add_conv_weights(next_block.conv1, +1, col, starts, device)
            pos_rows_all.extend(rows_list)
            pos_cols_all.extend(cols_list)
            
            rows_list, cols_list = add_conv_weights(next_block.shortcut[0], +1, col, starts, device)
            pos_rows_all.extend(rows_list)
            pos_cols_all.extend(cols_list)
            
            col += o_right
    
    if len(pos_rows_all) > 0:
        pos_rows_all = torch.cat(pos_rows_all)
        pos_cols_all = torch.cat(pos_cols_all)
    else:
        pos_rows_all = torch.tensor([], dtype=torch.long, device=device)
        pos_cols_all = torch.tensor([], dtype=torch.long, device=device)
    
    if len(neg_rows_all) > 0:
        neg_rows_all = torch.cat(neg_rows_all)
        neg_cols_all = torch.cat(neg_cols_all)
    else:
        neg_rows_all = torch.tensor([], dtype=torch.long, device=device)
        neg_cols_all = torch.tensor([], dtype=torch.long, device=device)
    
    if len(neg_cols_all) > 0:
        neg_cols_all, idx = torch.sort(neg_cols_all)
        neg_rows_all = neg_rows_all[idx]
    
    if len(pos_cols_all) > 0:
        pos_cols_all, idx = torch.sort(pos_cols_all)
        pos_rows_all = pos_rows_all[idx]
    
    pos_cols = split_sorted_by_column(pos_cols_all, pos_rows_all, n_hidden)
    neg_cols = split_sorted_by_column(neg_cols_all, neg_rows_all, n_hidden)
    
    return pos_cols, neg_cols