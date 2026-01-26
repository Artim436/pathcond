import torch
from torch import nn
from typing import Optional, Callable, Dict, Tuple, Union
from pathlib import Path
from torchvision.models.resnet import BasicBlock


from typing import Tuple, Union
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    

def _get_first_parametric_layer(module: nn.Module) -> nn.Module:
    """
    Recursively finds the first Linear or Conv2d layer in a module.
    """
    for child in module.children():
        if isinstance(child, (nn.Linear, nn.Conv2d)):
            return child
        # recurse
        found = _get_first_parametric_layer(child)
        if found is not None:
            return found
    return None


def get_model_input_size(
    model: nn.Module, 
    default_image_hw: Tuple[int, int] = (224, 224)
) -> Union[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Determines the expected input size for a PyTorch model by inspecting its
    first parametric layer (Linear or Conv2d), recursively.
    """

    first_layer = _get_first_parametric_layer(model)

    if first_layer is None:
        raise ValueError("Could not find a Linear or Conv2d layer in the model.")

    # --- Case 1: MLP / Linear ---
    if isinstance(first_layer, nn.Linear):
        return (1, first_layer.in_features)

    # --- Case 2: CNN / Conv2d ---
    if isinstance(first_layer, nn.Conv2d):
        H, W = default_image_hw
        return (1, first_layer.in_channels, H, W)

    # (Théoriquement inatteignable)
    raise ValueError(
        f"Unsupported layer type: {type(first_layer).__name__}"
    )


def _detect_model_type(model: nn.Module) -> str:
    """Détecte si le modèle est un MLP ou un CNN/ResNet."""
    is_mlp = False
    is_cnn = False
    for module in model.modules():
        if isinstance(module, nn.Linear):
            is_mlp = True  
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            is_cnn = True
            break    
    if is_cnn:
        return "CNN"
    elif is_mlp:
        return "MLP"
    else:
        return "UNKNOWN"

def count_hidden_channels_generic(model, module_type=None) -> int:
    """
    Compte les canaux de sortie (out_channels) de la première convolution
    de TOUS les BasicBlock trouvés dans un modèle.
    Ceci fonctionnera pour ResNet-18, 34, ou tout modèle utilisant BasicBlock.
    """
    total_channels = 0

    # 1. Utilisation de la fonction générique pour itérer sur tous les BasicBlock
    for name, block in iter_modules_by_type(model, module_type or BasicBlock):

        # 2. Le reste de votre logique reste la même
        #    'block' est ici l'instance de BasicBlock
        if hasattr(block, 'conv1') and isinstance(block.conv1, nn.Conv2d):
            total_channels += block.conv1.out_channels
        else:
            # Sécurité si un bloc BasicBlock n'a pas l'attribut 'conv1'
            print(f"Avertissement: Le bloc {name} ne possède pas l'attribut 'conv1' de type Conv2d.")

    return total_channels

def count_hidden_channels_full_conv(model) -> int:
    """
    Compte le nombre de hidden channels dans un CNN.
    
    
    On compte simplement les out_channels de chaque Conv2d 
    sauf la dernière couche.
    """
    
    # Extraire toutes les couches Conv2d dans l'ordre
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append({
                'name': name,
                'module': module,
                'out_channels': module.out_channels,
                'in_channels': module.in_channels
            })

    total_hidden = 0
    
    for i, conv in enumerate(conv_layers):
        is_last = (i == len(conv_layers) - 1)
        if not is_last:
            total_hidden += conv['out_channels']
 
    return total_hidden

def count_hidden_channels_resnet_c(model) -> int:
    """
    Compte le nombre de hidden channels dans un ResNet de type C.
    
    Selon la méthode décrite dans le papier, pour chaque bloc k:
    1. Left-rescaling: entre blocs k-1 et k (rescale l'entrée du bloc)
    2. Internal rescaling: entre Conv1 et Conv2 du bloc k
    3. Right-rescaling: entre blocs k et k+1 (rescale la sortie du bloc)
    
    Hidden channels:
    - Entre bloc k-1 et bloc k: nombre de canaux en sortie du bloc k-1
    - Entre Conv1 et Conv2 dans un bloc: out_channels de Conv1
    
    Total = (nombre de blocs + 1) rescalings inter-blocs + (nombre de blocs) rescalings internes
    """
    import torch.nn as nn
    
    total_hidden = 0
    
    # Extraire la structure des layers
    layers = []
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, name):
            layer = getattr(model, name)
            layers.append(layer)
    
    # 1. Rescaling après conv1 initial (avant layer1)
    if hasattr(model, 'conv1'):
        total_hidden += model.conv1.out_channels
    
    # 2. Pour chaque layer, compter les rescalings
    for layer_idx, layer in enumerate(layers):
        num_blocks = len(layer)
        
        for block_idx in range(num_blocks):
            block = layer[block_idx]
            
            # Rescaling interne: entre conv1 et conv2 du bloc
            if hasattr(block, 'conv1'):
                total_hidden += block.conv1.out_channels
            
            # Rescaling entre blocs (sauf pour le dernier bloc de la dernière layer)
            if not (layer_idx == len(layers) - 1 and block_idx == num_blocks - 1):
                if hasattr(block, 'conv2'):
                    total_hidden += block.conv2.out_channels
    
    return total_hidden



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


def iter_cnn_blocks(model: nn.Module):
    """
    Yields tuples (conv1, bn1, conv2) for supported CNN blocks.
    """
    for m in model.modules():

        # ResNet BasicBlock
        if hasattr(m, "conv1") and hasattr(m, "bn1") and hasattr(m, "conv2"):
            if isinstance(m.conv1, nn.Conv2d) and isinstance(m.conv2, nn.Conv2d):
                yield m.conv1, m.bn1, m.conv2

        # UNet DoubleConv
        elif isinstance(m, DoubleConv):
            conv1, bn1, _, conv2, _, _ = m.net
            yield conv1, bn1, conv2



def _ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p


@torch.no_grad()
def rebuild_optimizer_with_state_from_old(
    old_model: nn.Module,
    new_model: nn.Module,
    old_opt: torch.optim.Optimizer,
    *,
    opt_ctor: Optional[Callable[..., torch.optim.Optimizer]] = None,
    override_hparams: Optional[Dict] = None,
    allow_partial_slice: bool = False,   # True => copie coin supérieur gauche si shape a grandi
) -> torch.optim.Optimizer:
    """
    Recrée un optimizer pour new_model et transfère l'état par correspondance de noms.
    Gère Adam/AdamW (exp_avg, exp_avg_sq, step) et SGD(momentum) (momentum_buffer), etc.

    - opt_ctor: constructeur de l'optimizer (par défaut, même classe que old_opt)
    - override_hparams: pour changer lr/betas/eps/weight_decay… si besoin
    - allow_partial_slice: si True et shapes différentes, on copie par slice là où possible
    """
    if opt_ctor is None:
        opt_ctor = old_opt.__class__

    # 1) Hyperparams (lr, betas, eps, weight_decay, etc.)
    hp = dict(old_opt.defaults)
    if override_hparams:
        hp.update(override_hparams)

    # 2) Créer le nouvel optimizer
    new_opt = opt_ctor(new_model.parameters(), **hp)

    # 3) Index param par nom
    old_named = dict(old_model.named_parameters())
    new_named = dict(new_model.named_parameters())

    # 4) Copier état
    old_state = old_opt.state
    new_state = new_opt.state

    def _compatible_copy(t_old: torch.Tensor, t_new: torch.Tensor) -> torch.Tensor:
        """Copie t_old -> shape de t_new (identique, ou slice si allow_partial_slice=True)."""
        if t_old.shape == t_new.shape:
            return t_old.detach().clone().to(t_new.device)
        if allow_partial_slice:
            # Copie sur l'intersection des dimensions, le reste reste aux init de new_state
            out = torch.zeros_like(t_new, device=t_new.device)
            slices = tuple(slice(0, min(a, b)) for a, b in zip(t_old.shape, t_new.shape))
            out[slices] = t_old[slices].to(t_new.device)
            return out
        # shapes différentes non autorisées
        raise RuntimeError("Incompatible shapes without allow_partial_slice")

    copied, skipped = 0, 0
    for name, p_new in new_named.items():
        p_old = old_named.get(name)
        if p_old is None or p_old not in old_state:
            skipped += 1
            continue

        st_old = old_state[p_old]
        st_new = {}

        ok = True
        for k, v in st_old.items():
            if torch.is_tensor(v):
                try:
                    st_new[k] = _compatible_copy(v, torch.zeros_like(p_new) if v.shape == p_old.shape else v)
                    # Remap propre: si c'est un buffer de même shape que p_old, on le transpose vers p_new
                    if v.shape == p_old.shape:
                        st_new[k] = _compatible_copy(v, p_new)  # même shape que param => miroir
                    else:
                        # ex. buffers vectoriels (RMSprop: square_avg) / Adam: exp_avg, exp_avg_sq
                        # si même shape que p_old -> déjà géré; sinon on garde shape identique si allow_partial_slice True
                        pass
                except RuntimeError:
                    ok = False
                    break
            else:
                # ex. step (int), amsgrad flags, etc.
                st_new[k] = v

        if ok:
            new_state[p_new] = st_new
            copied += 1
        else:
            skipped += 1

    # 5) Copier les param_groups (lr, weight_decay, etc.)
    for g_old, g_new in zip(old_opt.param_groups, new_opt.param_groups):
        for k, v in g_old.items():
            if k != "params":
                g_new[k] = v

    print(f"Optimizer state transfer: copied={copied}, skipped={skipped}")
    return new_opt


def _param_start_offsets(model):
    """
    Retourne un dict {id(param_tensor): (start, end)} correspondant
    aux tranches [start:end) dans la vectorisation officielle PyTorch
    (parameters_to_vector), ce qui évite tout calcul d'offset fragile.
    """
    starts = {}
    offset = 0
    for p in model.parameters():
        n = p.numel()
        starts[id(p)] = (offset, offset + n)
        offset += n
    return starts, offset  # offset final == n_params


def split_sorted_by_column(col_all: torch.Tensor,
                           row_all: torch.Tensor,
                           n_hidden: int):
    """
    col_all est déjà trié !
    row_all a la même taille.
    On renvoie une liste de n_hidden tensors row_all[col == h],
    en exploitant les segments consécutifs.
    """

    # 1) Nombre d'éléments par colonne
    counts = torch.bincount(col_all, minlength=n_hidden)

    # 2) Début et fin des segments
    ends = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0], device=col_all.device), ends[:-1]])

    # 3) Découpage O(H)
    out = [row_all[starts[h]: ends[h]] for h in range(n_hidden)]

    return out
