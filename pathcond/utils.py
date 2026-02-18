import torch
from torch import nn
from typing import Optional, Callable, Dict, Tuple, Union
from pathlib import Path
from torchvision.models.resnet import BasicBlock

class MLPCompatibilityWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        layers = []
        linears = []
        for m in original_model.modules():
            if isinstance(m, nn.Linear):
                layers.append(m)
                linears.append(m)
            elif isinstance(m, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)):
                layers.append(m)
        self.model = nn.Sequential(*layers)
        self.input_dim = linears[0].in_features
        self.output_dim = linears[-1].out_features

    def forward(self, x):
        return self.original_model(x)


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
    default_image_hw: Tuple[int, int] = (32, 32)
) -> Union[Tuple[int, int], Tuple[int, int, int, int]]:
    """
    Determines the expected input size for a PyTorch model by inspecting its
    first parametric layer (Linear or Conv2d), recursively.
    """

    first_layer = _get_first_parametric_layer(model)

    if first_layer is None:
        raise ValueError("Could not find a Linear or Conv2d layer in the model.")
    if isinstance(first_layer, nn.Linear):
        return (1, first_layer.in_features)
    if isinstance(first_layer, nn.Conv2d):
        H, W = default_image_hw
        return (1, first_layer.in_channels, H, W)

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
    """
    total_channels = 0

    for name, block in iter_modules_by_type(model, module_type or BasicBlock):
        if hasattr(block, 'conv1') and isinstance(block.conv1, nn.Conv2d):
            total_channels += block.conv1.out_channels
        else:
            print(f"Avertissement: Le bloc {name} ne possède pas l'attribut 'conv1' de type Conv2d.")

    return total_channels

def count_hidden_channels_full_conv(model) -> int:
    """
    Compte le nombre de hidden channels dans un CNN.
    On compte les out_channels de chaque Conv2d 
    sauf la dernière couche.
    """
    
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
    
    Total = (nombre de blocs + 1) rescalings inter-blocs + (nombre de blocs) rescalings internes
    """
    import torch.nn as nn
    
    total_hidden = 0
    
    layers = []
    for name in ['layer1', 'layer2', 'layer3', 'layer4']:
        if hasattr(model, name):
            layer = getattr(model, name)
            layers.append(layer)
    
    if hasattr(model, 'conv1'):
        total_hidden += model.conv1.out_channels
    
    for layer_idx, layer in enumerate(layers):
        num_blocks = len(layer)
        
        for block_idx in range(num_blocks):
            block = layer[block_idx]
            
            if hasattr(block, 'conv1'):
                total_hidden += block.conv1.out_channels
            
            if not (layer_idx == len(layers) - 1 and block_idx == num_blocks - 1):
                if hasattr(block, 'conv2'):
                    total_hidden += block.conv2.out_channels
    
    return total_hidden



def iter_modules_by_type(model, module_type):
    for name, module in model.named_modules():
        if isinstance(module, module_type):
            if module is not model:
                yield name, module


def iter_cnn_blocks(model: nn.Module):
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




def _param_start_offsets(model):
    starts = {}
    offset = 0
    for p in model.parameters():
        n = p.numel()
        starts[id(p)] = (offset, offset + n)
        offset += n
    return starts, offset


def split_sorted_by_column(col_all: torch.Tensor,
                           row_all: torch.Tensor,
                           n_hidden: int):

    counts = torch.bincount(col_all, minlength=n_hidden)
    ends = torch.cumsum(counts, dim=0)
    starts = torch.cat([torch.tensor([0], device=col_all.device), ends[:-1]])
    out = [row_all[starts[h]: ends[h]] for h in range(n_hidden)]

    return out
