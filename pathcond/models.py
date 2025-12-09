from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn import init
import numpy as np
import torchvision.models as models


class MNISTMLP(nn.Module):
    def __init__(self, d_hidden1=256, d_hidden2=128, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.d_hidden1 = d_hidden1
        self.d_hidden2 = d_hidden2

        self.model = nn.Sequential(
            nn.Linear(784, d_hidden1),
            nn.ReLU(),
            nn.Linear(d_hidden1, d_hidden2),
            nn.ReLU(),
            nn.Linear(d_hidden2, 10),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)


    def forward_squared(self, x, device='cpu'):
        ''' Forward by applying the rule "Linear -> weights**2, bias**2" '''
        x = x.to(device)
        x = x.view(x.size(0), -1)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                W = layer.weight ** 2
                b = layer.bias ** 2 if layer.bias is not None else None
                x = F.linear(x, W, b)
            else:
                x = layer(x)
        return x
    
    def init_normal(self, mean=0.0, std=0.02, seed=0):
        torch.manual_seed(seed)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class Moons_MLP(nn.Module):
    """MLP for two moons (2 -> 32 -> 32 -> 1)."""
    def __init__(self, d_hidden1: int = 32, d_hidden2: int = 32, seed: int = 0):
        super(Moons_MLP, self).__init__()
        torch.manual_seed(seed)

        self.d_hidden1 = d_hidden1
        self.d_hidden2 = d_hidden2

        self.model = nn.Sequential(
            nn.Linear(2, d_hidden1, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden1, d_hidden2, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden2, 2, bias=False),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)      

    def forward_squared(self, x, device='cpu'):
        ''' Forward by applying the rule "Linear -> weights**2, bias**2" '''
        x = x.to(device)
        x = x.view(x.size(0), -1)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                W = layer.weight ** 2
                b = layer.bias ** 2 if layer.bias is not None else None
                x = F.linear(x, W, b)
            else:
                x = layer(x)
        return x
    
    def init__weights_normal(self, mean=0.0, std=0.02, seed=0):
        torch.manual_seed(seed)
        for m in self.model.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=mean, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
class Moons_MLP_unbalanced(nn.Module):
    """MLP for two moons (2 -> 32 -> 32 -> 1)."""
    def __init__(self, d_hidden1: int = 32, d_hidden2: int = 32, seed: int = 0):
        super(Moons_MLP_unbalanced, self).__init__()
        torch.manual_seed(seed)

        self.d_hidden1 = d_hidden1
        self.d_hidden2 = d_hidden2

        self.model = nn.Sequential(
            nn.Linear(2, d_hidden1, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden1, d_hidden2, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden2, d_hidden1, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden1, d_hidden2, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden2, d_hidden1, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden1, 2, bias=False),
        )

    def forward(self, x, device='cpu'):
        x = x.to(device)
        x = x.view(x.size(0), -1)
        return self.model(x)

    def forward_squared(self, x, device='cpu'):
        ''' Forward by applying the rule "Linear -> weights**2, bias**2" '''
        x = x.to(device)
        x = x.view(x.size(0), -1)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                W = layer.weight ** 2
                b = layer.bias ** 2 if layer.bias is not None else None
                x = F.linear(x, W, b)
            else:
                x = layer(x)
        return x
    

   
class toy_MLP(nn.Module):
    """Toy MLP for testing (10 -> 32 -> 1)."""
    def __init__(self, d_input: int = 2, d_hidden1: int = 2, seed: int = 0, teacher_init: bool = False):
        super(toy_MLP, self).__init__()
        torch.manual_seed(seed)

        self.d_hidden1 = d_hidden1
        self.d_input = d_input

        self.model = nn.Sequential(
            nn.Linear(d_input, d_hidden1, bias=True),
            nn.ReLU(),
            nn.Linear(d_hidden1, 1, bias=False),
        )
        if teacher_init:
            self._init_teacher_weights()

    def forward(self, x, device='cpu', return_activations: bool = False):
        z1 = self.model[0](x).to(device)  # pré-activation (avant ReLU)
        a1 = self.model[1](z1).to(device)  # activation (après ReLU)
        out = self.model[2](a1).to(device)  # sortie du modèle
        if return_activations:
            return out, a1
        else:
            return out

    def forward_squared(self, x, device='cpu'):
        ''' Forward by applying the rule "Linear -> weights**2, bias**2" '''
        x = x.to(device)
        x = x.view(x.size(0), -1)
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                W = layer.weight ** 2
                b = layer.bias ** 2 if layer.bias is not None else None
                x = F.linear(x, W, b)
            else:
                x = layer(x)
        return x

    def _init_teacher_weights(self):
        # mean of inputs
        with torch.no_grad():
            W1 = torch.zeros_like(self.model[0].weight)
            for i in range(min(self.d_input, self.d_hidden1)):
                W1[i, i] = 1.0
            self.model[0].weight.copy_(W1)
            self.model[0].bias.fill_(0.0)

            W2 = torch.zeros_like(self.model[2].weight)
            for i in range(self.d_hidden1):
                W2[0, i] = 1.0 / self.d_hidden1
            self.model[2].weight.copy_(W2)
    
    def get_mean_weights(self):
        with torch.no_grad():
            W1 = self.model[0].weight
            W2 = self.model[2].weight
            bias = self.model[0].bias
            mean_W1 = W1.mean().item()
            mean_W2 = W2.mean().item()
            mean_bias = bias.mean().item() if bias is not None else 0.0
        return mean_W1, mean_W2, mean_bias
    
    def get_weights(self):
        with torch.no_grad():
            W1 = self.model[0].weight.clone().detach()
            W2 = self.model[2].weight.clone().detach()
            bias = self.model[0].bias.clone().detach() if self.model[0].bias is not None else None
        return W1, W2, bias

    def get_weights_as_vector(self):
        return parameters_to_vector(self.parameters())


class MLP(nn.Sequential):
    def __init__(
        self,
        hidden_dims: List[int],
        seed: int = 0
    ) -> None:
        super().__init__()

        torch.manual_seed(seed)
        self.input_dim = hidden_dims[0]
        self.output_dim = hidden_dims[-1]
        self.hidden_dims = hidden_dims[1:-1]

        layers = []
        prev = self.input_dim

        # Hidden layers: Linear + ReLU, bias=True
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h, bias=True))
            layers.append(nn.ReLU())
            prev = h

        # Output layer: bias=False (comme dans ton code)
        layers.append(nn.Linear(prev, self.output_dim, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    

def apply_init(model: nn.Module, scheme: str, gain: float = None, a=0.05, std=0.02, seed=0) -> None:
    """
    scheme in {"xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal",
              "uniform", "normal", "zeros", "orthogonal"}
    """
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, gain=gain if gain is not None else 1.0)
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(m.bias, -bound, bound)
            elif scheme == "xavier_normal":
                nn.init.xavier_normal_(m.weight, gain=gain if gain is not None else 1.0)
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(m.bias, -bound, bound)
            elif scheme == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(m.bias, -bound, bound)
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(m.bias, -bound, bound)
            elif scheme == "uniform":
                nn.init.uniform_(m.weight, a=-a, b=a)
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
                    init.uniform_(m.bias, -bound, bound)
            elif scheme == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=std)
            elif scheme == "normal_wide":
                nn.init.normal_(m.weight, mean=0.0, std=std*5)
                if m.bias is not None:
                    nn.init.normal_(m.bias, mean=0.0, std=std*5)
            elif scheme == "default":
                pass  # laisse les poids tels quels
            elif scheme == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=gain if gain is not None else 1.0)
                if m.bias is not None:
                    nn.init.orthogonal_(m.bias, gain=gain if gain is not None else 1.0)
            elif scheme == "ones":
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.ones_(m.bias)
            elif scheme == "twos":
                nn.init.constant_(m.weight, 2.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 2.0)
            elif scheme == "constant_0.5":
                nn.init.constant_(m.weight, 0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.5)
            else:
                raise ValueError(f"Schéma d'init inconnu: {scheme}")



def resnet18_mnist(num_classes=10, seed: int = 0) -> nn.Module:
    # Load normal ResNet-18
    torch.manual_seed(seed)
    model = models.resnet18(weights=None)

    # Replace first conv: 7x7 stride 2 → 3x3 stride 1, and adapt to 1 channel
    model.conv1 = nn.Conv2d(
        1, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # Remove maxpool (too destructive for 28x28)
    model.maxpool = nn.Identity()

    # Replace final classification layer
    model.fc = nn.Linear(512, num_classes)

    return model

def resnet18_cifar10(num_classes=10, seed: int = 0) -> nn.Module:
    # Load normal ResNet-18
    torch.manual_seed(seed)
    model = models.resnet18(weights=None)

    # Replace first conv: 7x7 stride 2 → 3x3 stride 1, and adapt to 3 channels
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # Remove maxpool (too destructive for 32x32)
    model.maxpool = nn.Identity()

    # Replace final classification layer
    model.fc = nn.Linear(512, num_classes)

    return model