from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class MNISTMLP(nn.Module):
    """Simple MLP for MNIST (28x28 -> 10 classes)."""
    def __init__(self, d_hidden1: int = 256, d_hidden2: int = 128, p_drop: float = 0.2, seed: int = 0):
        super(MNISTMLP, self).__init__()
        torch.manual_seed(seed)

        self.d_hidden1 = d_hidden1
        self.d_hidden2 = d_hidden2
        self.drop = nn.Dropout(p_drop)

        self.model = nn.Sequential(
            nn.Linear(784, d_hidden1, bias=True),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden1, d_hidden2, bias=True),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(d_hidden2, 10, bias=False),
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