from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.nn import init
import numpy as np
import torchvision.models as models
from typing import List
from collections import OrderedDict


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

        layers.append(nn.Linear(prev, self.output_dim, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)

class MLP_BN(nn.Sequential):
    def __init__(self, hidden_dims: List[int], seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        
        input_dim = hidden_dims[0]
        output_dim = hidden_dims[-1]
        hidden_dims_list = hidden_dims[1:-1]

        layers = OrderedDict()
        prev = input_dim

        for i, h in enumerate(hidden_dims_list):
            # Ajout de noms explicites : 'lin', 'bn', 'relu'
            layers[f"lin{i}"] = nn.Linear(prev, h, bias=False)
            layers[f"bn{i}"] = nn.BatchNorm1d(h) # "bn" est maintenant dans le nom
            layers[f"relu{i}"] = nn.ReLU()
            prev = h

        layers["output"] = nn.Linear(prev, output_dim, bias=True)

        # On passe l'OrderedDict au constructeur de nn.Sequential
        self.model = nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)
    
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


## FUll conv pierre stock

class FullyConvolutional(nn.Module):
    """
    Fully convolutional architecture for cifar10 datset as described in
    Gitman et al., 'Comparison of Batch Normalization and Weight Normalization
    Algorithms for the Large-scale Image Classification'.
    """

    def __init__(self, bias=False):
        super(FullyConvolutional, self).__init__()
        # first block
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        # second block
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        # third block
        self.conv7 = nn.Conv2d(256, 320, 3, 1, 1, bias=bias)
        self.conv8 = nn.Conv2d(320, 320, 1, 1, 0, bias=bias)
        self.conv9 = nn.Conv2d(320, 10, 1, 1, 0, bias=bias)
        self.pool3 = nn.AvgPool2d(8)
        # relu
        self.relu = nn.ReLU()
        # batch norm
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        # reset parameters
        # self.reset_parameters()

    # def reset_parameters(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.bn1(x)
        x = self.pool1(self.relu(x))

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        x = self.bn2(x)
        x = self.pool2(self.relu(x))

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.conv9(x)
        x = self.pool3(self.relu(x))

        return x.flatten(1)


class FullyConvolutional_without_bn(nn.Module):
    """
    Fully convolutional architecture for cifar10 datset as described in
    Gitman et al., 'Comparison of Batch Normalization and Weight Normalization
    Algorithms for the Large-scale Image Classification'.
    """

    def __init__(self, bias=False):
        super(FullyConvolutional_without_bn, self).__init__()
        # first block
        self.conv1 = nn.Conv2d(3, 128, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(128, 128, 3, 1, 1, bias=bias)
        self.pool1 = nn.MaxPool2d(3, 2, 1)
        # second block
        self.conv4 = nn.Conv2d(128, 256, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.conv6 = nn.Conv2d(256, 256, 3, 1, 1, bias=bias)
        self.pool2 = nn.MaxPool2d(3, 2, 1)
        # third block
        self.conv7 = nn.Conv2d(256, 320, 3, 1, 1, bias=bias)
        self.conv8 = nn.Conv2d(320, 320, 1, 1, 0, bias=bias)
        self.conv9 = nn.Conv2d(320, 10, 1, 1, 0, bias=bias)
        self.pool3 = nn.AvgPool2d(8)
        # relu
        self.relu = nn.ReLU()
        # batch norm
        # reset parameters
        # self.reset_parameters()

    # def reset_parameters(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        x = self.pool1(self.relu(x))

        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.conv6(x)
        x = self.pool2(self.relu(x))

        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.conv9(x)
        x = self.pool3(self.relu(x))

        return x.flatten(1)
    


class BasicBlock(nn.Module):
    """
    Bloc résiduel basique pour ResNet-18
    Utilise toujours une convolution 1x1 pour le shortcut (Type C)
    """
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # Première convolution 3x3
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Deuxième convolution 3x3
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Type C: TOUJOURS utiliser une convolution 1x1 pour le shortcut
        # Même quand in_channels == out_channels et stride == 1
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                     stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet18TypeC(nn.Module):
    """
    ResNet-18 de type C selon (He et al., 2015b)
    Tous les shortcuts utilisent des convolutions 1x1 apprises
    """
    def __init__(self, num_classes=1000):
        super(ResNet18TypeC, self).__init__()
        
        self.in_channels = 64
        
        # Couche initiale (conv1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, 
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Blocs résiduels
        # ResNet-18: [2, 2, 2, 2] blocs par couche
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Couche de classification
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
        # Initialisation des poids
        # self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        """Crée une couche avec plusieurs blocs résiduels"""
        layers = []
        
        # Premier bloc avec potentiellement stride > 1
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        
        # Blocs suivants
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(self.in_channels, out_channels, stride=1))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialise les poids selon la méthode de He et al."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Couche initiale
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Blocs résiduels
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
def resnet18_c_cifar10(num_classes=10, seed: int = 0) -> nn.Module:
    # Load normal ResNet-18
    torch.manual_seed(seed)
    model = ResNet18TypeC(num_classes=num_classes)

    # Replace first conv: 7x7 stride 2 → 3x3 stride 1, and adapt to 3 channels
    model.conv1 = nn.Conv2d(
        3, 64, kernel_size=3, stride=1, padding=1, bias=False
    )
    # Remove maxpool (too destructive for 32x32)
    model.maxpool = nn.Identity()

    return model


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


# -----------------------------
# Model: UNet
# -----------------------------

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
    

class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_c: int = 64, bilinear: bool = True):
        super().__init__()
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return torch.clamp(x, 0.0, 1.0)
    
    def squared_forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = _doubleconv_forward_sq(x, self.in_conv)
        x2 = _down_forward_sq(x1, self.down1)
        x3 = _down_forward_sq(x2, self.down2)
        x4 = _down_forward_sq(x3, self.down3)
        x5 = _down_forward_sq(x4, self.down4)

        x = _up_forward_sq(x5, x4, self.up1)
        x = _up_forward_sq(x,  x3, self.up2)
        x = _up_forward_sq(x,  x2, self.up3)
        x = _up_forward_sq(x,  x1, self.up4)

        x = _outconv_forward_sq(x, self.out_conv)
        return torch.clamp(x, 0.0, 1.0)
    
    # UNet.squared_forward = squared_forward


def grad_2(model, inputs):
    def fct(model, inputs):
        return model.squared_forward(inputs).sum()
    grad2 = torch.autograd.grad(fct(model, inputs), model.parameters(), create_graph=True)
    grad2 = [g.view(-1) for g in grad2]  # Aplatir les gradients
    return torch.cat(grad2)  # Concaténer les gradients en un seul tenseur

def grad(model, inputs):
    def fct(model, inputs):
        return model.forward(inputs).sum()
    grad = torch.autograd.grad(fct(model, inputs), model.parameters())
    grad = [g.view(-1) for g in grad]  # Aplatir les gradients
    return torch.cat(grad)  # Concaténer les gradients en un seul tenseur


class DoubleConvWOBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = self.relu(x)
        return x

class DownWOBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConvWOBN(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class UpWOBN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConvWOBN(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConvWOBN(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class UNetWOBN(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_c: int = 64, bilinear: bool = True):
        super().__init__()
        self.in_conv = DoubleConvWOBN(in_channels, base_c)
        self.down1 = DownWOBN(base_c, base_c * 2)
        self.down2 = DownWOBN(base_c * 2, base_c * 4)
        self.down3 = DownWOBN(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = DownWOBN(base_c * 8, base_c * 16 // factor)

        self.up1 = UpWOBN(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = UpWOBN(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = UpWOBN(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = UpWOBN(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_conv(x)
        return torch.clamp(x, 0.0, 1.0)
    
    def squared_forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = _doubleconv_forward_sq(x, self.in_conv)
        x2 = _down_forward_sq(x1, self.down1)
        x3 = _down_forward_sq(x2, self.down2)
        x4 = _down_forward_sq(x3, self.down3)
        x5 = _down_forward_sq(x4, self.down4)

        x = _up_forward_sq(x5, x4, self.up1)
        x = _up_forward_sq(x,  x3, self.up2)
        x = _up_forward_sq(x,  x2, self.up3)
        x = _up_forward_sq(x,  x1, self.up4)

        x = _outconv_forward_sq(x, self.out_conv)
        return torch.clamp(x, 0.0, 1.0)