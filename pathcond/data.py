from __future__ import annotations
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch


def mnist_loaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    torch.manual_seed(0)
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size*2, shuffle=True, num_workers=num_workers)
    return train_dl, test_dl


def cifar10_loaders(batch_size: int = 128, num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    torch.manual_seed(0)
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size*2, shuffle=True, num_workers=num_workers)
    return train_dl, test_dl
