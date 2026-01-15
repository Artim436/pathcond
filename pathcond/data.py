from __future__ import annotations
from typing import Tuple
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons


def mnist_loaders(batch_size: int = 128, num_workers: int = 2, seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    torch.manual_seed(seed)
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size*2, shuffle=True, num_workers=num_workers)
    return train_dl, test_dl


def cifar10_loaders(batch_size: int = 128, num_workers: int = 2, seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    torch.manual_seed(seed)
    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size*2, shuffle=True, num_workers=num_workers)
    return train_dl, test_dl

def moons_loaders(seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    X, y = make_moons(n_samples=1000, noise=0.1, random_state=seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, X_test, y_train, y_test

class SyntheticDeblurDataset(Dataset):
    def __init__(self, train: bool = True, seed: int = 0):
        torch.manual_seed(seed)
        self.dataset = datasets.CIFAR10(root='./data', train=train, download=True)
        self.blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.5))
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, _ = self.dataset[idx]
        img = img.convert('RGB')
        sharp = self.to_tensor(img)
        blurred_img = self.blur(img)
        blurry = self.to_tensor(blurred_img)
        return blurry, sharp
    
def get_deblur_loaders(batch_size: int = 128, num_workers: int = 2, seed: int = 0) -> Tuple[DataLoader, DataLoader]:
    torch.manual_seed(seed)
    train_set = SyntheticDeblurDataset(train=True, seed=seed)
    test_set = SyntheticDeblurDataset(train=False, seed=seed)
    train_dl = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl  = DataLoader(test_set,  batch_size=batch_size*2, shuffle=True, num_workers=num_workers)
    return train_dl, test_dl
    