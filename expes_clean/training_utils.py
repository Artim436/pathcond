import torch
from pathcond.rescaling_polyn import reweight_model, optimize_rescaling_polynomial

def train_one_epoch(model, loader, criterion, optimizer, device, fraction: float = 1.0) -> float:
    """
    Entraîne le modèle sur une fraction du dataset.

    Args:
        model: nn.Module
        loader: DataLoader
        criterion: fonction de perte
        optimizer: Optimizer
        device: cuda ou cpu
        fraction: fraction du dataset à utiliser (0 < fraction <= 1.0)

    Returns:
        float: perte moyenne sur les échantillons vus
    """
    model.train()
    total_loss = 0.0
    total_samples = 0

    # nombre de batches maximum à parcourir
    max_batches = int(len(loader) * fraction)
    model = model.to(device)

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
       
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model = model.to(device)
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total


def rescaling_path_cond(model, nb_iter_optim_rescaling=10, device="cpu", data="moons", enorm=False):
    if data == "moons":
        input = torch.randn(1, 2).to(device)
    elif data == "mnist":
        input = torch.randn(1, 28*28).to(device)
    elif data == "cifar10":
        input = torch.randn(1, 3, 32, 32).to(device)
    else:
        raise NotImplementedError("Data not implemented")
    
    orig_output = model(input)
    BZ, Z = optimize_rescaling_polynomial(model, n_iter=nb_iter_optim_rescaling, tol=1e-6, enorm=enorm)

    return_model = reweight_model(model, BZ, Z, enorm=enorm)

    return_model = return_model.to(device).eval()
    new_output = return_model(input)
    assert torch.allclose(orig_output, new_output, atol=1e-5), "Outputs differ after rescaling!"
    return return_model
