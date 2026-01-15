import torch
from pathcond.rescaling_polyn import reweight_model, optimize_rescaling_polynomial

def train_one_epoch(model, loader, criterion, optimizer, device, fraction: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    max_batches = int(len(loader) * fraction)
    
    # Note: On ne fait plus model.to(device) ici, on suppose qu'il y est déjà.

    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        # Transfert asynchrone pour gagner un peu de temps sur GPU
        x, y = x.to(device), y.to(device)
       
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        total_samples += y.size(0)

    return total_loss / total_samples

def train_one_epoch_full_batch(model, X, y, criterion, optimizer) -> float:
    # X et y doivent être sur le device AVANT l'appel
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate_full_batch(model, X, y) -> float:
    model.eval()
    pred = model(X).argmax(dim=1)
    return (pred == y).sum().item() / y.size(0)

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / total

def rescaling_path_cond(model, nb_iter_optim_rescaling=10, device="cpu", data="moons", enorm=False):
    # On crée l'input directement sur le bon device
    shape_map = {"moons": (1, 2), "mnist": (1, 784), "cifar10": (1, 3, 32, 32)}
    if data not in shape_map:
        raise NotImplementedError("Data not implemented")
    
    dummy_input = torch.randn(*shape_map[data], device=device)
    
    # On passe en eval pour l'optimisation du rescaling
    model.eval()
    with torch.no_grad():
        orig_output = model(dummy_input)
    
    # On optimise
    BZ, Z = optimize_rescaling_polynomial(model, n_iter=nb_iter_optim_rescaling, tol=1e-2, enorm=enorm)

    # On applique le reweighting (normalement in-place ou créant un nouveau modèle)
    new_model = reweight_model(model, BZ, Z, enorm=enorm)
    new_model.to(device) # Sécurité, mais devrait déjà y être
    
    with torch.no_grad():
        new_output = new_model(dummy_input)
    
    # Vérification numérique
    diff = torch.abs(orig_output - new_output).max()
    if diff > 1e-2:
        print(f"Warning: Outputs differ after rescaling! Max diff: {diff:.4f}")
        
    return new_model