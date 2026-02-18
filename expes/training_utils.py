import torch
from pathcond.rescaling_polyn import reweight_model, optimize_rescaling_polynomial

def train_one_epoch(model, loader, criterion, optimizer, device, fraction: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    max_batches = int(len(loader) * fraction)
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

def train_one_epoch_enc_dec(model, loader, criterion, optimizer, device, fraction: float = 1.0) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0
    max_batches = int(len(loader) * fraction)
    
    # Note: On ne fait plus model.to(device) ici, on suppose qu'il y est déjà.
    start_fwd = torch.cuda.Event(enable_timing=True)
    end_fwd = torch.cuda.Event(enable_timing=True)
    start_bwd = torch.cuda.Event(enable_timing=True)
    end_bwd = torch.cuda.Event(enable_timing=True)
    timer_forward = 0.0
    timer_backward = 0.0
    nb_measures = 0
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        # Transfert asynchrone pour gagner un peu de temps sur GPU
        x = x.to(device)
        x = x.view(x.size(0), -1)
        optimizer.zero_grad()
        start_fwd.record()
        output = model(x)
        end_fwd.record()
        loss = criterion(output, x)  # Reconstruction loss
        start_bwd.record()
        loss.backward()
        end_bwd.record()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)
        timer_forward += start_fwd.elapsed_time(end_fwd) / 1000.0
        timer_backward += start_bwd.elapsed_time(end_bwd) / 1000.0
    return total_loss / total_samples, timer_forward, timer_backward

def evaluate_enc_dec(model, loader, device) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = torch.nn.MSELoss()  # Sum to accumulate total loss

    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            x = x.view(x.size(0), -1)

            output = model(x)
            loss = criterion(output, x)  # Reconstruction loss
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

    return total_loss / total_samples  # Return average loss

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

def rescaling_path_cond(model, nb_iter_optim_rescaling=10, device="cpu", data="moons", enorm=False, is_resnet_c=False) -> tuple[torch.nn.Module, torch.Tensor]:
    BZ, Z = optimize_rescaling_polynomial(model, n_iter=nb_iter_optim_rescaling, tol=1e-2, enorm=enorm, resnet=is_resnet_c)
    new_model = reweight_model(model, BZ, Z, enorm=enorm)  
    return new_model, Z