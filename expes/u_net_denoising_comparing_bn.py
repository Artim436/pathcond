from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from pathcond.models import UNet, DoubleConv, UNetWOBN, DoubleConvWOBN
from pathcond.rescaling_polyn import optimize_rescaling_polynomial, reweight_model
from pathcond.utils import _ensure_outdir
import copy
import time
from pathcond.data import SyntheticDeblurDataset, get_deblur_loaders
from tqdm import tqdm
import numpy as np


# -----------------------------
# Synthetic Deblur Dataset (CIFAR10 based)
# -----------------------------


# -----------------------------
# Metrics
# -----------------------------

def psnr(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    mse = F.mse_loss(torch.clamp(pred, 0, 1), torch.clamp(target, 0, 1), reduction="none")
    mse = mse.view(mse.size(0), -1).mean(dim=1)
    psnr_vals = 10.0 * torch.log10(1.0 / (mse + eps))
    return psnr_vals.mean().item()

# -----------------------------
# Training and evaluation
# -----------------------------

def train_one_epoch(model, loader, optimizer, device, loss_fn, fraction: float = 1.0):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    max_batches = int(len(loader) * fraction)
    for x, y in loader:
        if max_batches <= 0:
            break
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_psnr += psnr(pred.detach(), y.detach())
    n = len(loader)
    return total_loss / n, total_psnr / n

@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    total_psnr = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_psnr += psnr(pred, y)
    return total_psnr / len(loader)


def rescaling_path_dynamics(model, device="cpu", module_type=DoubleConv):
    inputs = torch.randn(16, 3, 32, 32).to(device)
    BZ_opt, Z_opt = optimize_rescaling_polynomial(model, n_iter=5, tol=1e-6, module_type=module_type)
    final_model = reweight_model(model, BZ_opt, Z_opt, module_type=DoubleConv).to(dtype=torch.float32, device=device)
    final_model = final_model.to(device).eval()
    model.eval()

    final_output = final_model(inputs)
    original_output = model(inputs)
    assert torch.allclose(original_output, final_output, atol=1e-2)

    return final_model

def main():
    parser = argparse.ArgumentParser(description="Train UNet for synthetic CIFAR10 deblurring")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--nb-lr", type=int, default=10)
    parser.add_argument("--nb-iter", type=int, default=3)
    parser.add_argument("--frac", type=float, default=1.0)
    args = parser.parse_args()
    epochs = args.epochs
    nb_lr = args.nb_lr
    nb_iter = args.nb_iter
    batch_size = args.batch_size
    frac = args.frac
    base_channels = args.base_channels
    learning_rates = torch.logspace(-4, 0, nb_lr)
    PSNR_TRAIN = torch.zeros((nb_lr, nb_iter, epochs, 4))  # sgd, ref_sgd, sgd_wobn, ref_sgd_wobn
    LOSS = torch.zeros((nb_lr, nb_iter, epochs, 4))
    PSNR_TEST = torch.zeros((nb_lr, nb_iter, epochs, 4))
    TIME = torch.zeros((nb_lr, nb_iter, epochs, 4))
    # EPOCHS = torch.zeros((nb_lr, nb_iter, 1, 2))

    epochs_teleport = [0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    for lr_index, lr in tqdm(enumerate(learning_rates)):
        for it in range(nb_iter):
            torch.manual_seed(it)
            torch.cuda.manual_seed_all(it)
            np.random.seed(it)
            random.seed(it)

            train_loader, test_loader = get_deblur_loaders(batch_size=batch_size, seed=it)

            model = UNet(in_channels=3, out_channels=3, base_c=base_channels).to(device)
            model_rescaled = copy.deepcopy(model)
            model_WOBN = UNetWOBN(in_channels=3, out_channels=3, base_c=base_channels).to(device)
            model_rescaled_WOBN = copy.deepcopy(model_WOBN)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            optimizer_WOBN = torch.optim.SGD(model_WOBN.parameters(), lr=lr)

            loss_fn = nn.MSELoss()

            best_psnr = 0.0
            start_basline = time.time()
            for epoch in range(epochs):
                train_loss, train_psnr = train_one_epoch(model, train_loader, optimizer, device, loss_fn, fraction=frac)
                val_psnr = evaluate(model, test_loader, device)
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                PSNR_TRAIN[lr_index, it, epoch, 2] = train_psnr
                LOSS[lr_index, it, epoch, 2] = train_loss
                PSNR_TEST[lr_index, it, epoch, 2] = val_psnr
                TIME[lr_index, it, epoch, 2] = time.time() - start_basline
                # EPOCHS[lr_index, it, 0, 1] = epoch + 1
                print(f"[Baseline] it {it+1}/{nb_iter} | LR {lr:.1e} | Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | Test PSNR: {val_psnr:.2f} dB")
            end_basline = time.time()

            best_psnr_wobn = 0.0
            start_basline = time.time()
            for epoch in range(epochs):
                train_loss, train_psnr = train_one_epoch(model_WOBN, train_loader, optimizer_WOBN, device, loss_fn, fraction=frac)
                val_psnr = evaluate(model_WOBN, test_loader, device)
                if val_psnr > best_psnr_wobn:
                    best_psnr_wobn = val_psnr
                PSNR_TRAIN[lr_index, it, epoch, 3] = train_psnr
                LOSS[lr_index, it, epoch, 3] = train_loss
                PSNR_TEST[lr_index, it, epoch, 3] = val_psnr
                TIME[lr_index, it, epoch, 3] = time.time() - start_basline
                # EPOCHS[lr_index, it, 0, 1] = epoch + 1
                print(f"[Baseline WOBN] it {it+1}/{nb_iter} | LR {lr:.1e} | Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | Test PSNR: {val_psnr:.2f} dB")
            end_basline = time.time()
            

            loss_fn_rescaled = nn.MSELoss()
            time_teleport = 0.0
            best_psnr_pc = 0.0
            start_pathcond = time.time()
            for epoch in range(epochs):
                if epoch in epochs_teleport:
                    start = time.time()
                    model_rescaled = rescaling_path_dynamics(model_rescaled, device=device, module_type=DoubleConv)
                    end = time.time()
                    time_teleport += end - start
                    optimizer_rescaled = torch.optim.SGD(model_rescaled.parameters(), lr=lr)
                train_loss, train_psnr = train_one_epoch(model_rescaled, train_loader, optimizer_rescaled, device, loss_fn_rescaled, fraction=frac)
                val_psnr = evaluate(model_rescaled, test_loader, device)
                
                if val_psnr > best_psnr_pc:
                    best_psnr_pc = val_psnr
                PSNR_TRAIN[lr_index, it, epoch, 0] = train_psnr
                LOSS[lr_index, it, epoch, 0] = train_loss
                PSNR_TEST[lr_index, it, epoch, 0] = val_psnr
                TIME[lr_index, it, epoch, 0] = time.time() - start_pathcond
                # EPOCHS[lr_index, it, 0, 0] = epoch + 1
                print(f"[Rescaled] it {it+1}/{nb_iter} | LR {lr:.1e} | Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | Test PSNR: {val_psnr:.2f} dB")
            end_pathcond = time.time()

            optimizer_rescaled_WOBN = torch.optim.SGD(model_rescaled_WOBN.parameters(), lr=lr)
            loss_fn_rescaled = nn.MSELoss()
            time_teleport_wobn = 0.0
            best_psnr_pc_wobn = 0.0
            start_pathcond = time.time()
            for epoch in range(epochs):
                if epoch in epochs_teleport:
                    start = time.time()
                    model_rescaled_WOBN = rescaling_path_dynamics(model_rescaled_WOBN, device=device, module_type=DoubleConvWOBN)
                    end = time.time()
                    time_teleport_wobn += end - start
                    optimizer_rescaled_WOBN = torch.optim.SGD(model_rescaled_WOBN.parameters(), lr=lr)
                train_loss, train_psnr = train_one_epoch(model_rescaled_WOBN, train_loader, optimizer_rescaled_WOBN, device, loss_fn_rescaled, fraction=frac)
                val_psnr = evaluate(model_rescaled_WOBN, test_loader, device)
                
                if val_psnr > best_psnr_pc_wobn:
                    best_psnr_pc_wobn = val_psnr
                PSNR_TRAIN[lr_index, it, epoch, 1] = train_psnr
                LOSS[lr_index, it, epoch, 1] = train_loss
                PSNR_TEST[lr_index, it, epoch, 1] = val_psnr
                TIME[lr_index, it, epoch, 1] = time.time() - start_pathcond
                # EPOCHS[lr_index, it, 0, 0] = epoch + 1
                print(f"[Rescaled WOBN] it {it+1}/{nb_iter} | LR {lr:.1e} | Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | Test PSNR: {val_psnr:.2f} dB")
            end_pathcond = time.time()


            print(f"Training complete. Best Test PSNR: {best_psnr:.2f} dB")
            print(f"Training complete. Best WOBN Test PSNR: {best_psnr_wobn:.2f} dB")
            print(f"Training complete. Best Rescaled Test PSNR: {best_psnr_pc:.2f} dB")
            print(f"Training complete. Best Rescaled WOBN Test PSNR: {best_psnr_pc_wobn:.2f} dB")
            print(f"Baseline training time: {end_basline - start_basline:.2f} seconds")
            print(f"Path-Cond training time (including teleport): {end_pathcond - start_pathcond:.2f} seconds")
            print(f"Total teleportation time: {time_teleport:.2f} seconds")
            print(f"Total teleportation time WOBN: {time_teleport_wobn:.2f} seconds")

    torch.save({
        "PSNR_TRAIN": PSNR_TRAIN,
        "LOSS": LOSS,
        "PSNR_TEST": PSNR_TEST,
        "TIME": TIME,
    }, _ensure_outdir("results/unet/comparing_bn/") / "dic.pt")

if __name__ == "__main__":
    main()