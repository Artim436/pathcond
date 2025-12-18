from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from pathcond.models import UNet, DoubleConv
from pathcond.rescaling_polyn import optimize_rescaling_polynomial, reweight_model
from pathcond.utils import _ensure_outdir
import copy
import time
from pathcond.data import SyntheticDeblurDataset
from tqdm import tqdm


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

def train_one_epoch(model, loader, optimizer, device, loss_fn):
    model.train()
    total_loss = 0.0
    total_psnr = 0.0
    for x, y in loader:
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
def evaluate(model, loader, device):
    model.eval()
    total_psnr = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x)
        total_psnr += psnr(pred, y)
    return total_psnr / len(loader)


def rescaling_path_dynamics(model, device="cpu"):
    inputs = torch.randn(16, 3, 32, 32).to(device)
    BZ_opt, Z_opt = optimize_rescaling_polynomial(model, n_iter=10, tol=1e-6, module_type=DoubleConv)
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
    args = parser.parse_args()
    epochs = args.epochs
    nb_lr = args.nb_lr
    nb_iter = args.nb_iter
    batch_size = args.batch_size
    base_channels = args.base_channels
    learning_rates = torch.logspace(-4, -1, nb_lr)
    PSNR_TRAIN = torch.zeros((nb_lr, nb_iter, epochs, 2))  # sgd, ref_sgd
    LOSS = torch.zeros((nb_lr, nb_iter, epochs, 2))
    PSNR_TEST = torch.zeros((nb_lr, nb_iter, epochs, 2))
    TIME = torch.zeros((nb_lr, nb_iter, 1, 2))
    EPOCHS = torch.zeros((nb_lr, nb_iter, 1, 2))

    epochs_teleport = [0]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    for lr_index, lr in tqdm(enumerate(learning_rates)):
        for it in range(nb_iter):
            torch.manual_seed(it)

            train_set = SyntheticDeblurDataset(train=True)
            test_set = SyntheticDeblurDataset(train=False)

            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_set, batch_size=batch_size*2, shuffle=False, num_workers=2)

            model = UNet(in_channels=3, out_channels=3, base_c=base_channels).to(device)
            model_copy = copy.deepcopy(model)
            optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            loss_fn = nn.MSELoss()

            best_psnr = 0.0
            start_basline = time.time()
            for epoch in range(epochs):
                train_loss, train_psnr = train_one_epoch(model, train_loader, optimizer, device, loss_fn)
                val_psnr = evaluate(model, test_loader, device)
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                PSNR_TRAIN[lr_index, it, epoch, 1] = train_psnr
                LOSS[lr_index, it, epoch, 1] = train_loss
                PSNR_TEST[lr_index, it, epoch, 1] = val_psnr
                TIME[lr_index, it, 0, 1] = time.time() - start_basline
                EPOCHS[lr_index, it, 0, 1] = epoch + 1
                print(f"[Baseline] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | Test PSNR: {val_psnr:.2f} dB")
            end_basline = time.time()
            

            loss_fn_rescaled = nn.MSELoss()
            time_teleport = 0.0
            best_psnr_pc = 0.0
            start_pathcond = time.time()
            for epoch in range(epochs):
                if epoch in epochs_teleport:
                    start = time.time()
                    model_rescaled = rescaling_path_dynamics(model_copy, device=device)
                    end = time.time()
                    time_teleport += end - start
                    optimizer_rescaled = torch.optim.SGD(model_rescaled.parameters(), lr=lr)
                train_loss, train_psnr = train_one_epoch(model_rescaled, train_loader, optimizer_rescaled, device, loss_fn_rescaled)
                val_psnr = evaluate(model_rescaled, test_loader, device)
                
                if val_psnr > best_psnr_pc:
                    best_psnr_pc = val_psnr
                PSNR_TRAIN[lr_index, it, epoch, 0] = train_psnr
                LOSS[lr_index, it, epoch, 0] = train_loss
                PSNR_TEST[lr_index, it, epoch, 0] = val_psnr
                TIME[lr_index, it, 0, 0] = time.time() - start_pathcond
                EPOCHS[lr_index, it, 0, 0] = epoch + 1
                print(f"[Rescaled] Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Train PSNR: {train_psnr:.2f} dB | Test PSNR: {val_psnr:.2f} dB")
            end_pathcond = time.time()


            print(f"Training complete. Best Test PSNR: {best_psnr:.2f} dB")
            print(f"Training complete. Best Rescaled Test PSNR: {best_psnr_pc:.2f} dB")
            print(f"Baseline training time: {end_basline - start_basline:.2f} seconds")
            print(f"Path-Cond training time (including teleport): {end_pathcond - start_pathcond:.2f} seconds")
            print(f"Total teleportation time: {time_teleport:.2f} seconds")

    torch.save({
        "PSNR_TRAIN": PSNR_TRAIN,
        "LOSS": LOSS,
        "PSNR_TEST": PSNR_TEST,
        "TIME": TIME,
        "EPOCHS": EPOCHS,
    }, _ensure_outdir("results/unet/deblurring/") / "dic.pt")

if __name__ == "__main__":
    main()