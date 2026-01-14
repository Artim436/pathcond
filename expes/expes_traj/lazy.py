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
from pathcond.data import SyntheticDeblurDataset, get_deblur_loaders
from tqdm import tqdm
import numpy as np
from pathcond.models import toy_MLP
import matplotlib.pyplot as plt
from expes.lazy_plot import main as plot_lazy_experiment


seed = 0

teacher = toy_MLP(d_input=2, d_hidden1=3, teacher_init=False)
teacher.init_spherical(seed=seed)

student = toy_MLP(d_input=2, d_hidden1=50, teacher_init=False)
student.init_spherical_symmetrized(alpha=1, tau=0.1, seed=seed)

student_rescaled = toy_MLP(d_input=2, d_hidden1=50, teacher_init=False)
student_rescaled.init_spherical_symmetrized(alpha=1, tau=0.1, seed=seed)

BZ, Z = optimize_rescaling_polynomial(student_rescaled, n_iter=100, tol=1e-6, module_type='MLP')

student_rescaled = reweight_model(student_rescaled, BZ, Z)

input  = torch.randn(5, 2)
output = student(input)
output_rescaled = student_rescaled(input)
assert torch.allclose(output, output_rescaled, atol=1e-6)


print(torch.max(torch.abs(BZ)))


# Simple synthetic dataset (teacher-student)
def make_dataset(n=20):
    x = torch.randn(n, 2)
    with torch.no_grad():
        y = teacher(x)
    return x, y

# Dataset
x_train, y_train = make_dataset()

# Training setup
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(student.parameters(), lr=1e-4)
optimizer_rescaled = torch.optim.SGD(student_rescaled.parameters(), lr=1e-4)

# Dataset

# Record trajectories - PRÉ-ALLOCATION
T = 1000000
n_neurons = student.model[0].out_features
n_neurons_rescaled = student_rescaled.model[0].out_features
d_input = 2

# Pré-allocation des arrays numpy
trajectories = np.zeros((n_neurons, T, d_input), dtype=np.float32)
trajectories_rescaled = np.zeros((n_neurons_rescaled, T, d_input), dtype=np.float32)
LOSS = np.zeros(T, dtype=np.float32)
LOSS_rescaled = np.zeros(T, dtype=np.float32)

# Signs
with torch.no_grad():
    signs = torch.sign(student.model[2].weight.view(-1)).clone()
    signs_rescaled = torch.sign(student_rescaled.model[2].weight.view(-1)).clone()


# Training rescaled model
for t in range(T):
    with torch.no_grad():
        W = student_rescaled.model[0].weight
        a = student_rescaled.model[2].weight.view(-1)
        for i in range(W.shape[0]):
            trajectories_rescaled[i, t] = (a[i].abs() * W[i]).cpu().numpy()
    # if t % 100000 == 0:
    #     BZ, Z = optimize_rescaling_polynomial(student_rescaled, n_iter=100, tol=1e-6, module_type='MLP')
    #     student_rescaled = reweight_model(student_rescaled, BZ, Z)
    #     optimizer_rescaled = torch.optim.SGD(student_rescaled.parameters(), lr=1e-5) 
    optimizer_rescaled.zero_grad()
    
    y_pred_rescaled = student_rescaled(x_train)
    loss_rescaled = criterion(y_pred_rescaled, y_train)
    LOSS_rescaled[t] = loss_rescaled.item()

    loss_rescaled.backward()
    optimizer_rescaled.step()



# Training standard model
for t in range(T):
    with torch.no_grad():
        W = student.model[0].weight
        a = student.model[2].weight.view(-1)
        for i in range(W.shape[0]):
            trajectories[i, t] = (a[i].abs() * W[i]).cpu().numpy()
    optimizer.zero_grad()

    y_pred = student(x_train)
    loss = criterion(y_pred, y_train)
    LOSS[t] = loss.item()

    loss.backward()
    optimizer.step()






torch.save({
    'trajectories': trajectories,
    'trajectories_rescaled': trajectories_rescaled,
    'signs': signs,
    'signs_rescaled': signs_rescaled, 
    'LOSS': LOSS,
    'LOSS_rescaled': LOSS_rescaled,
    'teacher': teacher,
}, 'results/lazy/toy_MLP_lazy_training_trajectories.pt')



# Plotting
plot_lazy_experiment()