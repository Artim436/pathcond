#%%
import torch
from pathcond.rescaling_polyn import optimize_rescaling_polynomial, reweight_model
from pathcond.models import apply_init
import torch.nn as nn
import torch.nn.functional as F
from pathcond.models import MLP
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import copy
import numpy as np
from matplotlib.lines import Line2D

cmap = plt.cm.get_cmap('tab10')


def get_theta(model_simple):
    # u,v,w = theta[0], theta[1], theta[2] 
    theta = []
    for name, param in model_simple.named_parameters():
        theta.append(param.data)
    return torch.Tensor(theta)[[2,0,1]]

def print_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)
        print(param.data)
    print('----')

def train_model(m, epochs, lr, X, Y):
    losses = []
    parameters = []
    phi = []
    optimizer = optim.SGD(m.parameters(), lr=lr)
    for _ in range(epochs):

        m.train()
        optimizer.zero_grad()
        outputs = m(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        theta = get_theta(m)
        parameters.append(theta)
        phi.append(torch.Tensor([theta[0]*theta[1], theta[0]*theta[2]]))

    return losses, parameters, phi

#%%
model_simple = MLP(hidden_dims = [1, 1, 1])


n = 50
epochs = 15000
lr = 1e-5

torch.manual_seed(36)
X = torch.randn(n, 1)
relu = nn.ReLU()
Y = relu(X)
theta_opt = torch.Tensor([1.0, 1.0, 0.0]).to(torch.float)
criterion = nn.MSELoss()
all_init = [
    torch.Tensor([2.0, 1.0, 2.0]), # v, w, u
    torch.Tensor([5.0, 1.0, 3.0]),
    torch.Tensor([3.0, 1.0, 4.0])
    ]


results = {}
all_losses = []
all_losses_rescaled = []

all_phi = []
all_phi_rescaled = []

for i, init in enumerate(all_init):
    
    model = copy.deepcopy(model_simple)
    vector_to_parameters(init, model.parameters())
    model_before_train = copy.deepcopy(model)
    losses, parameters, phi = train_model(model, epochs, lr, X, Y)
    all_losses.append(losses)
    all_phi.append(phi)


    BZ, Z = optimize_rescaling_polynomial(model_before_train)
    rescaling = torch.exp(-0.5 * BZ)
    model_pathcond = reweight_model(model_before_train, BZ, Z)
    losses_rescaled, parameters_rescaled, phi_rescaled = train_model(model_pathcond, epochs, lr, X, Y)
    all_losses_rescaled.append(losses_rescaled)
    all_phi_rescaled.append(phi_rescaled)


#%%
def plot_trajectories(ax, traj, linestyle=None, color=None, s=20):

    ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, linewidth=1.5, linestyle=linestyle)
    
    ax.scatter(traj[0, 0], traj[0, 1], color=color, s=s, alpha=0.9)          # debut
    ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=s+50, alpha=0.9, marker='+')  # fin
    
    # mid_idx = 0  
    # # if mid_idx < len(traj) - 1:
    # dx = traj[mid_idx + 1, 0] - traj[mid_idx, 0]
    # dy = traj[mid_idx + 1, 1] - traj[mid_idx, 1]
    # ax.arrow(traj[mid_idx, 0], traj[mid_idx, 1], dx, dy,
    #              head_width=0.02*torch.max(traj), head_length=0.02*torch.max(traj),
    #              fc=color, ec=color, alpha=0.7, length_includes_head=True)
        
fs=15 
figsize=(8, 4) 
s=20
legend_lines = [
    Line2D([0], [0], linestyle='--', color='black',  label='Vanilla'),
    Line2D([0], [0], linestyle='-', color='black', label='Rescaled'),
]

fig, axes = plt.subplots(1, 2, figsize=figsize)
for i, init in enumerate(all_init):
    loss = all_losses[i]
    loss_rescaled = all_losses_rescaled[i]

    axes[0].plot(loss, lw=2, linestyle='--', color=cmap(i))
    axes[0].plot(loss_rescaled, lw=2, linestyle='-', color=cmap(i))
    axes[0].set_xlabel("Epochs", fontsize=fs)
    axes[0].grid(alpha=0.5)
    axes[0].legend(handles=legend_lines, fontsize=fs)
    axes[0].set_yscale('log')
    axes[0].set_title('Loss during SGD',fontsize=fs+2)
    axes[0].tick_params(axis='both', which='major', labelsize=fs-1)
    axes[0].tick_params(axis='both', which='minor', labelsize=fs-1)
    axes[0].set_xscale('log')

    traj_rescaled = torch.stack(all_phi_rescaled[i])
    traj = torch.stack(all_phi[i])
    plot_trajectories(axes[1], traj, linestyle='--', color=cmap(i), s=s)
    plot_trajectories(axes[1], traj_rescaled, linestyle='-', color=cmap(i), s=s)

    # axes[1].legend(fontsize=fs)
    axes[1].set_xlabel('$uv$', fontsize=fs)
    axes[1].set_ylabel('$uw$', fontsize=fs)
    axes[1].set_title('Trajectory in $\\Phi$ space',fontsize=fs+2)

plt.tight_layout()

plt.show()


# %%
