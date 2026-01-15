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
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

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
model_simple = MLP(hidden_dims = [1, 1, 1]) # u rho(v x + w)


n = 50
epochs = 15000
lr = 7e-3

torch.manual_seed(36)
X = torch.randn(n, 1)
relu = nn.ReLU()
Y = relu(X)
theta_opt = torch.Tensor([1.0, 1.0, 0.0]).to(torch.float)
criterion = nn.MSELoss()
all_init = [ # v, w, u
    torch.Tensor([1.5, 0.5, 1.0]), #lr 7e-3, epochs 15000
    torch.Tensor([3.0, 1.0, 4.0])
    ]


all_losses = []
all_losses_rescaled = []

all_phi = []
all_phi_rescaled = []

all_theta = []
all_theta_rescaled =[]

for i, init in enumerate(all_init):
    
    model = copy.deepcopy(model_simple)
    vector_to_parameters(init, model.parameters())
    model_before_train = copy.deepcopy(model)
    losses, parameters, phi = train_model(model, epochs, lr, X, Y)
    all_losses.append(losses)
    all_phi.append(phi)
    all_theta.append(parameters)


    BZ, Z = optimize_rescaling_polynomial(model_before_train)
    rescaling = torch.exp(-0.5 * BZ)
    model_pathcond = reweight_model(model_before_train, BZ, Z)
    losses_rescaled, parameters_rescaled, phi_rescaled = train_model(model_pathcond, epochs, lr, X, Y)
    all_losses_rescaled.append(losses_rescaled)
    all_phi_rescaled.append(phi_rescaled)
    all_theta_rescaled.append(parameters_rescaled)


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

def plot_theta_trajectories_3d(ax, traj, linestyle=None, color=None, s=20):
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            color=color, linestyle=linestyle, linewidth=1.5, alpha=0.7)

    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
               color=color, s=s, alpha=0.9)              # start
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
               color=color, s=s+50, alpha=0.9, marker='+')  # end


fs=15 
figsize=(8, 4) 
s=20
legend_lines = [
    Line2D([0], [0], linestyle='--', color='black',  label='Vanilla'),
    Line2D([0], [0], linestyle='-', color='black', label='Rescaled'),
]

fig = plt.figure(figsize=(12, 3))

gs = gridspec.GridSpec(
    1, 3,
    width_ratios=[1, 1, 1],
    wspace=0.25   # THIS works for 3D
)

ax0 = fig.add_subplot(gs[0])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[2], projection='3d')

for i, init in enumerate(all_init):
    loss = all_losses[i]
    loss_rescaled = all_losses_rescaled[i]

    ax0.plot(loss, lw=2, linestyle='--', color=cmap(i))
    ax0.plot(loss_rescaled, lw=2, linestyle='-', color=cmap(i))
    ax0.set_xlabel("Epochs", fontsize=fs)
    ax0.grid(alpha=0.5)
    ax0.legend(handles=legend_lines, fontsize=fs)
    ax0.set_yscale('log')
    ax0.set_title('Loss during SGD',fontsize=fs+2)
    ax0.tick_params(axis='both', which='major', labelsize=fs-1)
    ax0.tick_params(axis='both', which='minor', labelsize=fs-1)
    ax0.set_xscale('log')

    traj_rescaled = torch.stack(all_phi_rescaled[i])
    traj = torch.stack(all_phi[i])
    plot_trajectories(ax1, traj, linestyle='--', color=cmap(i), s=s)
    plot_trajectories(ax1, traj_rescaled, linestyle='-', color=cmap(i), s=s)

    # axes[1].legend(fontsize=fs)
    ax1.set_xlabel('$uv$', fontsize=fs)
    ax1.set_ylabel('$uw$', fontsize=fs)
    ax1.set_title('Trajectory in $\\Phi$ space',fontsize=fs+2)

    # θ trajectories (3D)
    traj_theta = torch.stack(all_theta[i])
    traj_theta_rescaled = torch.stack(all_theta_rescaled[i])
    plot_theta_trajectories_3d(ax2, traj_theta, linestyle='--', color=cmap(i))
    plot_theta_trajectories_3d(ax2, traj_theta_rescaled, linestyle='-', color=cmap(i))

    if i == 0:
        ax2.scatter(theta_opt[0], theta_opt[1], theta_opt[2], label='$\\theta^\\star$', 
                    color='black', s=50, marker='+')
    else:
        ax2.scatter(theta_opt[0], theta_opt[1], theta_opt[2], 
                    color='black', s=50, marker='+')        


    #ax2.zaxis.set_tick_params(pad=-2)
    ax2.set_zticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])

    ax2.set_xlabel('$w$', fontsize=fs, labelpad=-11)
    ax2.set_ylabel('$v$', fontsize=fs, labelpad=-11)
    ax2.set_zlabel('$u$', fontsize=fs, labelpad=-11)
    ax2.set_title('Trajectory in $\\Theta$ space', fontsize=fs+2)

    # get current position of 3D axis

    # move it left (reduce x0)
    pos = ax2.get_position()

    ax2.set_position([
        pos.x0 -0.01,  # ← shift left (adjust value)
        pos.y0,
        pos.width,
        pos.height
    ])
    ax2.legend(fontsize=fs)

plt.tight_layout()
plt.show()


# %%
