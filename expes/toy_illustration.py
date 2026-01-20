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
from matplotlib.patches import Patch
from matplotlib.colors import LogNorm
from matplotlib.ticker import LogLocator
from matplotlib.patches import FancyArrowPatch

cmap = plt.cm.get_cmap('tab10')
# cmap = plt.cm.get_cmap('viridis', 3+2)

def get_theta(model_simple):
    # u,v,w = theta[0], theta[1], theta[2] 
    theta = []
    for name, param in model_simple.named_parameters():
        theta.append(param.data)
    return torch.Tensor(theta)[[2,0,1]]

def get_phi(model_simple):
    theta = get_theta(model_simple)
    return torch.Tensor([theta[0]*theta[1], theta[0]*theta[2]])

def print_params(model):
    for name, param in model.named_parameters():
        print(name, param.shape)
        print(param.data)
    print('----')

def train_model(m, epochs, lr, X, Y):
    losses = []
    parameters = []
    phi = []
    with torch.no_grad():
        outputs = m(X)
        loss = criterion(outputs, Y)
        losses.append(loss.item())
        theta = get_theta(m)
        parameters.append(theta)
        phi.append(get_phi(m))
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
        phi.append(get_phi(m))

    return losses, parameters, phi




def loss_theta(theta, X, Y):
    v, u, w = theta
    y_pred = u * torch.relu(v * X + w)
    return F.mse_loss(y_pred, Y).item()



def plot_scaling_orbit(ax, u, v, w,
                       lambda_min=0.1, lambda_max=5.0, n=200,
                       color='black', linestyle=':', lw=2, label=None):
    """
    Plot lambda -> (u/lambda, lambda v, lambda w) in Theta space
    """
    lam = torch.linspace(lambda_min, lambda_max, n)

    U = u / lam
    V = v * lam
    W = w * lam

    ax.plot(U.numpy(), V.numpy(), W.numpy(),
            color=color, linestyle=linestyle, 
            linewidth=lw, alpha=0.5, label=label)

def plot_loss_levelsets_phi(ax, 
                            loss_phi,
                            uv_range, 
                            uw_range,
                            n=100, 
                            levels=10, 
                            cmap="Greys", 
                            zorder=None):
    uv = torch.linspace(*uv_range, n)
    uw = torch.linspace(*uw_range, n)
    UV, UW = torch.meshgrid(uv, uw, indexing='ij')

    L = torch.zeros_like(UV)
    for i in range(n):
        for j in range(n):
            phi = torch.tensor([UV[i, j], UW[i, j]])
            L[i, j] = loss_phi(phi, X, Y)

    ax.contourf(
        UV.numpy(), UW.numpy(), L.numpy(),
        levels=levels,
        cmap=cmap,
        alpha=0.6,
        linewidths=1.0, 
        zorder=zorder
    )


def loss_phi(phi, X, Y):
    u, v, w = phi_to_theta(phi)
    y_pred = u * torch.relu(v * X + w)
    return F.mse_loss(y_pred, Y).item()


def plot_loss_levelsets_phi(ax, 
                            loss_phi,
                            uv_range, 
                            uw_range,
                            n=100, 
                            levels=10, 
                            cmap="Greys",
                            alpha=0.8, 
                            zorder=None, 
                            log_scale=False):
    uv = torch.linspace(*uv_range, n)
    uw = torch.linspace(*uw_range, n)
    UV, UW = torch.meshgrid(uv, uw, indexing='ij')

    L = torch.zeros_like(UV)
    for i in range(n):
        for j in range(n):
            phi = torch.tensor([UV[i, j], UW[i, j]])
            L[i, j] = loss_phi(phi, X, Y)

    if log_scale:
        # Ensure no zero or negative values for log scale
        L_safe = L.clone()
        L_safe[L_safe <= 0] = 1e-12

        # Create logarithmically spaced levels
        log_min = L_safe.min().item()
        log_max = L_safe.max().item()
        levels = np.logspace(np.log10(log_min), np.log10(log_max), levels)
        norm = LogNorm(vmin=log_min, vmax=log_max)
    else:
        norm = None


    cf = ax.contourf(
        UV.numpy(), UW.numpy(), L.numpy(),
        levels=levels,
        cmap=cmap,
        alpha=alpha,
        linewidths=1.0, 
        zorder=zorder, 
        norm=norm
    )

    return cf

def phi_to_theta(phi):
    a, b = phi
    u = torch.ones((), device=phi.device, dtype=phi.dtype)
    return u, a, b

def ideal_gd_phi(phi0, X, Y, lr=1e-3, nepochs=100):
    phi = phi0.clone().detach().requires_grad_(True)
    all_phi = [phi.detach().clone()]

    def loss_(theta):
        u, v, w = theta
        y_pred = u * torch.relu(v * X + w)
        return F.mse_loss(y_pred, Y)

    for _ in range(nepochs):
        loss = loss_(phi_to_theta(phi))
        loss.backward()

        with torch.no_grad():
            phi -= lr * phi.grad
        phi.grad.zero_()

        all_phi.append(phi.detach().clone())

    return all_phi


#%%
model_simple = MLP(hidden_dims = [1, 1, 1]) # u rho(v x + w)


n = 100
epochs = 15000
lr = 5e-3

torch.manual_seed(36)
X = torch.randn(n, 1)
relu = nn.ReLU()
Y = relu(X)
lamd_orange = 1.0
theta_opt = torch.Tensor([1.0, 1.0, 0.0]).to(torch.float)
criterion = nn.MSELoss()
all_init = [ # v, w, u
    torch.Tensor([4.0, 3.0, 1.0]),
    torch.Tensor([2.0*lamd_orange, 4.0*lamd_orange, 1.0/lamd_orange]),
    torch.Tensor([4.0, -2.0, 1.0])
    ]


all_losses = []
all_losses_rescaled = []

all_phi = []
all_phi_rescaled = []

all_phi_ideal = []

all_theta = []
all_theta_rescaled =[]

for i, init in enumerate(all_init):

    phi0 = torch.Tensor([init[2]*init[0], init[2]*init[1]])# uv, uw
    idel_phis = ideal_gd_phi(phi0, X, Y, lr=lr, nepochs=epochs)
    all_phi_ideal.append(idel_phis)
    
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

def plot_trajectories(ax, traj, linestyle=None, color=None, s=20, label=None):

    ax.plot(traj[:, 0], traj[:, 1], color=color, alpha=0.7, linewidth=2.5, linestyle=linestyle, zorder=3, label=label)
    ax.scatter(traj[0, 0], traj[0, 1], color=color, s=s, alpha=0.9, zorder=4)          # debut
    ax.scatter(traj[-1, 0], traj[-1, 1], color=color, s=s+50, alpha=0.9, marker='+', zorder=4)  # fin
    
    mid_idx = 0  
    # if mid_idx < len(traj) - 1:
    dx = traj[mid_idx + 1, 0] - traj[mid_idx, 0]
    dy = traj[mid_idx + 1, 1] - traj[mid_idx, 1]
    ax.arrow(traj[mid_idx, 0], traj[mid_idx, 1], dx, dy,
                 head_width=0.02*torch.max(traj), 
                 head_length=0.02*torch.max(traj),
                 fc=color, ec=color, alpha=0.7, length_includes_head=True)

def plot_theta_trajectories_3d(ax, traj, linestyle=None, color=None, s=20):
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2],
            color=color, linestyle=linestyle, linewidth=2.5, alpha=0.7)

    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2],
               color=color, s=s, alpha=0.9)              # start
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2],
               color=color, s=s+50, alpha=0.9, marker='+')  # end


fs=15 
s=20
# legend_lines = [
#     Line2D([0], [0], linestyle='--', color='black',  label='Vanilla'),
#     Line2D([0], [0], linestyle='-', color='black', label='Rescaled'),
# ]
legend_lines = [
    Line2D([0], [0], linestyle='--', color='black',  label='Vanilla GD'),
    Line2D([0], [0], linestyle='-', color='black', label='PathCond'),
    Line2D([0], [0], linestyle=':', color='black', label='GD for $\\ell(\\Phi)$'),
]

fig = plt.figure(figsize=(20, 3.5))

gs = gridspec.GridSpec(
    1, 3,
    width_ratios=[1, 1, 1],
    wspace=0.3   # THIS works for 3D
)

ax0 = fig.add_subplot(gs[2])
ax1 = fig.add_subplot(gs[1])
ax2 = fig.add_subplot(gs[0], projection='3d')

for i, init in enumerate(all_init):
    loss = all_losses[i]
    loss_rescaled = all_losses_rescaled[i]

    ax0.plot(loss, lw=2, linestyle='--', color=cmap(i))
    ax0.plot(loss_rescaled, lw=2, linestyle='-', color=cmap(i))
    ax0.set_xlabel("iter", fontsize=fs)
    # ax0.legend(handles=legend_lines, fontsize=fs-1)
    ax0.set_yscale('log')
    ax0.set_title('Loss $L(\\theta)$ during GD',fontsize=fs+2)
    ax0.tick_params(axis='both', which='major', labelsize=fs-1)
    ax0.tick_params(axis='both', which='minor', labelsize=fs-1)
    ax0.set_xscale('log')

    traj_rescaled = torch.stack(all_phi_rescaled[i])
    traj = torch.stack(all_phi[i])
    traj_ideal = torch.stack(all_phi_ideal[i])
    plot_trajectories(ax1, traj, linestyle='--', color=cmap(i), s=s)
    plot_trajectories(ax1, traj_rescaled, linestyle='-', color=cmap(i), s=s)
    plot_trajectories(ax1, traj_ideal, linestyle=':', color=cmap(i), s=s)

    if i == 0:
        ax1.scatter(theta_opt[0]*theta_opt[1], 
                    theta_opt[0]*theta_opt[2], 
                    label='$\\Phi(\\theta_{\\text{opt}})$', 
                    color='black', 
                    s=50, 
                    marker='+',
                    zorder=10)   
        
    if i == 0:
        x0, y0 = traj[0, 0], traj[0, 1]
        ax1.text(
            x0, y0,
            "start",
            fontsize=fs-3,
            ha='center',
            va='bottom',
            color=cmap(i),
            zorder=6
        )
        
    # ax1.legend(handles=legend_lines, fontsize=fs-1, bbox_to_anchor=(0.1, -0.0))
    ax1.set_xlabel('$uv$', fontsize=fs)
    ax1.set_ylabel('$uw$', fontsize=fs)
    ax1.set_title('Trajectory in $\\Phi$ space',fontsize=fs+2)



    # θ trajectories (3D)
    traj_theta = torch.stack(all_theta[i])
    traj_theta_rescaled = torch.stack(all_theta_rescaled[i])
    plot_theta_trajectories_3d(ax2, traj_theta, linestyle='--', color=cmap(i))
    plot_theta_trajectories_3d(ax2, traj_theta_rescaled, linestyle='-', color=cmap(i))

    if i == 0:
        ax2.scatter(theta_opt[0], theta_opt[1], theta_opt[2], 
                    label='$\\theta_{\\text{opt}}$', 
                    color='black', s=50, marker='+')
    else:
        ax2.scatter(theta_opt[0], theta_opt[1], theta_opt[2], 
                    color='black', s=50, marker='+')        


    if i == 0:
        x0, y0, z0 = traj_theta[0, 0], traj_theta[0, 1], traj_theta[0, 2]
        ax2.text(
            x0, y0, z0,
            "start",
            fontsize=fs-3,
            ha='center',
            va='bottom',
            color=cmap(i)
        )
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
        pos.x0,  # ← shift left (adjust value)
        pos.y0,
        pos.width,
        pos.height
    ])


ax0.grid(True, which='major', alpha=0.9)

cf = plot_loss_levelsets_phi(
    ax1,
    loss_phi,
    uv_range=(0, 4.5),
    uw_range=(-3, 4.5),
    # uv_range=(-50, 50),
    # uw_range=(-50, 50),
    levels=15,
    cmap="Greys",
    log_scale=True,
    alpha=0.35,
    zorder=-1,
    n=250
)


cbar = fig.colorbar(cf, ax=ax1, pad=0.02)
cbar.set_label('Loss $\\ell(\\Phi)$', fontsize=fs-2)
cbar.ax.tick_params(labelsize=fs-2)

cbar.locator = LogLocator(base=10.0)  # ticks at 10^n
cbar.update_ticks()


u0, v0, w0 = theta_opt.tolist()   # or any fixed (u,v,w)

plot_scaling_orbit(
    ax2,
    u=u0, v=v0, w=w0,
    lambda_min=0.45,
    lambda_max=6,
    color='grey',
    linestyle='-',
    label='$\\theta \\sim \\theta_{\\text{opt}}$'
)
handles_3d, labels_3d = ax2.get_legend_handles_labels()

ax2.legend(
    handles=handles_3d + legend_lines,
    fontsize=fs-2,
    # loc='upper right',
    bbox_to_anchor=(0.9, 0.9)
)
ax2.view_init(elev=15, azim=60)

plt.tight_layout()
plt.savefig('../../../icml2026/fig/toy_illustration.pdf', 
            bbox_inches="tight", pad_inches=0.1)
plt.show()


# %%
