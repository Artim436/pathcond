# %%
import time
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import copy
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import torch.optim as optim
import torch.nn as nn
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model, compute_diag_G, optimize_rescaling_gd, compute_matrix_B
import torch
import numpy as np
# # %%
# a = 0.04231291636824608
# b = 1074.84521484375
# c = -0.06107430160045624
# np.roots([a, b, c])
# # %%
# a_t = torch.Tensor([a]).to(torch.float64)
# b_t = torch.Tensor([b]).to(torch.float64)
# c_t = torch.Tensor([c]).to(torch.float64)
# disc = b_t**2 - 4.0 * a_t * c_t
# sqrt_disc = torch.sqrt(disc)
# x1 = (-b_t + sqrt_disc) / (2.0 * a_t)
# x2 = (-b_t - sqrt_disc) / (2.0 * a_t)
# print(x1)
# %%


def fisher_diag(model, X):
    """
    Compute diag(J^T J) where J = d f(X,θ)/dθ.
    Returns a flat vector of size (D,)
    """
    params = [p for p in model.parameters() if p.requires_grad]
    D = sum(p.numel() for p in params)

    # Forward
    y = model(X).view(-1)

    # Backprop a vector of ones through outputs
    # This gives per-sample gradient accumulation
    grads = torch.autograd.grad(
        y, params, grad_outputs=torch.ones_like(y),
        create_graph=True, retain_graph=True
    )

    # Now grads is the gradient of sum(f) wrt params.
    # We need squared grads per output, so we do per-output trick:
    diag = torch.zeros(D, device=X.device)
    for i in range(y.numel()):
        gi = torch.autograd.grad(y[i], params, retain_graph=True)
        flat_gi = torch.cat([g.reshape(-1) for g in gi])
        diag += flat_gi.pow(2)

    return diag


def rescaling_nn(model,
                 X=None,
                 method='path',
                 n_iter=10,
                 tol=1e-6,
                 reg=None):

    # --- Setup: device/dtype and network structure ---
    device = next(model.parameters()).device
    dtype = torch.float64

    # Collect linear layers; exclude the final (output) layer from hidden count
    linear_indices = [i for i, layer in enumerate(model.model) if isinstance(layer, nn.Linear)]
    n_params = sum(p.numel() for p in model.parameters())
    n_hidden_neurons = sum(model.model[i].out_features for i in linear_indices[:-1])

    # Parameters to optimize
    zold = torch.zeros(n_hidden_neurons, dtype=dtype, device=device)

    # Problem-specific matrices/vectors
    B = compute_matrix_B(model).to(device=device, dtype=dtype)     # shape: [m, n_hidden_neurons]
    if method == 'fisher':
        if X is None:
            raise ValueError('X must be defined for fisher rescaling')
        diag_G = fisher_diag(model, X)
    elif method == 'path':
        diag_G = compute_diag_G(model).to(device=device, dtype=dtype)
    if reg is not None:
        diag_G += reg
    mask_in = (B == -1)
    mask_out = (B == 1)
    card_in = mask_in.sum(dim=0)   # [H]
    card_out = mask_out.sum(dim=0)  # [H]

    # Maintain BZ incrementally: BZ = B @ Z
    BZ = torch.zeros(n_params, dtype=dtype, device=device)
    # print(f"Initial obj: {OBJ[0]:.6f}")
    for k in range(n_iter):
        znew, BZ = one_pass_z(zold, BZ, diag_G, B, mask_in, mask_out, card_in, card_out)
        delta_total = torch.linalg.norm(znew - zold)
        if delta_total < tol:
            print(f"Converged after {k+1} iterations (delta_total={delta_total:.6e} < tol={tol})")
            break

        zold = znew

    # alpha = n_params/torch.sum(torch.exp(BZ) * diag_G).item()
    return BZ


def one_pass_z(z, BZ, g, B, mask_in, mask_out, card_in, card_out):
    # Do one pass on every z_h
    # Maintain BZ incrementally: BZ = B @ Z
    n_params_tensor = g.shape[0]
    H = z.shape[0]
    # BZ = torch.zeros(n_params_tensor)

    for h in range(H):
        b_h = B[:, h]

        A_h = int(card_in[h].item()) - int(card_out[h].item())

        # Leave-one-out energy vector
        Y_h = BZ - b_h * z[h]
        y_bar = Y_h.max()
        E = torch.exp(Y_h - y_bar) * g

        # sums using masks
        B_h = (E * mask_out[:, h]).sum()
        C_h = (E * mask_in[:, h]).sum()
        # D_h = rest of elements
        D_h = E.sum() - B_h - C_h

        # Polynomial coefficients
        a = B_h * (A_h + n_params_tensor)
        b = D_h * A_h
        c = C_h * (A_h - n_params_tensor)

        disc = b**2 - 4.0 * a * c
        sqrt_disc = torch.sqrt(disc)
        x1 = (-b + sqrt_disc) / (2.0 * a)
        x2 = (-b - sqrt_disc) / (2.0 * a)

        z_new = torch.log(torch.maximum(x1, x2))

        # Update Z[h] and incrementally refresh BZ
        delta = z_new - z[h]
        BZ = BZ + b_h * delta
        z[h] = z_new
        if BZ.isnan().any():
            # print('iter = {}'.format(h))
            # print('disc = {}'.format(disc))
            print('a = {}, b = {}, c = {}'.format(a, b, c))
            # print('A_h = {}, B_h = {}, C_h = {}, D_h = {}'.format(A_h, B_h, C_h, D_h))
            # print('z_new = {}'.format(z_new))
            print('x1, x2 = {}'.format((x1, x2)))
            # print('E = {}'.format(E))
            # print('disc - b = {}'.format(disc-b))
            print('sqrt_disc = {}'.format(sqrt_disc))
            print('ac = {}'.format(b**2 - 4.0*a*c))
            # print('E * mask_out[:, h] = {}'.format(E * mask_out[:, h]))
            raise ValueError('Nan in BZ')

    return z, BZ


X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float64)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            # nn.Linear(15, 10),
            # nn.ReLU(),
            # nn.Linear(10, 5),
            # nn.ReLU(),
            # nn.Linear(5, 2),
            # nn.ReLU(),
            # nn.Linear(10, 15),
            # nn.ReLU(),
            nn.Linear(10, 2)
        )

    def forward(self, x, device='cpu'):
        x = x.to(device)
        return self.model(x)


nb_iter = 10
lr = 0.005
epochs = 2000
rescale_every = 50
reg = 1e-4
method = 'fisher'
torch.manual_seed(50)

model_simple = SimpleNN().double()
BZ_opt = rescaling_nn(model=model_simple, X=X_train, method=method, n_iter=nb_iter, tol=1e-6, reg=reg)

model_rescaled = reweight_model(model_simple, BZ_opt)
model_teleport_first = copy.deepcopy(model_simple)
model_teleport_second = copy.deepcopy(model_simple)

criterion = nn.CrossEntropyLoss()
all_model = [
    model_simple,
    model_rescaled,
    model_teleport_second
]
all_names = [
    'vanilla',
    'init. rescaled',
    'teleport'
]
loss_histories = {}
all_rescaling = []
all_diag_G = []
for name in all_names:
    loss_histories[(name, 'loss')] = []
    loss_histories[(name, 'acc_test')] = []
    loss_histories[(name, 'acc_train')] = []
total_time_forward_rescaled = 0
total_time_forward_inplace = 0
for model, name in zip(all_model, all_names):
    print('Do {}'.format(name))
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        loss_histories[(name, 'loss')].append(loss.item())

        _, predicted = torch.max(outputs, 1)
        acc = (predicted == y_train).float().mean()
        loss_histories[(name, 'acc_train')].append(acc)
        # print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Train Acc: {acc:.4f}")

        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            acc_test = (predicted == y_test).float().mean()
            loss_histories[(name, 'acc_test')].append(acc_test)
        # print("Test Accuracy for model {0}: {1:.4f}".format(name, acc_test))

        if name == "teleport" and (epoch+1) % rescale_every == 0:
            # print(model)
            BZ_opt = rescaling_nn(
                model=model,
                X=X_train,
                method=method,
                n_iter=nb_iter,
                tol=1e-6,
                reg=reg)
            if BZ_opt.isnan().any():
                raise ValueError('Nan in BZ')
            st = time.time()
            param_vec = parameters_to_vector(model.parameters())
            rescaling = torch.exp(-0.5 * BZ_opt)
            diag_G = fisher_diag(model, X_train)
            all_diag_G.append(diag_G)
            all_rescaling.append(rescaling)
            reweighted_vec = param_vec * rescaling
            vector_to_parameters(reweighted_vec, model.parameters())
            ed = time.time()
            total_time_forward_inplace += ed - st
# %%
all_rescaling = torch.stack(all_rescaling, dim=0)
all_diag_G = torch.stack(all_diag_G, dim=0)
# # %%
# print('Time forward_rescaled = {}'.format(total_time_forward_rescaled))
# print('Time forward_inplace = {}'.format(total_time_forward_inplace))

# %%

cmap = plt.cm.get_cmap('tab10')


def plot_loss_dict(loss_histories, fs=15, figsize=(5, 5)):
    n = len(loss_histories)
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    for model_name in all_names:
        if model_name in ['vanilla', 'init. rescaled', 'teleport']:
            loss_history = loss_histories[(model_name, 'loss')]
            test_acc = loss_histories[(model_name, 'acc_test')]
            train_acc = loss_histories[(model_name, 'acc_train')]

            axes[0, 0].plot(loss_history, lw=2)
            axes[0, 0].set_xlabel("Epochs", fontsize=fs)
            axes[0, 0].set_ylabel("Training loss", fontsize=fs)
            axes[0, 0].grid(alpha=0.5)

            axes[0, 1].plot(train_acc, lw=2)
            axes[0, 1].set_xlabel("Epochs", fontsize=fs)
            axes[0, 1].set_ylabel("Train accuracy", fontsize=fs)
            axes[0, 1].grid(alpha=0.5)

            if model_name == 'teleport':
                write = 'teleport (every {})'.format(rescale_every)
            else:
                write = model_name
            axes[0, 2].plot(test_acc, label=write, lw=2)
            axes[0, 2].set_xlabel("Epochs", fontsize=fs)
            axes[0, 2].set_ylabel("Test accuracy", fontsize=fs)
            axes[0, 2].legend(fontsize=fs)
            axes[0, 2].grid(alpha=0.5)

    axes[1, 0].plot(all_rescaling.mean(1), lw=3, label='mean rescaling', color=cmap(0))
    # axes[3].fill_between(all_rescaling.mean(0), lw=3)
    axes[1, 0].fill_between(range(len(all_rescaling.mean(1))),
                            all_rescaling.mean(1)-torch.std(all_rescaling, axis=1),
                            all_rescaling.mean(1)+torch.std(all_rescaling, axis=1),
                            alpha=0.3, color=cmap(0))

    axes[1, 0].set_xlabel("teleport step", fontsize=fs)
    axes[1, 0].grid(alpha=0.5)
    axes[1, 0].set_title('Scaling (+/- std)', fontsize=fs+2)
    axes[1, 0].legend(fontsize=fs)

    to_plot = all_diag_G.mean(1)
    axes[1, 1].plot(all_diag_G.mean(1), lw=3, label='mean $G_{ii}$', color=cmap(1))
    axes[1, 1].plot(all_diag_G.min(1)[0], lw=3, label='min $G_{ii}$',
                    color=cmap(1),
                    linestyle='--', alpha=0.4)
    axes[1, 1].plot(all_diag_G.max(1)[0],
                    lw=3,
                    label='max $G_{ii}$',
                    color=cmap(1),
                    linestyle='--',
                    alpha=0.4)

    axes[1, 1].fill_between(range(len(to_plot)),
                            all_diag_G.mean(1)-torch.std(all_diag_G, axis=1),
                            all_diag_G.mean(1)+torch.std(all_diag_G, axis=1),
                            alpha=0.3,
                            color=cmap(1))

    # axes[3].fill_between(all_rescaling.mean(0), lw=3)
    axes[1, 1].set_xlabel("teleport step", fontsize=fs)
    axes[1, 1].grid(alpha=0.5)
    axes[1, 1].set_title('$\\operatorname{diag}(G)$', fontsize=fs+2)
    axes[1, 1].legend(fontsize=fs)

    to_plot = all_diag_G.max(1)[0] / all_diag_G.min(1)[0]
    axes[1, 2].plot(to_plot, lw=3, label='$\kappa(\\operatorname{diag}(G))$', color=cmap(2))
    # axes[3].fill_between(all_rescaling.mean(0), lw=3)
    axes[1, 2].set_xlabel("teleport step", fontsize=fs)
    axes[1, 2].grid(alpha=0.5)
    axes[1, 2].set_title('Condition number', fontsize=fs+2)
    axes[1, 2].legend(fontsize=fs)

    plt.suptitle('Two-moons experiment with Fisher rescaling', fontsize=fs+3)

    plt.tight_layout()

    plt.show()


plot_loss_dict(loss_histories, fs=16, figsize=(12, 7))
# %%
