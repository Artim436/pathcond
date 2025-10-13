# %%
import torch
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model, compute_diag_G, optimize_rescaling_gd, compute_matrix_B
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import math
import copy
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import time

# %%
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# %%
plt.scatter(X[:, 0], X[:, 1])


# %%
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Linear(4, 2, bias=False),
            # nn.ReLU(),
            # nn.Linear(10, 15),
            # nn.ReLU(),
            # nn.Linear(15, 2)
        )

    def forward(self, x, device='cpu'):
        x = x.to(device)
        return self.model(x)


# %%
model = SimpleNN()

B = compute_matrix_B(model).to(torch.float)

# %%
B.T @ torch.ones(B.shape[0])
# %%
nb_iter = 10
lr = 0.1
epochs = 1500
rescale_every = 200
torch.manual_seed(50)

model_simple = SimpleNN()
for m in model_simple.modules():
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)

BZ_opt, Z_opt, alpha, OBJ = optimize_neuron_rescaling_polynomial(
    model=model_simple, n_iter=nb_iter, verbose=True, tol=1e-6)

BZ_opt = BZ_opt.to(torch.float)

print("BZ shape: ", BZ_opt.shape)

print("alpha: ", alpha)


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
            BZ_opt = optimize_neuron_rescaling_polynomial(
                model=model,
                n_iter=nb_iter,
                verbose=False,
                tol=1e-6).to(torch.float)
            st = time.time()
            param_vec = parameters_to_vector(model.parameters())
            rescaling = torch.exp(-0.5 * BZ_opt)
            diag_G = compute_diag_G(model)
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
        if model_name in [
            'vanilla',
            'init. rescaled',
            'teleport'
        ]:
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

    plt.suptitle('Two-moons experiment', fontsize=fs+3)

    plt.tight_layout()

    plt.show()


plot_loss_dict(loss_histories, fs=16, figsize=(12, 7))


# %%
diag_G*torch.exp(BZ_opt)
# %%
