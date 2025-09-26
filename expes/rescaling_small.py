# %%
import torch
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model, forward_with_rescaled
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
# 1. Génération du dataset Two Moons
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 2. Définition du réseau


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 2),
            # nn.ReLU(),
            # nn.Linear(16, 16),
            # nn.ReLU(),
            # nn.Linear(16, 2)  # sortie binaire
        )

    def forward(self, x, device='cpu'):
        x = x.to(device)
        return self.model(x)


# %%
nb_iter = 10
verbose = True
lr = 1.0
epochs = 1000
rescale_every = 10
# torch.manual_seed(3)

model_simple = SimpleNN()

BZ_opt, Z_opt, alpha, OBJ_hist = optimize_neuron_rescaling_polynomial(
    model=model_simple, n_iter=nb_iter, verbose=verbose, tol=1e-6)

model_rescaled = reweight_model(model_simple, BZ_opt)
model_teleport_first = copy.deepcopy(model_simple)
model_teleport_second = copy.deepcopy(model_simple)

criterion = nn.CrossEntropyLoss()
all_model = [
    model_simple,
    model_rescaled,
    # model_teleport_first,
    model_teleport_second
]
all_names = [
    'vanilla',
    'init. rescaled',
    # 'teleport_forward',
    'teleport'
]
loss_histories = {}
all_rescaling = []

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
        if name == "teleport_forward" and (epoch+1) % rescale_every == 0:

            BZ_opt = optimize_neuron_rescaling_polynomial(
                model=model,
                n_iter=nb_iter,
                verbose=False,
                tol=1e-6)
            st = time.time()
            outputs = forward_with_rescaled(model, X_train, torch.exp(-0.5 * BZ_opt))
            ed = time.time()
            total_time_forward_rescaled += ed - st

        else:
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
                tol=1e-6)
            st = time.time()
            param_vec = parameters_to_vector(model.parameters())
            rescaling = torch.exp(-0.5 * BZ_opt)
            all_rescaling.append(rescaling)
            reweighted_vec = param_vec * rescaling
            vector_to_parameters(reweighted_vec, model.parameters())
            ed = time.time()
            total_time_forward_inplace += ed - st
# %%
all_rescaling = torch.stack(all_rescaling, dim=0)
# %%
print('Time forward_rescaled = {}'.format(total_time_forward_rescaled))
print('Time forward_inplace = {}'.format(total_time_forward_inplace))

# %%


def plot_loss_dict(loss_histories, fs=15, figsize=(5, 5)):
    n = len(loss_histories)
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    for model_name in all_names:
        if model_name in ['vanilla', 'init. rescaled', 'teleport']:
            loss_history = loss_histories[(model_name, 'loss')]
            test_acc = loss_histories[(model_name, 'acc_test')]
            train_acc = loss_histories[(model_name, 'acc_train')]

            axes[0].plot(loss_history, label=f"{model_name}", lw=2)
            axes[0].set_xlabel("Epochs", fontsize=fs)
            axes[0].set_ylabel("Training loss", fontsize=fs)
            # axes[0].legend(fontsize=fs)
            axes[0].grid(alpha=0.5)

            axes[1].plot(train_acc, label=f"{model_name}", lw=2)
            axes[1].set_xlabel("Epochs", fontsize=fs)
            axes[1].set_ylabel("Train accuracy", fontsize=fs)
            # axes[1].legend(fontsize=fs)
            axes[1].grid(alpha=0.5)

            axes[2].plot(test_acc, label=f"{model_name}", lw=2)
            axes[2].set_xlabel("Epochs", fontsize=fs)
            axes[2].set_ylabel("Test accuracy", fontsize=fs)
            axes[2].legend(fontsize=fs)
            axes[2].grid(alpha=0.5)

    axes[3].plot(all_rescaling.mean(0), lw=3)
    # axes[3].fill_between(all_rescaling.mean(0), lw=3)
    axes[3].fill_between(range(len(all_rescaling.mean(0))),
                         all_rescaling.mean(0)-torch.std(all_rescaling, axis=0),
                         all_rescaling.mean(0)+torch.std(all_rescaling, axis=0),
                         alpha=0.5)
    axes[3].set_xlabel("steps", fontsize=fs)
    axes[3].grid(alpha=0.5)
    axes[3].set_title('Mean rescaling +/- std', fontsize=fs+2)

    plt.suptitle('Two-moons experiment', fontsize=fs+3)

    plt.tight_layout()

    plt.show()


plot_loss_dict(loss_histories, fs=16, figsize=(15, 5))


# %%
