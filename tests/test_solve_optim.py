# %%
# Test solve min F(z)
import torch
import matplotlib.pyplot as plt
from pathcond.rescaling_polyn import compute_in_out_other_h


def one_not_in_span_rank(B, tol=1e-8):
    m = B.shape[0]
    ones = torch.ones((m, 1), device=B.device)

    # Compute ranks
    rank_B = torch.linalg.matrix_rank(B, tol=tol)
    rank_aug = torch.linalg.matrix_rank(torch.cat([B, ones], dim=1), tol=tol)

    return rank_aug > rank_B


def generate_B(shape, frac_neg=0.3, frac_zero=0.4, frac_pos=0.3, device="cpu", max_attempts=50):

    draw = True
    attempts = 0
    while draw:

        total = frac_neg + frac_zero + frac_pos
        probs = torch.tensor([frac_neg, frac_zero, frac_pos], dtype=torch.float, device=device) / total

        # Possible values
        values = torch.tensor([-1, 0, 1], device=device)

        # Sample from categorical distribution
        flat_samples = torch.multinomial(probs, num_samples=shape[0] * shape[1], replacement=True)
        mat = values[flat_samples].reshape(shape).to(torch.float)

        if one_not_in_span_rank(mat):
            draw = False
        else:
            attempts += 1

        if attempts >= max_attempts:
            raise ValueError('Max attempts reached')

    return mat


def loss(z, g, B):
    # B is n times H
    Bz = B @ z
    n = B.shape[0]
    if torch.all(g > 0):
        return n*torch.logsumexp(torch.log(g) + Bz, 0) - Bz.sum()
    else:
        v = g*torch.exp(Bz)
        return n*torch.log(v.sum()) - Bz.sum()


def update_z_polynomial(z, g, B):
    # Do one pass on every z_h
    # Maintain BZ incrementally: BZ = B @ Z
    BZ = B @ z
    n_params_tensor = B.shape[0]
    y_bar = 0.0  # Track max for numerical stability (not strictly necessary)
    for h in range(H):
        # Column for neuron h
        b_h = B[:, h]  # shape: [m]

        # Partition indices (must return index sets for rows of B/diag_G)
        in_h, out_h, other_h = compute_in_out_other_h(b_h)

        # Ensure torch index tensors on the right device
        in_h_t = torch.as_tensor(in_h, dtype=torch.long)
        out_h_t = torch.as_tensor(out_h, dtype=torch.long)
        other_h_t = torch.as_tensor(other_h, dtype=torch.long)

        card_in_h = int(in_h_t.numel())
        card_out_h = int(out_h_t.numel())
        # Leave-one-out energy vector: exp( (B @ Z) - b_h * Z[h] ) * diag_G
        # Using the maintained BZ avoids a full matmul here.
        Y_h = BZ - b_h * z[h]  # shape: [m]
        E = torch.exp(Y_h - y_bar) * g  # shape: [m]

        # Polynomial coefficients components
        # A_h is scalar (int), others are sums over selected rows of E
        A_h = (card_in_h - card_out_h)
        B_h = E[out_h_t].sum()

        C_h = E[in_h_t].sum()
        D_h = E[other_h_t].sum()

        # Polynomial: P(X) = a*X^2 + b*X + c where
        a = B_h * (A_h + n_params_tensor)
        b = D_h * A_h
        c = C_h * (A_h - n_params_tensor)

        if a <= 0.0:
            raise ValueError(
                f"Non-positive a={a} in quadratic for neuron {h} at iter {k}, A_h={A_h}, B_h={B_h}, C_h={C_h}, D_h={D_h}")
        if c >= 0.0:
            raise ValueError(
                f"Non-negative c={c} in quadratic for neuron {h} at iter {k}, A_h={A_h}, B_h={B_h}, C_h={C_h}, D_h={D_h}")

        # Degenerate to linear if a ~ 0
        if abs(a) < 1e-20:
            if abs(b) >= 1e-20:
                x = -c / b
                if x > 0.0:
                    z_new = torch.log(x)
                else:
                    raise ValueError(
                        f"Non-positive root {x} in linear case for neuron {h} at iter {k}, a={a}, b={b}, c={c}")
            else:
                if abs(c) < 1e-20:
                    raise ValueError(f"a = {a}, b = {b}, c = {c} all ~ 0 for neuron {h}")
                else:
                    raise ValueError(f"a = {a}, b = {b} both ~ 0 but c = {c} != 0 for neuron {h}")
        else:
            disc = torch.square(b) - 4.0 * a * c
            if disc > 0.0:
                sqrt_disc = torch.sqrt(disc)
                x1 = (-b + sqrt_disc) / (2.0 * a)
                x2 = (-b - sqrt_disc) / (2.0 * a)
                candidates = [x for x in (x1, x2) if x > 0.0]
                if len(candidates) != 1:
                    print("candidates:", candidates, x1, x2, a, b, c)
                    raise ValueError(
                        f"Unexpected number of positive roots {len(candidates)} for neuron {h}")
                z_new = torch.log(candidates[0])
            else:
                raise ValueError(f"Negative or infinit discriminant {disc} in quadratic for neuron {h}")
        # Update Z[h] and incrementally refresh BZ
        delta = z_new - float(z[h])
        if delta != 0.0:
            BZ = BZ + b_h * delta  # rank-1 update instead of recomputing B @ Z
            if abs(z_new) > abs(z[h]):
                if z_new > y_bar:
                    y_bar = z_new
            z[h] = z_new
    return z


# %%
n = 50
H = 5
g = torch.rand(n)
g[0] = 0
print(g.min())
B = generate_B((n, H))


OPTIMIZERS = {'SGD': torch.optim.SGD,
              'Adam': torch.optim.Adam,
              'NAdam': torch.optim.NAdam,
              'LBFGS': torch.optim.LBFGS}

lr = 1e-3
optimizer_name = 'SGD'
projection = True
loss_grad_descent = []
loss_polym = []
all_z_gd = []
all_z_polym = []

z = torch.ones(H, dtype=torch.float, requires_grad=True)
z_nograd = torch.ones(H)

loss_zero = loss(torch.zeros(H), g, B)
optimizer = OPTIMIZERS[optimizer_name]([z], lr=lr)
max_iter = 100

loss_grad_descent.append(loss(z.clone().detach(), g, B))
loss_polym.append(loss(z_nograd, g, B))

for i in range(max_iter):
    # do gd pass
    optimizer.zero_grad()
    output = loss(z, g, B)
    output.backward()
    optimizer.step()
    with torch.no_grad():
        loss_grad_descent.append(loss(z, g, B).item())
        all_z_gd.append(z.clone().detach())

    # do closed form
    z_nograd = update_z_polynomial(z_nograd, g, B)
    loss_polym.append(loss(z_nograd, g, B))
    all_z_polym.append(z_nograd)

all_z_polym = torch.stack(all_z_polym, dim=0)
all_z_gd = torch.stack(all_z_gd, dim=0)
# %%
fs = 14
cmap = plt.cm.get_cmap('tab10')
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(loss_grad_descent, '-', lw=2, label=optimizer_name)
ax[0].plot(loss_polym, '-', lw=2, label='Poly. update')

ax[0].hlines(y=loss_zero, xmin=0, xmax=max(len(loss_polym), len(loss_grad_descent)) +
             1, linestyle='--', color='k', alpha=0.5, label='Loss at z = 0')
ax[0].set_ylabel('Loss', fontsize=fs)
ax[0].set_xlabel('iter', fontsize=fs)
ax[0].legend(fontsize=fs)
# ax[0].set_yscale('log')
ax[0].grid(alpha=0.5)

scaling_polym = torch.exp(-0.5*(all_z_polym @ B.T))
scaling_gd = torch.exp(-0.5*(all_z_gd @ B.T))

ax[1].plot(range(len(scaling_gd.mean(1))), scaling_gd.mean(1), lw=3,
           label='mean scaling {}'.format(optimizer_name), color=cmap(0))
ax[1].plot(range(len(scaling_polym.mean(1))), scaling_polym.mean(1), lw=3, label='mean scaling polyn', color=cmap(1))

# axes[3].fill_between(all_rescaling.mean(0), lw=3)
ax[1].fill_between(range(len(scaling_gd.mean(1))),
                   scaling_gd.mean(1)-torch.std(scaling_gd, axis=1),
                   scaling_gd.mean(1)+torch.std(scaling_gd, axis=1),
                   alpha=0.3, color=cmap(0))

ax[1].fill_between(range(len(scaling_polym.mean(1))),
                   scaling_polym.mean(1)-torch.std(scaling_polym, axis=1),
                   scaling_polym.mean(1)+torch.std(scaling_polym, axis=1),
                   alpha=0.3, color=cmap(1))


ax[1].set_xlabel("iter", fontsize=fs)
ax[1].grid(alpha=0.5)
ax[1].set_title('Scaling (+/- std)', fontsize=fs)
ax[1].legend(fontsize=fs)

plt.suptitle('Solving $\min_z \ n\log(\sum g_i \exp((Bz)_i)) - \sum (Bz)_i$', fontsize=fs+2)

plt.tight_layout()
# %%
Bz_final = B @ z_nograd
print(Bz_final)
# %%
Bz_final = B @ z
print(Bz_final)
# %%
