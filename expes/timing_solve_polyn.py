# %% Time expe
import time
import torch
from pathcond.rescaling_polyn import compute_in_out_other_h
import matplotlib.pyplot as plt
# %%


def one_not_in_span_rank(B, tol=1e-8):
    n = B.shape[0]
    ones = torch.ones((n, 1), device=B.device)

    # Compute ranks
    rank_B = torch.linalg.matrix_rank(B, tol=tol)
    rank_aug = torch.linalg.matrix_rank(torch.cat([B, ones], dim=1), tol=tol)

    return rank_aug > rank_B


def generate_B(shape, frac_neg=0.3, frac_zero=0.4, frac_pos=0.3,
               device="cpu", max_attempts=50):
    m, n = shape
    total = frac_neg + frac_zero + frac_pos
    probs = torch.tensor([frac_neg, frac_zero, frac_pos], dtype=torch.float, device=device) / total

    attempts = 0
    while True:
        mat = torch.empty((m, n), device=device)

        # Build each column with exact fractions
        for j in range(n):
            # counts
            n_neg = int(round(probs[0].item() * m))
            n_zero = int(round(probs[1].item() * m))
            n_pos = m - n_neg - n_zero  # ensure sum matches exactly

            col = torch.cat([
                -torch.ones(n_neg, device=device),
                torch.zeros(n_zero, device=device),
                torch.ones(n_pos, device=device)
            ])

            # Shuffle the column
            perm = torch.randperm(m, device=device)
            mat[:, j] = col[perm]

        # Check span condition
        if one_not_in_span_rank(mat):
            return mat.to(torch.float)

        attempts += 1
        if attempts >= max_attempts:
            raise ValueError("Max attempts reached")


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
        y_bar = Y_h.max()
        E = torch.exp(Y_h - y_bar) * g  # shape: [m]

        # Polynomial coefficients components
        # A_h is scalar (int), others are sums over selected rows of E
        A_h = (card_in_h - card_out_h)
        B_h = E[out_h_t].sum() if out_h.numel() > 0 else torch.tensor(0.0, device=z.device)
        C_h = E[in_h_t].sum() if in_h.numel() > 0 else torch.tensor(0.0, device=z.device)
        D_h = E[other_h_t].sum() if other_h.numel() > 0 else torch.tensor(0.0, device=z.device)

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
                # print('E.max = {}'.format(E.max()))
                # print('a = {}, b = {}, c = {}'.format(a, b, c))
                # print('candidates = {}'.format(candidates))
            else:
                # print('a = {}, b = {}, c = {}'.format(a, b, c))
                # print('Ah = {}, Bh = {}, Ch = {}, Dh = {}'.format(A_h, B_h, C_h, D_h))
                # print('h = {}'.format(h))
                # #print('Y_h = {}'.format(Y_h))
                # print('BZ = {}'.format(BZ))
                # print('b_h = {}'.format(b_h))
                # print('candidates = {}'.format(candidates))

                raise ValueError(f"Negative or infinit discriminant {disc} in quadratic for neuron {h}")
        # Update Z[h] and incrementally refresh BZ
        delta = z_new - float(z[h])
        if delta != 0.0:
            BZ = BZ + b_h * delta  # rank-1 update instead of recomputing B @ Z
            # if abs(z_new) > abs(z[h]):
            #     if z_new > y_bar:
            #         y_bar = z_new
            z[h] = z_new
    return z


@torch.jit.script
def update_z_polynomial_jit(z, g, B):
    # Do one pass on every z_h
    # Maintain BZ incrementally: BZ = B @ Z
    BZ = B @ z
    n_params_tensor = B.shape[0]
    H = B.shape[1]

    mask_in = (B == -1)
    mask_out = (B == 1)
    mask_other = (B == 0)
    card_in = mask_in.sum(dim=0)   # [H]
    card_out = mask_out.sum(dim=0)  # [H]

    for h in range(H):
        b_h = B[:, h]

        # directly use precomputed card
        A_h = int(card_in[h].item()) - int(card_out[h].item())

        # Leave-one-out energy vector
        Y_h = BZ - b_h * z[h]
        y_bar = Y_h.max()
        E = torch.exp(Y_h - y_bar) * g

        # sums using masks
        B_h = (E * mask_out[:, h]).sum()
        C_h = (E * mask_in[:, h]).sum()
        D_h = (E * mask_other[:, h]).sum()

        # Polynomial coefficients
        a = B_h * (A_h + n_params_tensor)
        b = D_h * A_h
        c = C_h * (A_h - n_params_tensor)

        disc = b * b - 4.0 * a * c
        sqrt_disc = torch.sqrt(disc)
        x1 = (-b + sqrt_disc) / (2.0 * a)
        x2 = (-b - sqrt_disc) / (2.0 * a)

        z_new = torch.log(torch.maximum(x1, x2))

        # Update Z[h] and incrementally refresh BZ
        delta = z_new - z[h]
        BZ = BZ + b_h * delta
        z[h] = z_new

    return z


# %%
n = 512
H = 256
a = 0
b = 1
g = (b-a)*torch.rand(n)+a
B = generate_B((n, H), frac_neg=0.4, frac_zero=0.2, frac_pos=0.4)
z = torch.randn(H)

z_new = update_z_polynomial(z, g, B)
z_new_jit = update_z_polynomial_jit(z, g, B)
# %%
print(z_new_jit-z_new)
# %%
all_H = [2**k for k in range(1, 13)]  # number of hidden neurons
all_n = [2**k for k in range(1, 13)]  # number of parameters
n_repeat = 10
time_jit = torch.zeros((n_repeat, len(all_n), len(all_H)))
time_no_jit = torch.zeros((n_repeat, len(all_n), len(all_H)))

for r in range(n_repeat):
    for i, n in enumerate(all_n):
        for j, H in enumerate(all_H):
            if n > H:
                try:
                    a = 0
                    b = 1
                    g = (b-a)*torch.rand(n)+a
                    B = generate_B((n, H), frac_neg=0.4, frac_zero=0.2, frac_pos=0.4)
                    z = torch.randn(H)

                    st = time.time()
                    _ = update_z_polynomial(z, g, B)
                    ed = time.time()
                    time_no_jit[r, i, j] = ed - st

                    st = time.time()
                    _ = update_z_polynomial_jit(z, g, B)
                    ed = time.time()
                    time_jit[r, i, j] = ed - st
                except ValueError as e:
                    print((r, n, H))
                    raise e

# %%

# %%
cmap = plt.cm.get_cmap('tab10')
fs = 15
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

for k, i in enumerate([10, 11]):
    n = all_n[i]
    all_H_n = [h for j, h in enumerate(all_H) if h < n]
    idx_h = [j for j, h in enumerate(all_H) if h < n]
    timenojit = time_no_jit[:, i, idx_h]
    timejit = time_jit[:, i, idx_h]

    ax[k].plot(all_H_n, timenojit.mean(0), label='No jit', lw=2, c=cmap(0))
    ax[k].plot(all_H_n, timejit.mean(0), label='jit', lw=2, c=cmap(1))

    ax[k].fill_between(all_H_n,
                       timejit.mean(0)-torch.std(timejit, axis=0),
                       timejit.mean(0)+torch.std(timejit, axis=0),
                       alpha=0.3, color=cmap(1))

    ax[k].fill_between(all_H_n,
                       timenojit.mean(0)-torch.std(timenojit, axis=0),
                       timenojit.mean(0)+torch.std(timenojit, axis=0),
                       alpha=0.3, color=cmap(0))

    ax[k].grid(alpha=0.5)
    ax[k].legend(fontsize=fs)
    ax[k].set_yscale('log')
    ax[k].set_xscale('log')
    ax[k].tick_params(axis='both', which='major', labelsize=fs-1)
    ax[k].tick_params(axis='both', which='minor', labelsize=fs-2)

    ax[k].set_xlabel('# hidden neurons $H$', fontsize=fs)
    ax[k].set_ylabel('time (in sec.)', fontsize=fs)
    ax[k].set_title('\n # params $n$ = {}'.format(n), fontsize=fs+2)
plt.suptitle('Timing one pass $z$', fontsize=fs+2)
plt.tight_layout()
# %%
