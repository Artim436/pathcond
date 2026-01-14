import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
import time
from torch.nn.utils import parameters_to_vector
from pathcond.rescaling_polyn import optimize_rescaling_polynomial, reweight_model
from enorm import ENorm 
from tqdm import tqdm
from pathcond.models import MLP
from torch import Tensor
from typing import List, Tuple


def get_param_norm(model):
    return torch.norm(parameters_to_vector(model.parameters())).item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def track_rescaling_iterations(hidden_size=128, n_iter_path=100, n_iter_enorm=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialisation d'un seul modèle
    model_base = MLP(hidden_dims=[784, hidden_size, hidden_size//2, 10]).to(device)
    num_params = count_parameters(model_base)
    
    print(f"Modèle avec {num_params} paramètres")
    
    inputs = torch.randn(1, 784).to(device)
    orig_output = model_base(inputs)
    initial_norm = get_param_norm(model_base)
    
    # --- Track Path Dynamics ---
    print("\n=== Path Dynamics ===")
    m_path = copy.deepcopy(model_base)
    norms_path = [initial_norm]
    times_path = [0.0]
    
    start_time = time.time()
    
    # Modifier optimize_rescaling_polynomial pour retourner l'historique
    # Ici on fait une version simplifiée qui appelle la fonction itérativement
    for i in tqdm(range(n_iter_path), desc="Path Dynamics"):
        bz, z = optimize_rescaling_polynomial(m_path, n_iter=1, tol=1e-6, enorm=True)
        m_path = reweight_model(m_path, bz, z, enorm=True)
        norms_path.append(get_param_norm(m_path))
        times_path.append(time.time() - start_time)
    
    path_output = m_path(inputs)
    print(f"Diff output Path Dynamics: {torch.norm(orig_output - path_output).item():.6e}")
    print(f"Final norm: {norms_path[-1]:.4f}")
    print(f"Total time: {times_path[-1]:.4f}s")

    # --- Track ENorm ---
    print("\n=== ENorm ===")
    m_enorm = copy.deepcopy(model_base)
    optimizer = optim.SGD(m_enorm.parameters(), lr=1.0, momentum=0.9)
    enorm_obj = ENorm(m_enorm.named_parameters(), model_type="linear", optimizer=optimizer)
    
    norms_enorm = [initial_norm]
    times_enorm = [0.0]
    
    start_time = time.time()
    for i in tqdm(range(n_iter_enorm), desc="ENorm"):
        enorm_obj.step()
        norms_enorm.append(get_param_norm(m_enorm))
        times_enorm.append(time.time() - start_time)
    
    enorm_output = m_enorm(inputs)
    print(f"Diff output ENorm: {torch.norm(orig_output - enorm_output).item():.6e}")
    print(f"Final norm: {norms_enorm[-1]:.4f}")
    print(f"Total time: {times_enorm[-1]:.4f}s")

    # --- GÉNÉRATION DU GRAPHIQUE STYLE ICML ---
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    fig, ax1 = plt.subplots(1, 1, figsize=(3.25, 2.5))
    
    # --- Plot 1: Norme vs itérations ---
    iters_path = range(len(norms_path))
    iters_enorm = range(len(norms_enorm))
    
    ax1.plot(iters_path, norms_path, '-', 
             color='#E74C3C', linewidth=2, label=r'DAG-ENorm $\bf{(Ours)}$', alpha=0.9)
    ax1.plot(iters_enorm, norms_enorm, '-', 
             color='#3498DB', linewidth=2, label='ENorm', alpha=0.9)
    ax1.axhline(y=initial_norm, color='#95a5a6', linestyle='--', 
                linewidth=1.5, label='Initial', alpha=0.7)

    ax1.set_xlabel('Iteration')
    ax1.set_ylabel(r'$\|\theta\|_2$')
    # ax1.set_title('Parameter norm evolution')
    ax1.legend(frameon=False, loc='best')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_yscale('log')
    
    # # --- Plot 2: Norme vs temps ---
    # ax2.plot(times_path, norms_path, '-', 
    #          color='#E74C3C', linewidth=2, label='Path Dynamics', alpha=0.9)
    # ax2.plot(times_enorm, norms_enorm, '-', 
    #          color='#3498DB', linewidth=2, label='ENorm', alpha=0.9)
    # ax2.axhline(y=initial_norm, color='#95a5a6', linestyle='--', 
    #             linewidth=1.5, label='Initial', alpha=0.7)

    # ax2.set_xlabel('Time (s)')
    # ax2.set_ylabel(r'$\|\theta\|_2$')
    # ax2.set_title('Parameter norm vs. computation time')
    # ax2.legend(frameon=False, loc='best')
    # ax2.grid(True, alpha=0.2, linestyle='--')
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    # ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('images/rescaling_iterations.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # --- Tableau récapitulatif ---
    print("\n" + "="*70)
    print(f"{'Method':<20} {'Initial Norm':<15} {'Final Norm':<15} {'Time (s)':<15}")
    print("="*70)
    print(f"{'Path Dynamics':<20} {initial_norm:<15.4f} {norms_path[-1]:<15.4f} {times_path[-1]:<15.4f}")
    print(f"{'ENorm':<20} {initial_norm:<15.4f} {norms_enorm[-1]:<15.4f} {times_enorm[-1]:<15.4f}")
    print("="*70)
    print(f"Norm reduction Path: {initial_norm / norms_path[-1]:.2f}x")
    print(f"Norm reduction ENorm: {initial_norm / norms_enorm[-1]:.2f}x")
    print(f"Speed ratio (ENorm/Path): {times_enorm[-1] / times_path[-1]:.2f}x")
    print("="*70)
    
    return {
        'norms_path': norms_path,
        'norms_enorm': norms_enorm,
        'times_path': times_path,
        'times_enorm': times_enorm,
        'initial_norm': initial_norm
    }

# Lancer l'expérience
results = track_rescaling_iterations(hidden_size=512, n_iter_path=10, n_iter_enorm=10)