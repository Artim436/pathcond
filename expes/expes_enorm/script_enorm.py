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

def benchmark_models(hidden_sizes=[16, 64, 256, 512, 1024], n_iter_path=1, n_iter_enorm=1000):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    results = {
        "size": [],
        "initial": [],
        "path_dynamics": [],
        "enorm": [],
        "time_path": [],
        "time_enorm": []
    }

    print(f"Démarrage du benchmark sur {len(hidden_sizes)} modèles...")

    for hs in tqdm(hidden_sizes):
        # Initialisation d'un modèle ComplexModel
        model_base = MLP(hidden_dims=[784, hs, hs//2, 10]).to(device)
        num_params = count_parameters(model_base)

        inputs = torch.randn(1,784).to(device)
        orig_output = model_base(inputs)
        
        results["size"].append(num_params)
        results["initial"].append(get_param_norm(model_base))

        # --- Test Path Dynamics ---
        m_path = copy.deepcopy(model_base)
        start_time = time.time()
        bz, z = optimize_rescaling_polynomial(m_path, n_iter=n_iter_path, tol=1e-6, enorm=True)
        m_path = reweight_model(m_path, bz, z, enorm=True)
        path_output = m_path(inputs)
        print("diff output path dynamics:", torch.norm(orig_output - path_output).item())
        time_path = time.time() - start_time
        
        results["path_dynamics"].append(get_param_norm(m_path))
        results["time_path"].append(time_path)

        # --- Test ENorm ---
        m_enorm = copy.deepcopy(model_base)
        optimizer = optim.SGD(m_enorm.parameters(), lr=1.0, momentum=0.9)
        try:
            enorm_obj = ENorm(m_enorm.named_parameters(), model_type="linear", optimizer=optimizer)
            start_time = time.time()
            for _ in range(n_iter_enorm):
                enorm_obj.step()
            time_enorm = time.time() - start_time

            enorm_output = m_enorm(inputs)
            print("diff output ENorm:", torch.norm(orig_output - enorm_output).item())
            
            results["enorm"].append(get_param_norm(m_enorm))
            results["time_enorm"].append(time_enorm)
        except:
            results["enorm"].append(None)
            results["time_enorm"].append(None)

    # --- GÉNÉRATION DU GRAPHIQUE STYLE ICML ---
    # Configuration du style ICML
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
    
    fig, ax1 = plt.subplots(1, 1, figsize=(3.25, 2.5)) # si 2 figures  figsize=(6.4, 3)
    
    # --- Plot 1: Norme des paramètres ---
    ax1.plot(results["size"], results["initial"], 'o-.', 
             color='#95a5a6', label='Initial', alpha=0.7, markersize=5, zorder=5)
    ax1.plot(results["size"], results["path_dynamics"], 's-', 
             color='#E74C3C', linewidth=2, label=r'DAG-ENorm $\bf{(Ours)}$', markersize=6)
    
    if any(results["enorm"]):
        ax1.plot(results["size"], results["enorm"], '^-', 
                 color='#3498DB', linewidth=2, label=f'ENorm', markersize=6)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Number of parameters')
    ax1.set_ylabel(r'$\|\theta\|_2$')
    #ax1.set_title('Parameter norm reduction')
    ax1.legend(frameon=False)
    ax1.grid(True, which="both", alpha=0.2, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # # --- Plot 2: Temps de calcul ---
    # ax2.plot(results["size"], results["time_path"], 's-', 
    #          color='#E74C3C', linewidth=2, label=f'Path Dynamics', markersize=6)
    
    # if any(results["time_enorm"]):
    #     valid_enorm_times = [(s, t) for s, t in zip(results["size"], results["time_enorm"]) if t is not None]
    #     if valid_enorm_times:
    #         sizes_enorm, times_enorm = zip(*valid_enorm_times)
    #         ax2.plot(sizes_enorm, times_enorm, '^-', 
    #                  color='#3498DB', linewidth=2, label=f'ENorm', markersize=6)

    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.set_xlabel('Number of parameters')
    # ax2.set_ylabel('Computation time (s)')
    # ax2.set_title('Computational efficiency')
    # ax2.legend(frameon=False)
    # ax2.grid(True, which="both", alpha=0.2, linestyle='--')
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    
    # Ajout d'annotations pour les tailles de modèles
    # for i, hs in enumerate(hidden_sizes[::2]):  # Annoter seulement 1 sur 2 pour éviter le surcharge
    #     idx = i * 2
    #     if idx < len(results["size"]):
    #         ax1.annotate(f"{hs}", (results["size"][idx], results["path_dynamics"][idx]), 
    #                     textcoords="offset points", xytext=(5, 5), ha='left', 
    #                     fontsize=8, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('images/benchmark_rescaling.pdf', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()
    
    # --- Affichage d'un tableau récapitulatif ---
    print("\n" + "="*80)
    print(f"{'Hidden Size':<12} {'Parameters':<15} {'Time Path (s)':<15} {'Time ENorm (s)':<15} {'Speedup':<10}")
    print("="*80)
    for i, hs in enumerate(hidden_sizes):
        speedup = results["time_enorm"][i] / results["time_path"][i] if results["time_enorm"][i] else "N/A"
        speedup_str = f"{speedup:.2f}x" if isinstance(speedup, float) else speedup
        print(f"{hs:<12} {results['size'][i]:<15} {results['time_path'][i]:<15.4f} "
              f"{results['time_enorm'][i] if results['time_enorm'][i] else 'N/A':<15} {speedup_str:<10}")
    print("="*80)

    return results

# Pour lancer le test
results = benchmark_models(hidden_sizes=[32, 64, 128, 256, 512], n_iter_path=100, n_iter_enorm=1000)