import torch
import torch.nn as nn
from typing import List, Dict, Any

from torch.nn.utils import parameters_to_vector
from pathcond.rescaling_polyn import optimize_neuron_rescaling_polynomial, reweight_model, compute_diag_G, optimize_rescaling_gd, compute_matrix_B, function_F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class MLP(nn.Sequential):
    def __init__(
        self,
        hidden_dims: List[int],
    ) -> None:
        super().__init__()

        self.input_dim = hidden_dims[0]
        self.output_dim = hidden_dims[-1]
        self.hidden_dims = hidden_dims[1:-1]

        layers = []
        prev = self.input_dim

        # Hidden layers: Linear + ReLU, bias=True
        for h in self.hidden_dims:
            layers.append(nn.Linear(prev, h, bias=True))
            layers.append(nn.ReLU())
            prev = h

        # Output layer: bias=False (comme dans ton code)
        layers.append(nn.Linear(prev, self.output_dim, bias=False))

        self.model = nn.Sequential(*layers)

    def forward(self, x, device='cpu'):
        x = x.to(device)
        return self.model(x)

def apply_init(model: nn.Module, scheme: str, gain: float = None) -> None:
    """
    scheme ∈ {"xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal",
              "uniform", "normal", "zeros", "orthogonal"}
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(m.weight, gain=gain if gain is not None else 1.0)
            elif scheme == "xavier_normal":
                nn.init.xavier_normal_(m.weight, gain=gain if gain is not None else 1.0)
            elif scheme == "kaiming_uniform":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif scheme == "uniform":
                nn.init.uniform_(m.weight, a=-0.05, b=0.05)
            elif scheme == "normal":
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            elif scheme == "zeros":
                nn.init.zeros_(m.weight)
            elif scheme == "default":
                pass  # laisse les poids tels quels
            elif scheme == "orthogonal":
                nn.init.orthogonal_(m.weight, gain=gain if gain is not None else 1.0)
            else:
                raise ValueError(f"Schéma d'init inconnu: {scheme}")

            if m.bias is not None:
                nn.init.zeros_(m.bias)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_grid(
    architectures: List[List[int]],
    inits: List[str],
    device: str = None,
    seed: int = 123
) -> List[Dict[str, Any]]:
    """
    Retourne une liste de dictionnaires avec les résultats pour chaque (arch, init).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch.manual_seed(seed)
    results = []

    for arch in architectures:
        for init_name in inits:

            model = MLP(arch).to(device)
            model.eval()


            gain = nn.init.calculate_gain("relu")
            apply_init(model, init_name, gain=gain)


            dG = compute_diag_G(model)  # Tensor de taille [#params] ou shape compatible


            with torch.no_grad():
                P = num_params(model)
                BZ = torch.zeros(P, device=device)

            outF = function_F(P, BZ, dG)

            BZ_opt = optimize_neuron_rescaling_polynomial(
                model=model,
                n_iter=15,
                verbose=False,
                tol=1e-6
            )

            # new_dG = dG*torch.exp(BZ_opt)

            # model_rescaled = reweight_model(model, BZ_opt)
            # new_dG_model = compute_diag_G(model_rescaled)

            # assert torch.allclose(new_dG, new_dG_model, atol=1e-5), "Incohérence entre dG recalculé et dG via modèle rescalé"

            outF2 = function_F(P, BZ_opt, dG)

            res = torch.abs(outF2 - outF) / torch.abs(outF)

            results.append({
                "architecture": arch,
                "init": init_name,
                "num_params": P,
                "F_output": res,
            })


    return results


def plot_boxplot(df: pd.DataFrame, title: str = "Distribution of F_output over 10 runs"):
    """
    Crée un boxplot soigné pour comparer F_output entre architectures et initialisations.

    Args:
        df (pd.DataFrame): DataFrame contenant les colonnes ["init", "arch_str", "F_output"].
        title (str): Titre du graphique.
    """

    # Style général (inspiré des figures de conf)
    sns.set_theme(style="whitegrid", context="talk", font_scale=1.3)
    plt.figure(figsize=(12, 6))

    # Couleurs par architecture
    palette = sns.color_palette("viridis", df["arch_str"].nunique())

    # Boxplot avec éléments esthétiques
    ax = sns.boxplot(
        data=df,
        x="init",
        y="F_output",
        hue="arch_str",
        palette=palette,
        linewidth=1.5,
        fliersize=3,
        boxprops=dict(alpha=0.8)
    )

    # Ajout des points individuels (jitter léger)
    sns.stripplot(
        data=df,
        x="init",
        y="F_output",
        hue="arch_str",
        dodge=True,
        jitter=0.15,
        alpha=0.4,
        size=4,
        palette=palette,
        ax=ax
    )

    # Gestion des légendes (fusion hue)
    handles, labels = ax.get_legend_handles_labels()
    n_arch = df["arch_str"].nunique()
    plt.legend(handles[:n_arch], labels[:n_arch],
               title="Architecture",
               bbox_to_anchor=(1.02, 1),
               loc="upper left",
               borderaxespad=0.)

    # Mise en forme du graphique
    plt.xticks(rotation=25, ha="right")
    plt.xlabel("Initialization scheme", fontsize=14, labelpad=10)
    plt.ylabel("Relative difference in F", fontsize=14, labelpad=10)
    plt.title(title, fontsize=18, pad=15, weight="bold")

    # Alléger le fond
    sns.despine(trim=True)
    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    architectures = [
        [2, 10, 10, 2],
        [2, 20, 2],
        [2, 10, 5, 10, 5, 10, 2]
    ]
    inits = [
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
        "default"
    ]

    n_runs = 10
    all_results = []

    for i in range(n_runs):
        print(f"\n=== Run {i+1}/{n_runs} ===")
        results = run_grid(architectures, inits, seed=42 + i)
        for r in results:
            r["run"] = i
        all_results.extend(results)

    # Conversion en DataFrame
    df = pd.DataFrame(all_results)
    df["arch_str"] = df["architecture"].apply(lambda a: "-".join(map(str, a)))
    df["F_output"] = df["F_output"].apply(lambda x: x.item() if isinstance(x, torch.Tensor) else x)

    # Moyenne et variance
    stats = df.groupby(["arch_str", "init"])["F_output"].agg(["mean", "std"]).reset_index()
    print("\n=== Moyenne et écart-type sur 10 runs ===")
    print(stats)

    plot_boxplot(df, title="Distribution of Relative Difference in F over 10 runs")
