from __future__ import annotations

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import random
from pathcond.models import UNet, DoubleConv
from pathcond.rescaling_polyn import optimize_rescaling_polynomial, reweight_model
from pathcond.utils import _ensure_outdir
import copy
import time
from pathcond.data import SyntheticDeblurDataset, get_deblur_loaders
from tqdm import tqdm
import numpy as np
from pathcond.models import toy_MLP
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap




def main():
    # Configuration du style ICML
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'lines.linewidth': 1.2,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    # Save data for later use
    data = torch.load('results/lazy/toy_MLP_lazy_training_trajectories.pt', weights_only=False)

    trajectories = data['trajectories']
    trajectories_rescaled = data['trajectories_rescaled']
    signs = data['signs']
    signs_rescaled = data['signs_rescaled']
    LOSS = data['LOSS']
    LOSS_rescaled = data['LOSS_rescaled']
    teacher = data['teacher']

    W_teacher = teacher.model[0].weight
    a_teacher = teacher.model[2].weight.view(-1)
    LAMBDA = torch.linspace(0, 100, 2000)

    n_neurons = trajectories.shape[0]
    n_neurons_rescaled = trajectories_rescaled.shape[0]

    # Créer une figure avec 2 lignes et 2 colonnes
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.5))

    # Fonction pour tracer les trajectoires
    def plot_trajectories(ax, traj_data, signs_data, t_end=None):
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        for i in range(len(traj_data)):
            traj = traj_data[i]  # shape (T, 2)
            if t_end is not None:
                traj = traj[:t_end]  # Limiter à t=2
            
            if signs_data[i].item() >= 0:
                ax.plot(traj[:, 0], traj[:, 1], color='#E74C3C', alpha=0.7, linewidth=0.8)
                ax.scatter(traj[-1, 0], traj[-1, 1], color='#E74C3C', s=8, alpha=0.9)
            else:
                ax.plot(traj[:, 0], traj[:, 1], color='#3498DB', alpha=0.7, linewidth=0.8)
                ax.scatter(traj[-1, 0], traj[-1, 1], color='#3498DB', s=8, alpha=0.9)
        
        ax.autoscale()

    # Plot 1: Standard à t=2 (top-left)
    plot_trajectories(axes[0, 0], trajectories, signs, t_end=2)
    
    # Plot 2: Standard trajectoire complète (top-right)
    plot_trajectories(axes[0, 1], trajectories, signs, t_end=None)
    
    # Plot 3: Rescaled à t=2 (bottom-left)
    plot_trajectories(axes[1, 0], trajectories_rescaled, signs_rescaled, t_end=2)
    
    # Plot 4: Rescaled trajectoire complète (bottom-right)
    plot_trajectories(axes[1, 1], trajectories_rescaled, signs_rescaled, t_end=None)

    # Calculer les limites pour la colonne de gauche (t=2)
    xlim_left_0 = axes[0, 0].get_xlim()
    ylim_left_0 = axes[0, 0].get_ylim()
    xlim_left_1 = axes[1, 0].get_xlim()
    ylim_left_1 = axes[1, 0].get_ylim()
    
    xlim_left = (min(xlim_left_0[0], xlim_left_1[0]), max(xlim_left_0[1], xlim_left_1[1]))
    ylim_left = (min(ylim_left_0[0], ylim_left_1[0]), max(ylim_left_0[1], ylim_left_1[1]))
    
    # Rendre les limites carrées pour la colonne de gauche
    x_range_left = xlim_left[1] - xlim_left[0]
    y_range_left = ylim_left[1] - ylim_left[0]
    max_range_left = max(x_range_left, y_range_left)
    
    x_center_left = (xlim_left[0] + xlim_left[1]) / 2
    y_center_left = (ylim_left[0] + ylim_left[1]) / 2
    
    xlim_left = (x_center_left - max_range_left/2, x_center_left + max_range_left/2)
    ylim_left = (y_center_left - max_range_left/2, y_center_left + max_range_left/2)
    
    # Calculer les limites pour la colonne de droite (trajectoire complète)
    xlim_right_0 = axes[0, 1].get_xlim()
    ylim_right_0 = axes[0, 1].get_ylim()
    xlim_right_1 = axes[1, 1].get_xlim()
    ylim_right_1 = axes[1, 1].get_ylim()
    
    xlim_right = (min(xlim_right_0[0], xlim_right_1[0]), max(xlim_right_0[1], xlim_right_1[1]))
    ylim_right = (min(ylim_right_0[0], ylim_right_1[0]), max(ylim_right_0[1], ylim_right_1[1]))
    
    # Rendre les limites carrées pour la colonne de droite
    x_range_right = xlim_right[1] - xlim_right[0]
    y_range_right = ylim_right[1] - ylim_right[0]
    max_range_right = max(x_range_right, y_range_right)
    
    x_center_right = (xlim_right[0] + xlim_right[1]) / 2
    y_center_right = (ylim_right[0] + ylim_right[1]) / 2
    
    xlim_right = (x_center_right - max_range_right/2, x_center_right + max_range_right/2)
    ylim_right = (y_center_right - max_range_right/2, y_center_right + max_range_right/2)
    
    # Appliquer les limites
    axes[0, 0].set_xlim(xlim_left)
    axes[0, 0].set_ylim(ylim_left)
    axes[1, 0].set_xlim(xlim_left)
    axes[1, 0].set_ylim(ylim_left)
    
    axes[0, 1].set_xlim(xlim_right)
    axes[0, 1].set_ylim(ylim_right)
    axes[1, 1].set_xlim(xlim_right)
    axes[1, 1].set_ylim(ylim_right)

    # Maintenant ajouter les Teacher directions (après avoir fixé les limites)
    with torch.no_grad():
        for ax in axes.flat:
            for i in range(W_teacher.shape[0]):
                point = (a_teacher[i].abs() * W_teacher[i]).cpu().numpy()
                if a_teacher[i] >= 0:
                    ax.plot(LAMBDA * point[0], LAMBDA * point[1],
                            color='#8B0000', linestyle='--', alpha=0.4, linewidth=1.0)
                else:
                    ax.plot(LAMBDA * point[0], LAMBDA * point[1],
                            color='#00008B', linestyle='--', alpha=0.4, linewidth=1.0)

    # Finaliser la mise en forme
    for ax in axes.flat:
        ax.set_aspect("equal", adjustable="box")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Ajouter les titres et labels
    # Première ligne (Standard)
    axes[0, 0].set_title(r"$\bf{Standard}$ - $t=2$", fontweight='normal')
    axes[0, 0].set_ylabel(r"$|a_i| w_{i,2}$")
    
    axes[0, 1].set_title(r"$\bf{Standard}$ - Full", fontweight='normal')
    
    # Deuxième ligne (Rescaled)
    axes[1, 0].set_title(r"$\bf{Rescaled}$ - $t=2$", fontweight='normal')
    axes[1, 0].set_xlabel(r"$|a_i| w_{i,1}$")
    axes[1, 0].set_ylabel(r"$|a_i| w_{i,2}$")
    
    axes[1, 1].set_title(r"$\bf{Rescaled}$ - Full", fontweight='normal')
    axes[1, 1].set_xlabel(r"$|a_i| w_{i,1}$")

    plt.tight_layout()
    plt.savefig("images/toy_MLP_lazy_training_trajectories.pdf", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Loss plot - style ICML
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, which='both')
    
    ax.plot(LOSS, label='Standard', color="#000000", linewidth=1.5)
    ax.plot(LOSS_rescaled, label='Rescaled', color="#000000", linewidth=1.5, linestyle='--')

    #vertical lines at 2 and 10000
    # ax.axvline(x=2, color='gray', linestyle=':', linewidth=1.0)
    # ax.axvline(x=10000, color='gray', linestyle=':', linewidth=1.0)
    
    ax.set_yscale('log')
    # ax.set_xscale('log')
    ax.set_xlabel('Training iteration')
    ax.set_ylabel('MSE loss')
    ax.legend(frameon=False, loc='center')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("images/toy_MLP_lazy_training_loss.pdf", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()