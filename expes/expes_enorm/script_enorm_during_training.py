import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy
from sklearn.datasets import make_moons
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from pathcond.rescaling_polyn import optimize_rescaling_polynomial, reweight_model
from enorm import ENorm 
from pathcond.models import MLP

def get_param_norm(model):
    return torch.norm(torch.cat([p.flatten() for p in model.parameters()])).item()

def create_moon_dataset(n_samples=1000, noise=0.1, seed=42):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)
    return TensorDataset(X, y)

def train_baseline(model, train_loader, n_epochs=100, lr=0.01):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
    
    norms = []
    losses = []
    
    for epoch in tqdm(range(n_epochs), desc="Baseline"):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        norms.append(get_param_norm(model))
        losses.append(epoch_loss / len(train_loader))
    
    return norms, losses

def train_path_dynamics(model, train_loader, n_epochs=100, lr=0.01, n_iter_path=5):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    
    # Application de Path Dynamics au début
    print("Applying Path Dynamics rescaling...")
    # bz, z = optimize_rescaling_polynomial(model, n_iter=1, tol=1e-6, enorm=True)
    # model = reweight_model(model, bz, z, enorm=True)
    
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
    
    norms = []
    losses = []
    
    for epoch in tqdm(range(n_epochs), desc="Path Dynamics"):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            bz, z = optimize_rescaling_polynomial(model, n_iter=1, tol=1e-6, enorm=True)
            model = reweight_model(model, bz, z, enorm=True)
            
            optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
        
        norms.append(get_param_norm(model))
        losses.append(epoch_loss / len(train_loader))
    
    return norms, losses

def train_enorm(model, train_loader, n_epochs=100, lr=0.01):
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0)
    
    try:
        enorm_obj = ENorm(model.named_parameters(), model_type="linear", optimizer=optimizer)
    except:
        print("ENorm not available, skipping...")
        return None, None
    
    norms = []
    losses = []
    
    for epoch in tqdm(range(n_epochs), desc="ENorm"):
        epoch_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            # Application ENorm après chaque step SGD
            enorm_obj.step()
            
            epoch_loss += loss.item()
        
        norms.append(get_param_norm(model))
        losses.append(epoch_loss / len(train_loader))
    
    return norms, losses

def run_comparison(hidden_size=64, n_epochs=100, lr=0.01, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Création du dataset
    dataset = create_moon_dataset(n_samples=1000, noise=0.1)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Modèle simple pour 2 moons
    def create_model(seed=42):
        torch.manual_seed(seed)
        return MLP(hidden_dims=[2, hidden_size, hidden_size, 2]).to(device)
    
    # --- Baseline ---
    print("\n=== Training Baseline ===")
    seed = 0
    model_original = create_model(seed=seed)
    model_baseline = copy.deepcopy(model_original)
    norms_baseline, losses_baseline = train_baseline(model_baseline, train_loader, n_epochs, lr)
    
    # --- Path Dynamics ---
    print("\n=== Training Path Dynamics ===")
    model_path = copy.deepcopy(model_original)
    # Copy weights from baseline for fair comparison
    model_path.load_state_dict(create_model(seed=seed).state_dict())
    norms_path, losses_path = train_path_dynamics(model_path, train_loader, n_epochs, lr)
    
    # --- ENorm ---
    print("\n=== Training ENorm ===")
    model_enorm = copy.deepcopy(model_original)
    model_enorm.load_state_dict(create_model(seed=seed).state_dict())
    norms_enorm, losses_enorm = train_enorm(model_enorm, train_loader, n_epochs, lr)
    
    # --- Plotting ---
    plt.rcParams.update({
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
    })
    
    #fig, ax1 = plt.subplots(1, 1, figsize=(3.25, 2.5)) # si 2 figures  figsize=(6.4, 3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 3))
    
    # Plot 1: Parameter norms
    epochs = range(1, n_epochs + 1)
    ax1.plot(epochs, norms_baseline, '-', color='#95a5a6', label='Baseline', linewidth=2, alpha=0.8)
    ax1.plot(epochs, norms_path, '-', color='#E74C3C', label=r'DAG-ENorm $\bf{(Ours)}$', linewidth=2)
    if norms_enorm:
        ax1.plot(epochs, norms_enorm, '-', color='#3498DB', label='ENorm', linewidth=2)
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(r'$\|\theta\|_2$')
    #ax1.set_title('Parameter norm evolution')
    ax1.legend(frameon=False, loc='best')
    ax1.grid(True, alpha=0.2, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Plot 2: Training loss
    ax2.plot(epochs, losses_baseline, '-', color='#95a5a6', label='Baseline', linewidth=2, alpha=0.8)
    ax2.plot(epochs, losses_path, '-', color='#E74C3C', label=r'DAG-ENorm $\bf{(Ours)}$', linewidth=2)
    if losses_enorm:
        ax2.plot(epochs, losses_enorm, '-', color='#3498DB', label='ENorm', linewidth=2)
    
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training loss')
    # ax2.set_title('Loss evolution')
    ax2.legend(frameon=False, loc='best')
    ax2.grid(True, alpha=0.2, linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('images/moons_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print final statistics
    print("\n" + "="*60)
    print(f"{'Method':<20} {'Final Norm':<15} {'Final Loss':<15}")
    print("="*60)
    print(f"{'Baseline':<20} {norms_baseline[-1]:<15.4f} {losses_baseline[-1]:<15.4f}")
    print(f"{'Path Dynamics':<20} {norms_path[-1]:<15.4f} {losses_path[-1]:<15.4f}")
    if norms_enorm:
        print(f"{'ENorm':<20} {norms_enorm[-1]:<15.4f} {losses_enorm[-1]:<15.4f}")
    print("="*60)

# Lancer l'expérience
run_comparison(hidden_size=16, n_epochs=100, lr=0.1, batch_size=32)