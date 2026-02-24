# PathCond — Path-Conditioned Training

**PathCond** is a principled and computationally cheap method to rescale ReLU neural networks at initialization. A single call to `rescaling_path_cond` before the first gradient step improves the conditioning of the loss landscape, accelerating training convergence without degrading generalization performance.

This repository contains the implementation described in our paper:

> **Path-conditioned training: a principled way to rescale ReLU neural networks**  
> *Arthur Lebeurrier, Titouan Vayer, Rémi Gribonval*  
> [[arXiv link](https://arxiv.org/abs/2602.19799)]

---

## Table of contents

- [How it works](#how-it-works)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Reproducing the paper experiments](#reproducing-the-paper-experiments)
- [Interactive demos](#interactive-demos)
- [License](#license)
- [Citation](#citation)
<!-- - [Repository structure](#repository-structure) -->

---

## How it works

For any ReLU network, there exist infinitely many weight configurations that compute *exactly* the same function. For a single hidden neuron with input weight $u$, output weight $v$, bias $b$, and a rescaling coefficient $\lambda > 0$:

$$v \cdot \text{ReLU}(u x + b) = \frac{v}{\lambda} \cdot \text{ReLU}(\lambda u x + \lambda b)$$

This leaves the network function **exactly unchanged** by ReLU positive homogeneity. PathCond chooses the scalings $\lambda$ to optimize a conditioning criterion.

### The lifted perspective

We consider a lifting of the network parameters, denoted $\Phi$, that is invariant under rescaling (like the network's realization map). Selecting an admissible rescaling corresponds to shaping the geometry of the optimization dynamics in this lifted space.

Specifically, we seek the rescaling that best aligns the **path magnitude matrix** $G = \partial\Phi^\top \partial\Phi$ with the identity.

### Algorithm

1. **Compute** $\text{diag}(G)$ via a single backward pass.
2. **Solve** the optimization problem to find the optimal rescalings (`update_z_polynomial`).
3. **Apply** the rescaling in-place to the model weights (`reweight_model_in_place`).

PathCond supports MLPs, convolutional networks, and more general architectures (see [our paper](#citation) for details).

---

## Installation

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0.

```bash
git clone https://github.com/[TODO: repo]/pathcond.git
cd pathcond

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Then install depending on your use case:

```bash
# Use PathCond in your own project — installs torch only
pip install -e .

# Reproduce the paper experiments — adds mlflow, tqdm, seaborn, scipy, ...
pip install -e ".[expes]"

# Run the demo notebooks — adds jupyter, scikit-learn, matplotlib, ...
pip install -e ".[demo]"

# Everything (everywhere all) at once
pip install -e ".[all]"
```

---

## Quick start

```python
from pathcond.pathcond import rescaling_path_cond

# Define your model, criterion, and optimizer as usual
model     = ...
criterion = ...
optimizer = ...

# Apply PathCond once, before the first gradient step
rescaling_path_cond(model)

# Standard training loop — no other changes needed
for inputs, targets in train_loader:
    optimizer.zero_grad()
    loss = criterion(model(inputs), targets)
    loss.backward()
    optimizer.step()
```

PathCond is a **drop-in, minimal-overhead modification** of your initialization: it only touches the weights, leaves the realized function unchanged, and requires no modification to the training loop.

---

<!-- ## Repository structure

```
pathcond/
├── pathcond/                    # Core library
│   ├── pathcond.py              #   rescaling_path_cond       — main entry point
│   ├── network_to_optim.py      #   compute_diag_G            — diagonal of G via one backward
│   │                            #   compute_B_mlp / _conv     — sparse path-neuron matrix B
│   ├── rescaling_polyn.py       #   update_z_polynomial       — dual variable solver
│   │                            #   reweight_model_in_place   — applies optimal rescaling
│   ├── models.py                #   model definitions used in experiments
│   ├── data.py                  #   dataset utilities
│   └── utils.py                 #   miscellaneous helpers
├── expes/                       # Experiment scripts
│   ├── run_expe_figure_2.py     #   run experiments for Figure 2
│   ├── plot_figure_2.py         #   generate Figure 2
│   ├── run_expe_figure_3.py     #   run experiments for Figure 3
│   ├── plot_figure_3.py         #   generate Figure 3
│   ├── training_utils.py        #   shared training utilities
│   └── train_script.py          #   generic training script
├── pathcond_demo.ipynb          # Demo — two moons (MLP)
├── pathcond_cifar10_demo.ipynb  # Demo — CIFAR-10 (CIFAR-NV conv net)
├── figures/                     # Output figures
├── data/                        # Downloaded datasets (auto-populated)
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

--- -->

## Reproducing the paper experiments

Experiments are tracked with [MLflow](https://mlflow.org/). Results are logged to `mlruns/` and can be browsed with:

```bash
mlflow ui
```

### Figure 2

```bash
python3 expes/run_expe_figure_2.py
python3 expes/plot_figure_2.py
```

### Figure 3

```bash
python3 expes/run_expe_figure_3.py
python3 expes/plot_figure_3.py
```

---

## Interactive demos

Two self-contained Jupyter notebooks provide a hands-on walkthrough of PathCond:

| Notebook | Architecture | Task |
|---|---|---|
| [`pathcond_demo.ipynb`](pathcond_demo.ipynb) | MLP | Two Moons (2D) | 
| [`pathcond_cifar10_demo.ipynb`](pathcond_cifar10_demo.ipynb) | CIFAR-NV (fully conv.) | CIFAR-10 |

Launch with:

```bash
jupyter notebook
```

---

## License

PathCond is released under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](LICENSE) license.

---

## Citation

If you use PathCond in your research, please cite:

```bibtex
@article{lebeurrier2026pathcond,
  title   = {Path-conditioned training: a principled way to rescale {ReLU} neural networks},
  author  = {Lebeurrier, Arthur and Vayer, Titouan and Gribonval, R{\'e}mi},
  year    = {2026},
  url     = {https://arxiv.org/abs/2602.19799}
}
```
