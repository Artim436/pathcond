# mnist-mlp

Packaged MLP (ReLU) classifier for MNIST.

## Install (dev mode)

```bash
pip install -e .'[dev']
```

## Train

```bash
mnist-train --h1 10 --h2 10 --epochs 10 --teleport-epoch 0  --nb-iter-optim-rescaling 1 --nb-iter 5
moons-multi-lr --epochs 1500 --frac 1 --nb-lr 5  --teleport-epoch 0 --nb-iter-optim-rescaling 15 --nb-iter 2
```


## Plot curves accuracy and lost MNIST boxplots Moons

```bash
mnist-plot-curves
moons-boxplots
```

## Compare G vs diag(G)

```bash
compare-g-diag-g --h1 10 --h2 10
```

## Test

```bash
pytest
```
