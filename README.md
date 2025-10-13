# mnist-mlp

Packaged MLP (ReLU) classifier for MNIST.

## Install (dev mode)

```bash
pip install -e .'[dev']
```

## Train

```bash
mnist-train --h1 10 --h2 10 --epochs 10 --teleport-epoch 0  --nb-iter-optim-rescaling 1 --nb-iter 5
moons-multi-lr --epochs 100000 --frac 1 --nb-lr 10  --teleport-epoch 0 --nb-iter-optim-rescaling 50 --nb-iter 5
ts-multi-lr --epochs 100 --frac 1 --nb-lr 10  --teleport-epoch 0 --nb-iter-optim-rescaling 50 --nb-iter 10
```


## Plot curves accuracy and lost MNIST boxplots Moons

```bash
mnist-plot-curves
moons-boxplots
ts-boxplots
```

## Compare G vs diag(G)

```bash
compare-g-diag-g --h1 10 --h2 10
```

## Test

```bash
pytest
```
