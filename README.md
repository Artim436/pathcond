# mnist-mlp

Packaged MLP (ReLU) classifier for MNIST.

## Install (dev mode)

```bash
pip install -e .'[dev']
```

## Train

```bash
mnist-train --epochs 10 --h1 512 --h2 512 --nb-lr 1 --teleport-epoch 0  --nb-iter-optim-rescaling 15 --nb-iter 1 --frac 1
moons-multi-lr --epochs 1000 --frac 1 --nb-lr 10  --teleport-epoch 0 --nb-iter-optim-rescaling 15 --nb-iter 3
ts-multi-lr --epochs 1000 --frac 1 --nb-init 5  --teleport-epoch 0 --nb-iter-optim-rescaling 15 --nb-iter 1
resnet-mnist-train --epochs 10 --nb-lr 10 --teleport-epoch 0 --nb-iter-optim-rescaling 1 --nb-iter 1 --frac 1
resnet-cifar-train --epochs 10 --nb-lr 10 --teleport-epoch 0 --nb-iter-optim-rescaling 1 --nb-iter 1 --frac 1
python3 expes/u_net_denoising.py --epochs 1 --nb-lr 1 --nb-iter 1 --frac 0.1
```


## Plot curves accuracy and lost MNIST boxplots Moons

```bash
mnist-plot-curves
moons-boxplots
ts-boxplots
plot-mnist-resnet
```

## Compare G vs diag(G)

```bash
compare-g-diag-g --h1 10 --h2 10
```

## Test

```bash
pytest
```
