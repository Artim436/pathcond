# Path-conditioned training
Pathcond is a principled and fast way to rescale ReLU neural netowrk at initialization. We show on several examples availale in this repo that rescale a MLP or a conv net at init  accelerates training and does not degrade generalization performance.

This repository contains the implementation of Pathcond as detailed in our paper [Path-conditioned training: a principled way to rescale ReLU neural networks]. The library is easy to use.


## Dependencies
todo
`
pip install -r requirements.txt
`

## How to use Pathcond

The training procedure consists in performing one pathcond rescaling at init.
```python
from enorm import ENorm


# defining model and optimizer
model = ...
criterion = ...
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9 weight_decay=1e-4)

# instantiating ENorm (here with asymmetric scaling coefficient c=1)
enorm = ENorm(model.named_parameters(), optimizer, c=1)

# training loop
for batch, target in train_loader:
  # forward pass
  output = model(input)
  loss = criterion(output, target)

  # backward pass
  optimizer.zero_grad()
  loss.backward()

  # SGD and ENorm steps
  optimizer.step()
  enorm.step()
```
Some precisions about the usage of ENorm (for details, see [our paper]()):

## Results
You can reproduce the results of our paper by running the following commands:

todo

## License
Pathcond is released under Creative Commons Attribution 4.0 International (CC BY 4.0) license, as found in the LICENSE file.

## Bibliography

todo