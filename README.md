# Path-conditioned training
Pathcond is a principled and fast way to rescale ReLU neural netowrk at initialization. We show on several examples availale in this repo that rescale a MLP or a conv net at init  accelerates training and does not degrade generalization performance.

This repository contains the implementation of Pathcond as detailed in our paper [Path-conditioned training: a principled way to rescale ReLU neural networks]. The library is easy to use.


## Dependencies
todo
`
pip install -r requirements.txt
`
sachant que beaucoup de packages utiles pour suivre les expe ou plot mais en pratqiue besoin que de torch 



## How to use Pathcond

The training procedure consists in performing one pathcond rescaling at init.
```python
from pathcond.pathcond import rescaling_path_cond


# defining model and optimizer
model = ...
criterion = ...
optimizer = ...

rescaling_path_cond(model_pathcond)

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
Some precisions about the usage of PathCond (for details, see [our paper]()):

process:
diag_G = compute_diag_G(model) in pathcond/network_to_optim.py
cost of 1 backward
where
G = partial Phi^top partial Phi


compute the number of hidden neurons
compute the matrix B (size of nb of parmeters times nb of hidden neurons) in a efficient way, sparse matrix with 2p non zeros coefficient swhere p is the nb of params of the network 
compute_B_mlp in pathcond/network_to_optim.py
different function for cnn or resnet or unet because need of finding for each hidden neurons shich parmaerws enters and which parmeetr go out



update the duale variable v = Bu thanks to the algorihm pathcond 
"update_z_polynomial" in the file pathcond/rescaling_polyn.py


reweight the model ie applying the optimal rescaling accroding to our crieria
reweight_model_in_place in pathcond/rescaling_polyn.py






## Results
You can reproduce the results of our paper by running the following commands:

todo

## License
Pathcond is released under Creative Commons Attribution 4.0 International (CC BY 4.0) license, as found in the LICENSE file.

## Bibliography

todo