import torch
from pathcond.rescaling_polyn import reweight_model_inplace, optimize_rescaling_polynomial


def rescaling_path_cond(model, nb_iter_optim_rescaling=10):
    BZ, Z = optimize_rescaling_polynomial(model, n_iter=nb_iter_optim_rescaling, tol=1e-2, resnet=False)
    reweight_model_inplace(model, BZ)
    return 0