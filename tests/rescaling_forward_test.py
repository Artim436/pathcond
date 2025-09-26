# %%
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.func import functional_call


def rescale_model_params_inplace(model: nn.Module, scaling: torch.Tensor):
    """
    Rescale all parameters of the model in-place but keep gradients flowing.

    Args:
        model (nn.Module): PyTorch model
        scaling (torch.Tensor): vector of size [n_params], constant
    """
    device = next(model.parameters()).device
    scaling = scaling.to(device)

    # Flatten all parameters
    from torch.nn.utils import parameters_to_vector, vector_to_parameters
    param_vec = parameters_to_vector(model.parameters())
    assert param_vec.shape == scaling.shape, "Scaling size mismatch"

    # Multiply by scaling (creates a new differentiable tensor)
    rescaled_vec = param_vec * scaling

    # Replace each parameter with a new nn.Parameter (keeps requires_grad)
    vector_to_parameters(rescaled_vec, model.parameters())

    return model


def forward_with_rescaled(model: nn.Module, x: torch.Tensor, scaling: torch.Tensor):
    """
    Forward pass with rescaled model parameters.
    Gradients flow w.r.t model parameters; scaling is constant.

    Args:
        model (nn.Module): PyTorch model
        x (torch.Tensor): input
        scaling (torch.Tensor): scaling vector of size [n_params]

    Returns:
        torch.Tensor: model output
    """
    # Flatten model parameters
    param_vec = parameters_to_vector(model.parameters())
    scaling = scaling.to(param_vec.device)
    assert param_vec.shape == scaling.shape, "Scaling vector size mismatch"

    # Vectorized rescaling
    rescaled_vec = param_vec * scaling

    # Build dictionary of rescaled parameters for functional_call
    rescaled_params = {
        name: tensor.view_as(p).requires_grad_()
        for name, tensor, p in zip(
            [name for name, _ in model.named_parameters()],
            rescaled_vec.split([p.numel() for p in model.parameters()]),
            model.parameters()
        )
    }

    # Forward using functional_call
    return functional_call(model, rescaled_params, (x,))


# Simple model
torch.manual_seed(3)
model = nn.Linear(2, 1)

# Dummy input
x = torch.Tensor([[0.3516, -1.2998]])

to_scale = 'naive'

if to_scale == 'naive':
    # Original in-place rescaling (not differentiable w.r.t scaling)
    scaling = torch.ones(sum(p.numel() for p in model.parameters())) * 10.0
    model = rescale_model_params_inplace(model, scaling)
    loss = model(x).sum()

elif to_scale == 'properly':
    # Differentiable rescaling using functional_call
    scaling = torch.ones(sum(p.numel() for p in model.parameters())) * 10.0
    output = forward_with_rescaled(model, x, scaling)

    # Compute loss
    loss = output.sum()

else:
    # Compute loss
    loss = model(x).sum()

# Compute gradient of output w.r.t. parameters
param_vec = parameters_to_vector(model.parameters())
grad = torch.autograd.grad(loss, model.parameters(), retain_graph=True)

# Flatten gradient into a single vector for convenience
grad_vec = torch.cat([g.reshape(-1) for g in grad])

print("Gradient vector w.r.t parameters:", grad_vec)
# %%
