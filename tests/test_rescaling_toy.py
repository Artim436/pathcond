import copy
import torch
import torch.nn as nn
import pytest

from pathcond.models import toy_MLP
from pathcond.rescaling_polyn import (
    apply_neuron_rescaling_mlp,
    compute_G_matrix,
    set_weights_for_path_norm,
    reset_model,
    compute_diag_G,
)

torch.manual_seed(0)


def make_model(sizes=(32, 16)):
    # Modèle plus petit pour des tests rapides
    m = toy_MLP(d_input=sizes[0], d_hidden1=sizes[1])
    # Met le modèle en eval pour des sorties stables et évite dropout
    m.eval()
    return m


def rand_input(batch=3):
    # Entrée aléatoire format MNIST: (B, 1, 28, 28)
    return torch.randn(batch, 1, 28, 28)


def clone_tensors(module):
    # Copie profonde des tenseurs de poids/biais pour comparaison
    out = {}
    for name, p in module.named_parameters():
        out[name] = p.detach().clone()
    return out


def clone_params(mod: nn.Module):
    return {k: v.detach().clone() for k, v in mod.named_parameters()}


def test_hidden_layer_rescaling_preserves_output_on_next_layer_compensation(sizes=(32, 16)):
    torch.manual_seed(0)
    m = toy_MLP(d_input=sizes[0], d_hidden1=sizes[1]).eval()
    x = torch.randn(3, 1, sizes[0])

    with torch.no_grad():
        y_ref = m(x)

    # layer_idx=0 -> couche Linear à index réel 0 (première cachée)
    layer_idx = 0
    neuron_idx = torch.randint(0, sizes[1], (1,)).item()
    lam = 1.7  # positif pour l'invariance ReLU
    m2 = apply_neuron_rescaling_mlp(m, layer_idx=layer_idx, neuron_idx=neuron_idx, lamda=lam)
    with torch.no_grad():
        y_new = m2(x)
        # Invariance (à tolérance numérique près)
        assert torch.allclose(y_ref, y_new, rtol=1e-6, atol=1e-7)


def test_weights_and_compensation_are_updated_correctly_for_hidden_layer(sizes=(32, 16)):
    torch.manual_seed(0)
    m = toy_MLP(d_input=sizes[0], d_hidden1=sizes[1]).eval()
    before = clone_params(m)

    neuron_idx = 4
    lam = 2.0
    m2 = apply_neuron_rescaling_mlp(m, layer_idx=0, neuron_idx=neuron_idx, lamda=lam)
    after = clone_params(m2)

    # Couche 0 (index réel 0): la ligne neuron_idx du poids doit être * lam
    w0_before = before["model.0.weight"][neuron_idx]
    w0_after = after["model.0.weight"][neuron_idx]
    assert torch.allclose(w0_after, w0_before * lam)

    # Biais de la couche 0 multiplié par lam (si présent)
    b0_before = before["model.0.bias"][neuron_idx]
    b0_after = after["model.0.bias"][neuron_idx]
    assert torch.allclose(b0_after, b0_before * lam)

    # Couche suivante (index réel 3): la colonne correspondante divisée par lam
    w3_before_col = before["model.2.weight"][:, neuron_idx]
    w3_after_col = after["model.2.weight"][:, neuron_idx]
    assert torch.allclose(w3_after_col, w3_before_col / lam)

    # Un autre neurone non ciblé est inchangé
    other = (neuron_idx + 1) % m.model[0].out_features
    assert torch.allclose(after["model.0.weight"][other], before["model.0.weight"][other])
    assert torch.allclose(after["model.2.weight"][:, other], before["model.2.weight"][:, other])


def test_original_model_is_not_modified(sizes=(32, 16)):
    torch.manual_seed(0)
    m = toy_MLP(d_input=sizes[0], d_hidden1=sizes[1]).eval()
    before = clone_params(m)

    _ = apply_neuron_rescaling_mlp(m, layer_idx=0, neuron_idx=2, lamda=1.3)
    after = clone_params(m)

    # Le modèle original ne doit pas changer (deepcopy dans la fonction)
    for k in before:
        assert torch.allclose(before[k], after[k]), f"Paramètre original modifié: {k}"


def test_layer_idx_bounds_raise(sizes=(32, 16)):
    m = toy_MLP(d_input=sizes[0], d_hidden1=sizes[1]).eval()
    # linear_indices = [0,3,6] -> len = 3 -> layer_idx doit être 0,1,2
    with pytest.raises(ValueError):
        _ = apply_neuron_rescaling_mlp(m, layer_idx=3, neuron_idx=0, lamda=1.0)


def test_reset_weights(sizes=(2, 2)):
    torch.manual_seed(0)
    m = toy_MLP(d_input=sizes[0], d_hidden1=sizes[1]).eval()
    orig_w = set_weights_for_path_norm(m, exponent=2, provide_original_weights=True)
    reset_model(m, orig_w)
    assert all(torch.allclose(p1, p2) for p1, p2 in zip(m.parameters(), orig_w.values()))


def test_set_weights_for_path_norm(sizes=(2, 2)):
    """ Test that we put all the weights to the power 2"""
    torch.manual_seed(0)
    m = toy_MLP(d_input=sizes[0], d_hidden1=sizes[1]).eval()
    orig_w = set_weights_for_path_norm(m, exponent=2, provide_original_weights=True)
    for (name, p), (name_orig, p_orig) in zip(m.named_parameters(), orig_w.items()):
        assert name == name_orig
        assert torch.allclose(p, p_orig ** 2)


def test_diag_G(sizes=(2, 2)):
    torch.manual_seed(0)
    m = toy_MLP(d_input=sizes[0], d_hidden1=sizes[1]).eval()

    G_mat = compute_G_matrix(m)
    diag_G_mat = torch.diag(G_mat)

    diag_G_func = compute_diag_G(m)

    assert torch.allclose(diag_G_mat, diag_G_func, rtol=1e-6, atol=1e-7)
