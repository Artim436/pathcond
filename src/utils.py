import torch
from torch import nn
from typing import Optional, Callable, Dict
from pathlib import Path


def _ensure_outdir(outdir: str) -> Path:
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p

@torch.no_grad()
def rebuild_optimizer_with_state_from_old(
    old_model: nn.Module,
    new_model: nn.Module,
    old_opt: torch.optim.Optimizer,
    *,
    opt_ctor: Optional[Callable[..., torch.optim.Optimizer]] = None,
    override_hparams: Optional[Dict] = None,
    allow_partial_slice: bool = False,   # True => copie coin supérieur gauche si shape a grandi
) -> torch.optim.Optimizer:
    """
    Recrée un optimizer pour new_model et transfère l'état par correspondance de noms.
    Gère Adam/AdamW (exp_avg, exp_avg_sq, step) et SGD(momentum) (momentum_buffer), etc.

    - opt_ctor: constructeur de l'optimizer (par défaut, même classe que old_opt)
    - override_hparams: pour changer lr/betas/eps/weight_decay… si besoin
    - allow_partial_slice: si True et shapes différentes, on copie par slice là où possible
    """
    if opt_ctor is None:
        opt_ctor = old_opt.__class__

    # 1) Hyperparams (lr, betas, eps, weight_decay, etc.)
    hp = dict(old_opt.defaults)
    if override_hparams:
        hp.update(override_hparams)

    # 2) Créer le nouvel optimizer
    new_opt = opt_ctor(new_model.parameters(), **hp)

    # 3) Index param par nom
    old_named = dict(old_model.named_parameters())
    new_named = dict(new_model.named_parameters())

    # 4) Copier état
    old_state = old_opt.state
    new_state = new_opt.state

    def _compatible_copy(t_old: torch.Tensor, t_new: torch.Tensor) -> torch.Tensor:
        """Copie t_old -> shape de t_new (identique, ou slice si allow_partial_slice=True)."""
        if t_old.shape == t_new.shape:
            return t_old.detach().clone().to(t_new.device)
        if allow_partial_slice:
            # Copie sur l'intersection des dimensions, le reste reste aux init de new_state
            out = torch.zeros_like(t_new, device=t_new.device)
            slices = tuple(slice(0, min(a, b)) for a, b in zip(t_old.shape, t_new.shape))
            out[slices] = t_old[slices].to(t_new.device)
            return out
        # shapes différentes non autorisées
        raise RuntimeError("Incompatible shapes without allow_partial_slice")

    copied, skipped = 0, 0
    for name, p_new in new_named.items():
        p_old = old_named.get(name)
        if p_old is None or p_old not in old_state:
            skipped += 1
            continue

        st_old = old_state[p_old]
        st_new = {}

        ok = True
        for k, v in st_old.items():
            if torch.is_tensor(v):
                try:
                    st_new[k] = _compatible_copy(v, torch.zeros_like(p_new) if v.shape == p_old.shape else v)
                    # Remap propre: si c'est un buffer de même shape que p_old, on le transpose vers p_new
                    if v.shape == p_old.shape:
                        st_new[k] = _compatible_copy(v, p_new)  # même shape que param => miroir
                    else:
                        # ex. buffers vectoriels (RMSprop: square_avg) / Adam: exp_avg, exp_avg_sq
                        # si même shape que p_old -> déjà géré; sinon on garde shape identique si allow_partial_slice True
                        pass
                except RuntimeError:
                    ok = False
                    break
            else:
                # ex. step (int), amsgrad flags, etc.
                st_new[k] = v

        if ok:
            new_state[p_new] = st_new
            copied += 1
        else:
            skipped += 1

    # 5) Copier les param_groups (lr, weight_decay, etc.)
    for g_old, g_new in zip(old_opt.param_groups, new_opt.param_groups):
        for k, v in g_old.items():
            if k != "params":
                g_new[k] = v

    print(f"Optimizer state transfer: copied={copied}, skipped={skipped}")
    return new_opt


def _param_start_offsets(model):
    """
    Retourne un dict {id(param_tensor): (start, end)} correspondant
    aux tranches [start:end) dans la vectorisation officielle PyTorch
    (parameters_to_vector), ce qui évite tout calcul d'offset fragile.
    """
    starts = {}
    offset = 0
    for p in model.parameters():
        n = p.numel()
        starts[id(p)] = (offset, offset + n)
        offset += n
    return starts, offset  # offset final == n_params