import torch

# Module types whose parameters (scale/shift) should not be weight-decayed:
# decaying normalization gains toward zero is a known training instability.
_NORM_MODULE_TYPES = (
    torch.nn.LayerNorm,
    torch.nn.GroupNorm,
    torch.nn.RMSNorm,
    torch.nn.modules.batchnorm._BatchNorm,
    torch.nn.modules.instancenorm._InstanceNorm,
)


def adamw_no_decay_norm_bias(model, lr, weight_decay, **kwargs):
    """AdamW with weight decay exempted for normalization-module parameters
    and biases, identified explicitly by module type and parameter name
    (not by tensor dimensionality, so unrelated 1-d parameters still decay).

    Takes the model instead of a parameter iterable (takes_model marker);
    otherwise a drop-in for torch.optim.AdamW in OptimizerConfig.
    """
    no_decay_ids = set()
    for module in model.modules():
        if isinstance(module, _NORM_MODULE_TYPES):
            no_decay_ids.update(id(p) for p in module.parameters(recurse=False))
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # endswith("bias") (not ".bias") also catches MultiheadAttention's
        # in_proj_bias / bias_k / bias_v naming.
        if id(p) in no_decay_ids or name.endswith("bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        **kwargs,
    )


adamw_no_decay_norm_bias.takes_model = True
