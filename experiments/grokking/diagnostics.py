import io
import gzip
import torch
import math


def compressed_size(state_dict, method="gzip"):
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw = buf.getvalue()
    if method == "gzip":
        compressed = gzip.compress(raw)
    else:
        raise ValueError(f"Unknown compression method: {method}")
    return len(compressed)


def weight_statistics(model, threshold=1e-4):
    all_params = torch.cat([p.data.flatten() for p in model.parameters()])
    n = all_params.numel()
    return dict(
        l1_norm=all_params.abs().sum().item(),
        l2_norm=all_params.norm(2).item(),
        near_zero_fraction=(all_params.abs() < threshold).float().mean().item(),
        n_params=n,
    )


def effective_rank(singular_values):
    sv = singular_values[singular_values > 0]
    if len(sv) == 0:
        return 0.0
    p = sv / sv.sum()
    entropy = -(p * p.log()).sum()
    return math.exp(entropy.item())


def layer_svd_summary(model):
    results = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            W = module.weight.data
            sv = torch.linalg.svdvals(W)
            results.append(
                dict(
                    name=name,
                    shape=list(W.shape),
                    effective_rank=effective_rank(sv),
                    top_sv=sv[:5].tolist(),
                    condition_number=(sv[0] / sv[-1]).item() if sv[-1] > 0 else float("inf"),
                )
            )
    return results
