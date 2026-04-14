import numpy as np
#from dataclasses import dataclass
import torch
from tqdm import tqdm
#from experiments.weather.data import denormalize_sample
#from experiments.climate.data.climateset_data_hp import deserialize_dataset_statistics

# def rmse_climate_hp(model, dataloader, device_id):
#     """
#     Compute RMSE for climate model outputs.
#     """
#     model.eval()
    
#     all_predictions = []
#     all_targets = []
    
#     with torch.no_grad():
#         for batch in dataloader:
#             # Move batch to device
#             batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v 
#                           for k, v in batch.items()}
            
#             # Get model output
#             output = model(batch_device)
#             predictions = output['logits_output'] # Extract predictions (B, C, N)
#             targets = batch_device['target']
            
#             all_predictions.append(predictions.cpu())
#             all_targets.append(targets.cpu())
    
#     # Concatenate all batches
#     all_predictions = torch.cat(all_predictions, dim=0)  # (N_samples, C, N_pixels)
#     all_targets = torch.cat(all_targets, dim=0)  # (N_samples, C, N_pixels)
    
#     # Compute RMSE per channel
#     squared_errors = (all_predictions - all_targets) ** 2
#     mse_per_channel = squared_errors.mean(dim=(0, 2))  # Average over samples and pixels
#     rmse_per_channel = torch.sqrt(mse_per_channel)
    
#     # Overall RMSE
#     overall_rmse = torch.sqrt(squared_errors.mean())
    
#     return {
#         'rmse_per_channel': rmse_per_channel,
#         'overall_rmse': overall_rmse,
#         'predictions': all_predictions,
#         'targets': all_targets,
#     }
def LLweighted_RMSE_Climax(
    preds: np.ndarray, y: np.ndarray, deg2rad: bool = True, mask=None
):
    """
    Latitude weighted root mean squared error taken from ClimaX.
    Allows to weight the  by the cosine of the latitude to account for gridding differences at equator vs. poles.
    Applied per variable.
    If given a mask, normalized by sum of that.
    """

    # lattitude weights
    lat_size = y.shape[-1]
    lats = np.linspace(-90, 90, lat_size)
    if deg2rad:
        weights = np.cos((np.pi * lats) / 180)
    else:
        weights = np.cos(lats)

    # they normalize the weights first
    weights = weights / weights.mean()

    if mask is not None:
        error = (((preds - y) ** 2) * weights * mask).sum() / mask.sum()
    else:
        error = (((preds - y) ** 2) * weights).mean()

    error = np.sqrt(error)

    return error

def rmse_climate_hp(model, dataloader, device_id, output_stats, denormalize=False):
    """
    Compute RMSE for climate model outputs in original (physical) units.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            
                        # Quick fix: remove sequence dimension if present
            if batch_device['input'].dim() == 4:
                batch_device['input'] = batch_device['input'].squeeze(1)
                batch_device['target'] = batch_device['target'].squeeze(1)
            
            output = model(batch_device)
            predictions = output['logits_output']
            targets = batch_device['target']
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)  # (N, C, P) or (N, T, C, P)
    all_targets     = torch.cat(all_targets,     dim=0)
    if all_predictions.dim() == 3 and all_targets.dim() == 2:
        all_targets = all_targets.unsqueeze(1)  # (N, P) → (N, 1, P)

    # Seq-to-seq case: (N, T, C, P) → (N*T, C, P) so RMSE is always over (batch, pixels)
    if all_predictions.dim() == 4:
        N, T, C, P = all_predictions.shape
        all_predictions = all_predictions.reshape(N * T, C, P)
        all_targets     = all_targets.reshape(N * T, C, P)

    # Denormalize both into physical units before computing RMSE
    squared_errors  = (all_predictions - all_targets) ** 2
    if all_targets.dim() == 3:
        mse_per_channel = squared_errors.mean(dim=(0, 2))  # → (C,)
    elif all_targets.dim() == 2:
        mse_per_channel = squared_errors.mean(dim=(0, 1))   # → (C,)
    rmse_per_channel = torch.sqrt(mse_per_channel)
    #overall_rmse     = torch.sqrt(squared_errors.mean())
    overall_rmse     = rmse_per_channel.mean()  # mean of per-channel RMSEs, matches ClimateSet's approach
    
    # print("==================================")
    # print(f"Computed RMSE - per channel before possible normalization: {rmse_per_channel}, overall: {overall_rmse}")
    # print("==================================")


    # print("---- DEBUG ----")
    # print("Pred mean per channel (norm):", all_predictions.mean(dim=(0,2)))
    # print("Pred std per channel (norm):",  all_predictions.std(dim=(0,2)))
    # print("Target mean per channel (norm):", all_targets.mean(dim=(0,2)))
    # print("Target std per channel (norm):",  all_targets.std(dim=(0,2)))

    
    if denormalize:
        mean = torch.tensor(output_stats['mean'], dtype=torch.float64)  # (1, C, 1)
        std  = torch.tensor(output_stats['std'],  dtype=torch.float64)

        all_predictions = all_predictions * std + mean
        all_targets     = all_targets     * std + mean

        squared_errors   = (all_predictions - all_targets) ** 2
        mse_per_channel  = squared_errors.mean(dim=(0, 2))  # → (C,)
        rmse_per_channel = torch.sqrt(mse_per_channel)
        overall_rmse     = rmse_per_channel.mean() 
        # overall_rmse     = torch.sqrt(squared_errors.mean())
    
    # print("==================================")
    # print(f"Computed RMSE - per channel: {rmse_per_channel}, overall: {overall_rmse}")
    # print("==================================")
    
    return {
        'rmse_per_channel': rmse_per_channel,
        'overall_rmse':     overall_rmse,
        'predictions':      all_predictions,
        'targets':          all_targets,
    }

def rmse_climate_nohp(model, dataloader, device_id, output_stats, denormalize=False):
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}

            output = model(batch_device)
            predictions = output['logits_output']
            targets = batch_device['target']

            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())

    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    print("Concatenated predictions and targets. Shapes:", all_predictions.shape, all_targets.shape)
    
    # Seq-to-seq case: (N, T, C, lat, lon) → (N*T, C, lat, lon)
    if all_predictions.dim() == 5:
        N, T, C, H, W = all_predictions.shape
        all_predictions = all_predictions.reshape(N * T, C, H, W)
        all_targets = all_targets.reshape(N * T, C, H, W)

    if denormalize and output_stats is not None:
        mean = torch.tensor(output_stats['mean'], dtype=torch.float32)
        std = torch.tensor(output_stats['std'], dtype=torch.float32)
        all_predictions = all_predictions * std + mean
        all_targets = all_targets * std + mean

    # Convert after reshape + denorm
    preds_np = all_predictions.numpy()
    tgts_np  = all_targets.numpy()

    rmse_per_channel = []
    for c in range(all_predictions.shape[1]):
        val = LLweighted_RMSE_Climax(preds_np[:, c, :, :], tgts_np[:, c, :, :])
        rmse_per_channel.append(val)

    overall_rmse = np.mean(rmse_per_channel)  # mean of per-channel lat-weighted RMSEs, matches ClimateSet

    return {
        'rmse_per_channel': rmse_per_channel,
        'overall_rmse':     overall_rmse,
        'predictions':      all_predictions,
        'targets':          all_targets,
    }

# def rmse_climate_nohp(model, dataloader, device_id, output_stats, denormalize=False):
#     """
#     Compute latitude-weighted RMSE for climate model outputs on a regular lat/lon grid.

#     Tensors are expected to have shape (N, C, lat, lon) or (N, T, C, lat, lon) for
#     seq-to-seq models.  Latitude is assumed to be the second-to-last spatial dimension
#     (dim=-2), consistent with xarray's standard (time, lat, lon) ordering.

#     Weight for each grid cell: cos(latitude), normalised so the mean weight equals 1.
#     This accounts for the shrinking area of grid cells toward the poles.
#     """
#     model.eval()

#     all_predictions = []
#     all_targets = []

#     with torch.no_grad():
#         for batch in dataloader:
#             batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v
#                             for k, v in batch.items()}

#             output = model(batch_device)
#             predictions = output['logits_output']
#             targets = batch_device['target']

#             all_predictions.append(predictions.cpu())
#             all_targets.append(targets.cpu())

#     all_predictions = torch.cat(all_predictions, dim=0)
#     all_targets = torch.cat(all_targets, dim=0)

    
#     # ── ClimaX comparison ────────────────────────────────────────────────────────
#     preds_np = all_predictions.numpy()
#     tgts_np  = all_targets.numpy()

#     # Seq-to-seq case: (N, T, C, lat, lon) → (N*T, C, lat, lon)
#     if all_predictions.dim() == 5:
#         N, T, C, H, W = all_predictions.shape
#         all_predictions = all_predictions.reshape(N * T, C, H, W)
#         all_targets = all_targets.reshape(N * T, C, H, W)

#     if denormalize and output_stats is not None:
#         mean = torch.tensor(output_stats['mean'], dtype=torch.float32)
#         std = torch.tensor(output_stats['std'], dtype=torch.float32)
#         all_predictions = all_predictions * std + mean
#         all_targets = all_targets * std + mean

#     # Latitude-weighted RMSE: dim=-2 is the latitude axis.
#     # Latitude values are assumed to span uniformly from -90° to 90°.
#     n_lat = all_predictions.shape[-2]
#     lat_degrees = torch.linspace(-90.0, 90.0, n_lat)
#     lat_weights = torch.cos(lat_degrees * (torch.pi / 180.0))  # (n_lat,)
#     lat_weights = lat_weights / lat_weights.mean()              # normalise: mean weight = 1
#     lat_weights = lat_weights.reshape(1, 1, n_lat, 1)          # broadcast over (N, C, lat, lon)

#     squared_errors = (all_predictions - all_targets) ** 2
#     weighted_sq_errors = squared_errors * lat_weights

#     mse_per_channel = weighted_sq_errors.mean(dim=(0, 2, 3))   # (C,)
#     rmse_per_channel = torch.sqrt(mse_per_channel)
#     overall_rmse = torch.sqrt(weighted_sq_errors.mean())

#     climax_per_channel = []
#     for c in range(all_predictions.shape[1]):
#         val = LLweighted_RMSE_Climax(preds_np[:, c, :, :], tgts_np[:, c, :, :])
#         climax_per_channel.append(val)

#     print("Per-channel RMSE comparison:")
#     for c, climax_val in enumerate(climax_per_channel):
#         nohp_val = rmse_per_channel[c].item()  # computed later, move this block after rmse_per_channel
#         print(f"  ch {c}: nohp={nohp_val:.6f}  climax={climax_val:.6f}  Δ={climax_val - nohp_val:+.6f}")

#     print(f"\nOverall RMSE:")
#     climax_overall = LLweighted_RMSE_Climax(preds_np, tgts_np)
#     print(f"  nohp={overall_rmse.item():.6f}  climax={climax_overall:.6f}  Δ={climax_overall - overall_rmse.item():+.6f}")

#     exit(0)
#     return {
#         'rmse_per_channel': rmse_per_channel,
#         'overall_rmse':     overall_rmse,
#         'predictions':      all_predictions,
#         'targets':          all_targets,
#     }