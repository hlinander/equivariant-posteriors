import numpy as np
import torch
from tqdm import tqdm



def LLweighted_RMSE_Climax(
    preds: np.ndarray, y: np.ndarray, lats: np.ndarray = None,
    deg2rad: bool = True, mask=None
):
    """
    Latitude-weighted RMSE.  Input shape: (N, lat, lon).

    Data is stored as (N, lat, lon) — lat is axis -2, lon is axis -1
    (confirmed by _compute_lat_weights in climateset_data_no_hp.py which
    documents the layout as (T, C, lat, lon)).

    lats : 1-D array of latitude values in degrees, length n_lat.
           If None, falls back to np.linspace(-90, 90, n_lat) (uniform approximation).
    """
    # lat is axis -2; lon is axis -1
    lat_size = y.shape[-2]
    if lats is None:
        lats = np.linspace(-90, 90, lat_size)

    if deg2rad:
        weights = np.cos((np.pi * lats) / 180)
    else:
        weights = np.cos(lats)

    # normalize weights
    weights = weights / weights.mean()

    # reshape to (1, lat, 1) so it broadcasts correctly over (N, lat, lon)
    weights = weights[np.newaxis, :, np.newaxis]

    if mask is not None:
        error = (((preds - y) ** 2) * weights * mask).sum() / mask.sum()
    else:
        error = (((preds - y) ** 2) * weights).mean()

    return np.sqrt(error)


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

    squared_errors  = (all_predictions - all_targets) ** 2
    if all_targets.dim() == 3:
        mse_per_channel = squared_errors.mean(dim=(0, 2))  # → (C,)
    elif all_targets.dim() == 2: # single-channel case
        mse_per_channel = squared_errors.mean(dim=(0, 1))   # → (C,)
    rmse_per_channel = torch.sqrt(mse_per_channel)
    overall_rmse     = rmse_per_channel.mean()  # mean of per-channel RMSEs, matching ClimateSet's approach
    
    mean_t = torch.tensor(output_stats['mean'], dtype=torch.float64)
    std_t  = torch.tensor(output_stats['std'],  dtype=torch.float64)
    preds_denorm  = all_predictions.double() * std_t + mean_t
    tgts_denorm   = all_targets.double()     * std_t + mean_t
    sq_denorm     = (preds_denorm - tgts_denorm) ** 2
    mse_denorm    = sq_denorm.mean(dim=(0, 2)) if tgts_denorm.dim() == 3 else sq_denorm.mean(dim=(0, 1))
    rmse_per_channel_denorm = torch.sqrt(mse_denorm)

    return {
        'rmse_per_channel':       rmse_per_channel,
        'overall_rmse':           overall_rmse,
        'rmse_per_channel_denorm': rmse_per_channel_denorm,
        'predictions':            all_predictions,
        'targets':                all_targets,
    }


def rmse_climate_nohp(model, dataloader, device_id, output_stats, lats=None, denormalize=False):
    """
    lats : 1-D numpy array of latitude values (degrees) matching the lat axis of the data
           (i.e. dataset.lats).  Passed straight through to LLweighted_RMSE_Climax so that
           real lat values are used instead of a synthetic linspace approximation.
    """
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

    # Normalized lat-weighted RMSE
    preds_np = all_predictions.numpy()
    tgts_np  = all_targets.numpy()

    rmse_per_channel = []
    for c in range(all_predictions.shape[1]):
        val = LLweighted_RMSE_Climax(preds_np[:, c, :, :], tgts_np[:, c, :, :], lats=lats)
        rmse_per_channel.append(val)

    overall_rmse = np.mean(rmse_per_channel)  # mean of per-channel lat-weighted RMSEs, matches ClimateSet

    # Denormalized lat-weighted RMSE
    mean_t = torch.tensor(output_stats['mean'], dtype=torch.float32)
    std_t  = torch.tensor(output_stats['std'],  dtype=torch.float32)
    preds_denorm_np = (all_predictions * std_t + mean_t).numpy()
    tgts_denorm_np  = (all_targets     * std_t + mean_t).numpy()

    rmse_per_channel_denorm = []
    for c in range(all_predictions.shape[1]):
        val = LLweighted_RMSE_Climax(preds_denorm_np[:, c, :, :], tgts_denorm_np[:, c, :, :], lats=lats)
        rmse_per_channel_denorm.append(val)

    return {
        'rmse_per_channel':        rmse_per_channel,
        'overall_rmse':            overall_rmse,
        'rmse_per_channel_denorm': rmse_per_channel_denorm,
        'predictions':             all_predictions,
        'targets':                 all_targets,
    }