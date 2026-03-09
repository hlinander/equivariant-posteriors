import numpy as np
from dataclasses import dataclass
import torch
from tqdm import tqdm
from experiments.weather.data import denormalize_sample
from experiments.climate.climateset_data_hp import deserialize_dataset_statistics

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

    # Seq-to-seq case: (N, T, C, P) → (N*T, C, P) so RMSE is always over (batch, pixels)
    if all_predictions.dim() == 4:
        N, T, C, P = all_predictions.shape
        all_predictions = all_predictions.reshape(N * T, C, P)
        all_targets     = all_targets.reshape(N * T, C, P)

    # Denormalize both into physical units before computing RMSE

    squared_errors  = (all_predictions - all_targets) ** 2
    mse_per_channel = squared_errors.mean(dim=(0, 2))   # → (C,)
    rmse_per_channel = torch.sqrt(mse_per_channel)
    overall_rmse     = torch.sqrt(squared_errors.mean())

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
        overall_rmse     = torch.sqrt(squared_errors.mean())
    
    # print("==================================")
    # print(f"Computed RMSE - per channel: {rmse_per_channel}, overall: {overall_rmse}")
    # print("==================================")

    return {
        'rmse_per_channel': rmse_per_channel,
        'overall_rmse':     overall_rmse,
        'predictions':      all_predictions,
        'targets':          all_targets,
    }