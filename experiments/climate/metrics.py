import numpy as np
from dataclasses import dataclass
import torch
from tqdm import tqdm
from experiments.weather.data import denormalize_sample
from experiments.climate.climateset_data import deserialize_dataset_statistics

def rmse_climate_hp(model, dataloader, device_id):
    """
    Compute RMSE for climate model outputs.
    
    Args:
        model: Climate model
        dataloader: DataLoader with validation data
        device_id: Device to run on
        
    Returns:
        Dictionary with RMSE results per output variable
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            batch_device = {k: v.to(device_id) if isinstance(v, torch.Tensor) else v 
                          for k, v in batch.items()}
            
            # Get model output
            output = model(batch_device)
            
            # Extract predictions (B, C, N)
            predictions = output['logits_output']
            targets = batch_device['target']
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all batches
    all_predictions = torch.cat(all_predictions, dim=0)  # (N_samples, C, N_pixels)
    all_targets = torch.cat(all_targets, dim=0)  # (N_samples, C, N_pixels)
    
    # Compute RMSE per channel
    squared_errors = (all_predictions - all_targets) ** 2
    mse_per_channel = squared_errors.mean(dim=(0, 2))  # Average over samples and pixels
    rmse_per_channel = torch.sqrt(mse_per_channel)
    
    # Overall RMSE
    overall_rmse = torch.sqrt(squared_errors.mean())
    
    return {
        'rmse_per_channel': rmse_per_channel,
        'overall_rmse': overall_rmse,
        'predictions': all_predictions,
        'targets': all_targets,
    }