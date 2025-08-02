import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import json  
import argparse
import shutil
from tqdm import tqdm
import pandas as pd
import datetime
import xarray as xr

from utils import *
from loss import *
from sampler import *

# Data and result directories
data_directory = '/root/autodl-tmp/train_sourth_ocean_uv'  # Changed to ocean data directory
result_directory = './ocean_models'  # Changed to ocean models directory

# Ocean-specific variables
variable_names = ['uo', 'vo']  # Ocean current components (u, v)
num_variables = 2  # Only 2 variables for ocean data
num_static_fields = 0  # No static fields for ocean (can add bathymetry later if needed)
max_horizon = 30  # Maximum time horizon in days (reduced for ocean forecasting)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Run ocean model with configuration from JSON file.')
parser.add_argument('config_path', type=str, help='Path to JSON configuration file.')
args = parser.parse_args()

def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config

config_path = args.config_path
config = load_config(config_path)

# Load config
name            = config['name']
spacing         = config['spacing']
delta_t         = config['delta_t']
t_max           = config['t_max']
t_min           = delta_t
batch_size      = config['batch_size']

num_epochs      = config['num_epochs']
weight_decay    = config['weight_decay']
learning_rate   = config['learning_rate']

filters         = config['filters']
conditioning_times   = config['conditioning_times']
model_choice    = config['model']

# Ocean-specific configuration
ocean_data_path = config.get('ocean_data_path', f'{data_directory}/cmems_mod_glo_phy_my_0.083deg_P1D-m_uo-vo_105.00E-121.25E_0.00N-23.92N_0.49m_2018-01-01-2018-12-31.nc')
depth_level = config.get('depth_level', 0)  # Surface level by default

# Copy config
result_path = Path(f'{result_directory}/{name}')
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, result_path / "config.json")

def compute_ocean_residual_stds(data_path, variables, t_max, device):
    """
    Compute residual standard deviations for ocean variables.
    This replaces the precomputed atmospheric residual stds.
    """
    # Load a sample of ocean data to compute statistics
    if isinstance(data_path, str) and data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        residual_stds = []
        
        for var_name in variables:
            if var_name in ds.data_vars:
                data = ds[var_name].values
                # Compute temporal differences (residuals)
                residuals = np.diff(data, axis=0)
                # Compute std over space and time for each lead time
                std_values = []
                for t in range(1, min(t_max + 1, residuals.shape[0] + 1)):
                    if t <= residuals.shape[0]:
                        std_val = np.std(residuals[:t])
                    else:
                        std_val = np.std(residuals)
                    std_values.append(std_val)
                
                # Pad if necessary
                while len(std_values) < t_max:
                    std_values.append(std_values[-1])
                    
                residual_stds.append(torch.tensor(std_values[:t_max], dtype=torch.float32))
            else:
                # Default values if variable not found
                residual_stds.append(torch.ones(t_max, dtype=torch.float32))
        
        residual_stds = torch.stack(residual_stds, dim=1).to(device)
        return residual_stds
    else:
        # Default fallback
        return torch.ones(t_max, len(variables), device=device)

def compute_ocean_normalization(data_path, variables, depth_level=0):
    """
    Compute normalization factors for ocean data.
    """
    if isinstance(data_path, str) and data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        mean_data = []
        std_data = []
        
        for var_name in variables:
            if var_name in ds.data_vars:
                data = ds[var_name].values
                if depth_level is not None and len(data.shape) > 3:
                    data = data[:, depth_level, :, :]
                
                mean_val = np.mean(data)
                std_val = np.std(data)
                mean_data.append(mean_val)
                std_data.append(std_val)
            else:
                mean_data.append(0.0)
                std_data.append(1.0)
        
        mean_data = torch.tensor(mean_data, dtype=torch.float32)
        std_data = torch.tensor(std_data, dtype=torch.float32)
        norm_factors = np.stack([mean_data, std_data], axis=0)
        
        return norm_factors, mean_data, std_data
    else:
        # Default fallback
        mean_data = torch.zeros(len(variables), dtype=torch.float32)
        std_data = torch.ones(len(variables), dtype=torch.float32)
        norm_factors = np.stack([mean_data, std_data], axis=0)
        return norm_factors, mean_data, std_data

# Compute ocean-specific statistics
residual_stds = compute_ocean_residual_stds(ocean_data_path, variable_names, t_max, device)

# Compute normalization factors for ocean data
norm_factors, mean_data, std_data = compute_ocean_normalization(ocean_data_path, variable_names, depth_level)
mean_data = mean_data.to(device)
std_data = std_data.to(device)

def renormalize(x, mean_ar=mean_data, std_ar=std_data):
    x = x * std_ar[None, :, None, None] + mean_ar[None, :, None, None]
    return x

# Ocean data time configuration
def get_ocean_sample_counts(data_path):
    """
    Get sample counts for ocean data based on available time steps.
    """
    if isinstance(data_path, str) and data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        n_samples = len(ds.time)
        
        # Split: 70% train, 20% val, 10% test
        n_train = int(0.95 * n_samples)
        n_val = int(0.05 * n_samples)
        
        ds.close()
        return n_samples, n_train, n_val
    else:
        # Default fallback
        return 365, 255, 73

n_samples, n_train, n_val = get_ocean_sample_counts(ocean_data_path)

# Get ocean grid dimensions
def get_ocean_dimensions(data_path, variables, depth_level=0):
    """
    Get the spatial dimensions of ocean data.
    """
    if isinstance(data_path, str) and data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        # Assuming lat, lon dimensions exist
        n_lat = len(ds.latitude) if 'latitude' in ds.dims else 32
        n_lon = len(ds.longitude) if 'longitude' in ds.dims else 64
        
        # Count actual variables considering depth
        n_vars = 0
        for var in variables:
            if var in ds.data_vars:
                var_data = ds[var]
                if depth_level is not None and 'depth' in var_data.dims:
                    n_vars += 1  # One level per variable
                else:
                    n_vars += var_data.shape[1] if len(var_data.shape) > 3 else 1
        
        ds.close()
        return n_vars, n_lat, n_lon
    else:
        return len(variables), 32, 64

num_variables, n_lat, n_lon = get_ocean_dimensions(ocean_data_path, variable_names, depth_level)

# Dataset configuration for ocean data
kwargs = {
    'dataset_path':     data_directory,
    # 'sample_counts':    (n_samples, n_train, n_val),
    'dimensions':       (num_variables, n_lat, n_lon),
    'max_horizon':      max_horizon,
    'norm_factors':     None,  # Will compute automatically
    'device':           device,
    'spacing':          spacing,
    'dtype':            'float32',
    'conditioning_times':    conditioning_times,
    'lead_time_range':  [t_min, t_max, delta_t],
    'variables':        variable_names,
    'depth_level':      depth_level,
    'random_lead_time': True,
}

# Define the batch samplers
update_t_per_batch = get_uniform_t_dist_fn(t_min=delta_t, t_max=t_max, delta_t=delta_t)

train_time_dataset = OceanDatasetERA5Style(lead_time=t_max, dataset_mode='train', 
                                           time_split_mode='date',
                                           train_date_range=('2000-01-01', '2021-12-31'), 
                                           val_date_range=('2022-01-01', '2022-12-31'),
                                           test_date_range=('2023-01-01', '2023-12-31'),**kwargs)
train_batch_sampler = DynamicKBatchSampler(train_time_dataset, batch_size=batch_size, drop_last=True, t_update_callback=update_t_per_batch, shuffle=True)
train_time_loader = DataLoader(train_time_dataset, batch_sampler=train_batch_sampler)

val_time_dataset = OceanDatasetERA5Style(lead_time=t_max, dataset_mode='val',
                                         time_split_mode='date',
                                         train_date_range=('2000-01-01', '2021-12-31'),
                                         val_date_range=('2022-01-01', '2022-12-31'),
                                         test_date_range=('2023-01-01', '2023-12-31'),**kwargs)
val_batch_sampler = DynamicKBatchSampler(val_time_dataset, batch_size=batch_size, drop_last=True, t_update_callback=update_t_per_batch, shuffle=True)
val_time_loader = DataLoader(val_time_dataset, batch_sampler=val_batch_sampler)

# Input dimensions for ocean model
input_times = (1 + len(conditioning_times))*num_variables + num_static_fields

if 'autoregressive' in model_choice:
    time_emb = 0
elif 'continuous' in model_choice:
    time_emb = 1
else:
    raise ValueError(f"Model choice {model_choice} not recognized.")

# Define the model with ocean-specific parameters
model = EDMPrecond(filters=filters, img_channels=input_times, out_channels=num_variables, 
                   img_resolution=max(n_lat, n_lon), time_emb=time_emb, 
                   sigma_data=1, sigma_min=0.02, sigma_max=88)

# Ocean-specific loss function
def create_ocean_loss_fn(lat_coords, lon_coords, device, precomputed_std):
    """
    Create a loss function suitable for ocean data.
    If WGCLoss is not suitable for ocean, this can be modified.
    """
    try:
        return WGCLoss(lat_coords, lon_coords, device, precomputed_std=precomputed_std)
    except:
        # Fallback to MSE loss if WGCLoss doesn't work for ocean data
        print("Warning: Using MSE loss instead of WGCLoss for ocean data")
        return nn.MSELoss()

# Get lat/lon coordinates for ocean data
def get_ocean_coordinates(data_path):
    """
    Extract latitude and longitude coordinates from ocean data.
    """
    if isinstance(data_path, str) and data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        lat = ds.latitude.values if 'latitude' in ds.coords else np.linspace(-90, 90, 32)
        lon = ds.longitude.values if 'longitude' in ds.coords else np.linspace(-180, 180, 64)
        ds.close()
        return lat, lon
    else:
        # Default coordinates
        return np.linspace(-90, 90, 32), np.linspace(-180, 180, 64)

lat, lon = get_ocean_coordinates(ocean_data_path)
# loss_fn = create_ocean_loss_fn(lat, lon, device, precomputed_std=residual_stds)

loss_fn = PhysicsConstrainedLoss(
    lat, lon, device, 
    weights=config.get('physics_weights', PHYSICS_CONFIG['physics_weights']),
    precomputed_std=residual_stds
)

print(f"Ocean Model: {name}", flush=True)
print(f"Model choice: {model_choice}", flush=True)
print(f"Variables: {variable_names}", flush=True)
print(f"Data shape: {num_variables} vars, {n_lat}x{n_lon} grid", flush=True)
print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)
model.to(device)

print("Lead times", kwargs['lead_time_range'], flush=True)

optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.001, end_factor=1.0, total_iters=1000)

loss_values = []
val_loss_values = []
best_val_loss = float('inf')

# Setup for logging
log_file_path = result_path / f'training_log.csv'
with open(log_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Average Training Loss', 'Validation Loss', 'Learning Rate'])

# Training loop
for epoch in range(num_epochs):
    
    # Training phase
    model.train()
    total_train_loss = 0
    for previous, current, time_label in tqdm(train_time_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]'):
        current = current.to(device)
        previous = previous.to(device)
        time_label = time_label.to(device)
        
        optimizer.zero_grad()   
        
        # # Scale time label for ocean time scales
        # if isinstance(loss_fn, nn.MSELoss):
        #     # Simple MSE loss
        #     output = model(previous, time_label/max_horizon)
        #     loss = loss_fn(output, current)
        # else:
        #     # WGC Loss or similar
        #     loss = loss_fn(model, current, previous, time_label/max_horizon)
        
        if isinstance(loss_fn, PhysicsConstrainedLoss):
            # Physics constrained loss
            loss, individual_losses = loss_fn(model, current, previous, time_label/max_horizon)
            
            # 可选：每100步打印一次物理损失组件
            if len(loss_values) % 100 == 0:
                loss_components = {k: v.item() if torch.is_tensor(v) else v for k, v in individual_losses.items()}
                print(f"Loss components: {loss_components}")
                
        elif isinstance(loss_fn, nn.MSELoss):
            output = model(previous, time_label/max_horizon)
            loss = loss_fn(output, current)
        else:
            # 原始WGCLoss
            loss, _ = loss_fn(model, current, previous, time_label/max_horizon)



        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()        
        
        # Warmup only for first 1000 steps
        if epoch * len(train_time_loader) + len(train_time_loader) < 1000:
            warmup_scheduler.step()

    avg_train_loss = total_train_loss / len(train_time_loader)
    
    # Validation phase
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for previous, current, time_label in tqdm(val_time_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
            current = current.to(device)
            previous = previous.to(device)
            time_label = time_label.to(device)
            
            if isinstance(loss_fn, nn.MSELoss):
                output = model(previous, time_label/max_horizon)
                loss = loss_fn(output, current)
            else:
                loss, _ = loss_fn(model, current, previous, time_label/max_horizon)
            total_val_loss += loss.item()
                
        avg_val_loss = total_val_loss / len(val_time_loader)

    # Checkpointing
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': avg_val_loss,
            'config': config
        }, result_path/f'best_model.pth')
        
    scheduler.step()
    
    loss_values.append([avg_train_loss])
    val_loss_values.append(avg_val_loss)
    
    current_lr = optimizer.param_groups[0]['lr']
    with open(log_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, avg_train_loss, avg_val_loss, current_lr])
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr:.2e}', flush=True)

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': num_epochs,
    'loss': avg_val_loss,
    'config': config,
    'normalization_factors': (mean_data.cpu(), std_data.cpu())
}, result_path/f'final_model.pth')

print(f"Training completed. Best validation loss: {best_val_loss:.6f}")
print(f"Models saved to: {result_path}")