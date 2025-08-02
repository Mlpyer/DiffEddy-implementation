import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import csv
import math
import json
import argparse
import shutil
from tqdm import tqdm
import pandas as pd
import datetime
import gc
import zarr
import xarray as xr

from utils import *
from loss import *
from sampler import *

# Ocean-specific directories and configuration
data_directory =  '/root/autodl-tmp/test_sourth_ocean_uv'
result_directory = './ocean_results'
model_directory = './ocean_models'

# Ocean-specific variables
variable_names = ['uo', 'vo']  # Ocean current components
num_variables = 2  # Only 2 variables for ocean data
num_static_fields = 0  # No static fields for ocean (can add bathymetry later if needed)
max_horizon = 4  # Maximum time horizon in days (reduced for ocean forecasting)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser(description='Run ocean prediction with configuration from JSON file.')
parser.add_argument('config_path', type=str, help='Path to JSON configuration file.')
args = parser.parse_args()

def load_config(json_file):
    with open(json_file, 'r') as file:
        config = json.load(file)
    return config

config_path = args.config_path
config = load_config(config_path)

# Load config
name = config['name']
spacing = config['spacing']
t_direct = config['t_direct']
t_max = config['t_max']
batch_size = config['batch_size']
t_min = t_direct
t_iter = config['t_iter']
n_ens = config['n_ens']
model_path = config['model']

# Ocean-specific configuration
ocean_data_path = config.get('ocean_data_path', f'{data_directory}/cmems_mod_glo_phy_myint_0.083deg_P1D-m_uo-vo_105.00E-121.25E_0.00N-23.92N_0.49m_2023-01-01-2023-12-31.nc')
depth_level = config.get('depth_level', 0)

print(name, flush=True)
print("[t_direct, t_iter, t_max]", [t_direct, t_iter, t_max], flush=True)
print("n_ens:", n_ens, flush=True)

# Copy config
result_path = Path(f'{result_directory}/{name}')
if not result_path.exists():
    result_path.mkdir(parents=True, exist_ok=True)
shutil.copy(config_path, result_path / "config.json")

def compute_ocean_normalization(data_path, variables, depth_level=0):
    """Compute normalization factors for ocean data."""
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
        
        ds.close()
        return norm_factors, mean_data, std_data
    else:
        # Default fallback
        mean_data = torch.zeros(len(variables), dtype=torch.float32)
        std_data = torch.ones(len(variables), dtype=torch.float32)
        norm_factors = np.stack([mean_data, std_data], axis=0)
        return norm_factors, mean_data, std_data

def get_ocean_sample_counts(data_path):
    """Get sample counts for ocean data based on available time steps."""
    if isinstance(data_path, str) and data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        n_samples = len(ds.time)
        
        # Split: 70% train, 20% val, 10% test
        n_train = int(0.7 * n_samples)
        n_val = int(0.2 * n_samples)
        
        ds.close()
        return n_samples, n_train, n_val
    else:
        return 365, 255, 73

def get_ocean_dimensions(data_path, variables, depth_level=0):
    """Get the spatial dimensions of ocean data."""
    if isinstance(data_path, str) and data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        n_lat = len(ds.latitude) if 'latitude' in ds.dims else 32
        n_lon = len(ds.longitude) if 'longitude' in ds.dims else 64
        
        n_vars = 0
        for var in variables:
            if var in ds.data_vars:
                var_data = ds[var]
                if depth_level is not None and 'depth' in var_data.dims:
                    n_vars += 1
                else:
                    n_vars += var_data.shape[1] if len(var_data.shape) > 3 else 1
        
        ds.close()
        return n_vars, n_lat, n_lon
    else:
        return len(variables), 32, 64

def get_ocean_coordinates(data_path):
    """Extract latitude and longitude coordinates from ocean data."""
    if isinstance(data_path, str) and data_path.endswith('.nc'):
        ds = xr.open_dataset(data_path)
        lat = ds.latitude.values if 'latitude' in ds.coords else np.linspace(-90, 90, 32)
        lon = ds.longitude.values if 'longitude' in ds.coords else np.linspace(-180, 180, 64)
        ds.close()
        return lat, lon
    else:
        return np.linspace(-90, 90, 32), np.linspace(-180, 180, 64)

# Load ocean normalization factors
norm_factors, mean_data, std_data = compute_ocean_normalization(ocean_data_path, variable_names, depth_level)
mean_data = mean_data.to(device)
std_data = std_data.to(device)

def renormalize(x, mean_ar=mean_data, std_ar=std_data):
    x = x * std_ar[None, :, None, None] + mean_ar[None, :, None, None]
    return x

# Get ocean data information
n_samples, n_train, n_val = get_ocean_sample_counts(ocean_data_path)
num_variables, n_lat, n_lon = get_ocean_dimensions(ocean_data_path, variable_names, depth_level)
lat, lon = get_ocean_coordinates(ocean_data_path)

# Load config of trained model
train_config_path = f'{model_directory}/{model_path}/config.json'
train_config = load_config(train_config_path)

# Constants and configurations loaded from trained model JSON
filters = train_config['filters']
max_trained_lead_time = train_config['t_max']
conditioning_times = train_config['conditioning_times']
delta_t = train_config['delta_t']
model_choice = train_config['model']

# Handle filters parameter
if isinstance(filters, list) and len(filters) > 0:
    base_filters = filters[0]
elif isinstance(filters, int):
    base_filters = filters
else:
    base_filters = 64
    print(f"Warning: Invalid filters format {filters}, using default value {base_filters}")

if t_iter > max_trained_lead_time:
    print(f"The iterative lead time {t_iter} is larger than the maximum trained lead time {max_trained_lead_time}")
if t_direct < delta_t:
    print(f"The direct lead time {t_direct} is smaller than the trained dt {delta_t}")

# Ocean dataset configuration
kwargs = {
    'dataset_path': data_directory,
    'sample_counts': (n_samples, n_train, n_val),
    'dimensions': (num_variables, n_lat, n_lon),
    'max_horizon': max_horizon,
    'norm_factors': None,  # Will compute automatically
    'device': device,
    'spacing': spacing,
    'dtype': 'float32',
    'conditioning_times': conditioning_times,
    'lead_time_range': [t_min, t_max, t_direct],
    'variables': variable_names,
    'depth_level': depth_level,
    'random_lead_time': False,  # Fixed for prediction
}

input_times = (1 + len(conditioning_times)) * num_variables + num_static_fields

if 'autoregressive' in model_choice:
    time_emb = 0
elif 'continuous' in model_choice:
    time_emb = 1
else:
    raise ValueError(f"Model choice {model_choice} not recognized.")

# Define the model
model = EDMPrecond(filters=base_filters, img_channels=input_times, out_channels=num_variables, 
                   img_resolution=max(n_lat, n_lon), time_emb=time_emb, 
                   sigma_data=1, sigma_min=0.02, sigma_max=88)

# Load trained model
try:
    checkpoint = torch.load(f'{model_directory}/{model_path}/best_model.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
except Exception as e:
    print(f"Error loading model: {e}")
    print("Trying to load from final_model.pth...")
    checkpoint = torch.load(f'{model_directory}/{model_path}/final_model.pth', map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

model.to(device)

print(f"Loaded ocean model {model_path}, {model_choice}", flush=True)
print("Num params: ", sum(p.numel() for p in model.parameters()), flush=True)

forecasting_times = t_min + t_direct * np.arange(0, 1 + (t_max-t_min)//t_direct)
dataset = OceanDatasetERA5Style(lead_time=forecasting_times, dataset_mode='test', 
                                time_split_mode='date',
                                train_date_range=('2000-01-01', '2021-12-31'), 
                                val_date_range=('2022-01-01', '2022-12-31'),
                                test_date_range=('2023-01-01', '2023-12-31'),**kwargs)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

sampler_fn = heun_sampler

print(f"Dataset contains {len(dataset)} samples", flush=True)
print(f"We do {len(loader)} batches", flush=True)

model.eval()

# Initialize the dimensions based on the first batch
previous, current, time_labels = next(iter(loader))
n_times = time_labels.shape[1] if len(time_labels.shape) > 1 else len(forecasting_times)
n_conditions = previous.shape[1]
dx = current.shape[2]
dy = current.shape[3]

predictions = zarr.open(f'{result_path}/{name}.zarr', mode='w', 
                       shape=(len(dataset), n_ens, n_times, num_variables, dx, dy), 
                       chunks=(1, n_ens, n_times, num_variables, dx, dy),
                       dtype='float32', overwrite=True)

start_idx = 0


# def create_gaussian_kernel_2d(kernel_size, sigma, device='cpu'):
#     """创建2D高斯核"""
#     # 创建1D高斯核
#     coords = torch.arange(kernel_size, dtype=torch.float32, device=device)
#     coords = coords - kernel_size // 2
    
#     # 计算高斯函数
#     gaussian_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#     gaussian_1d = gaussian_1d / gaussian_1d.sum()
    
#     # 创建2D高斯核
#     gaussian_2d = gaussian_1d[:, None] * gaussian_1d[None, :]
    
#     return gaussian_2d

# def gaussian_smooth_batch(x, kernel_size=5, sigma=1.0):
#     """对批次数据进行高斯平滑"""
#     B, C, H, W = x.shape
#     device = x.device
    
#     # 创建高斯核
#     kernel = create_gaussian_kernel_2d(kernel_size, sigma, device)
#     kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1, 1, K, K]
#     kernel = kernel.expand(C, 1, -1, -1)  # [C, 1, K, K]
    
#     # 使用分组卷积对每个通道独立处理
#     padding = kernel_size // 2
#     smoothed = F.conv2d(x, kernel, padding=padding, groups=C)
    
#     return smoothed

# def check_variance_stability(variance_history, window_size=10, stability_threshold=0.01):
#     """检查方差是否趋于稳定"""
#     if len(variance_history) < window_size:
#         return False
    
#     # 计算最近window_size个方差值的标准差
#     recent_variances = variance_history[-window_size:]
#     variance_std = np.std(recent_variances)
    
#     # 如果方差的标准差小于阈值，认为已经稳定
#     return variance_std < stability_threshold

# def get_latents_with_stable_smoothing(latent_shape, n_direct, alpha=1.0, 
#                                     kernel_size=5, sigma=1.0, 
#                                     max_warmup_steps=50, 
#                                     stability_window=10, 
#                                     stability_threshold=0.01,
#                                     device='cuda'):
#     """
#     改进的潜在变量生成函数，采用高斯核平滑并选择稳定的噪声
    
#     参数:
#     - latent_shape: (B, C, H, W) 潜在变量形状
#     - n_direct: 需要生成的步数
#     - alpha: OU过程相关性参数 (1.0=完全相关, 0.0=完全不相关)
#     - kernel_size: 高斯核大小
#     - sigma: 高斯核标准差
#     - max_warmup_steps: 最大预热步数
#     - stability_window: 稳定性检查窗口大小
#     - stability_threshold: 稳定性阈值
#     - device: 设备
    
#     返回:
#     - z: 标准化的潜在变量
#     - info: 包含生成过程信息的字典
#     """
#     B, C, H, W = latent_shape
    
#     # 初始化
#     alpha = torch.tensor(alpha, device=device)
#     variance_history = []
#     stability_achieved = False
#     stable_start_step = 0
    
#     print(f"开始生成潜在变量...")
#     print(f"目标步数: {n_direct}, 最大预热步数: {max_warmup_steps}")
    
#     # 第一阶段：预热阶段，生成直到方差稳定
#     warmup_z = torch.zeros((max_warmup_steps, B, C, H, W), device=device)
#     warmup_z[0] = torch.randn((B, C, H, W), device=device)
    
#     for t in range(1, max_warmup_steps):
#         # 生成原始噪声
#         raw_noise = torch.randn((B, C, H, W), device=device)
        
#         # 高斯平滑
#         smoothed_noise = gaussian_smooth_batch(raw_noise, kernel_size, sigma)
        
#         # 重新标准化平滑后的噪声
#         smoothed_noise = (smoothed_noise - smoothed_noise.mean()) / (smoothed_noise.std() + 1e-8)
        
#         # OU过程更新
#         warmup_z[t] = alpha.sqrt() * warmup_z[t-1] + (1 - alpha).sqrt() * smoothed_noise
        
#         # 计算当前方差
#         current_variance = warmup_z[t].var().item()
#         variance_history.append(current_variance)
        
#         # 检查稳定性
#         if check_variance_stability(variance_history, stability_window, stability_threshold):
#             stability_achieved = True
#             stable_start_step = t
#             print(f"方差在第 {t} 步达到稳定状态")
#             break
        
#         # 显示进度
#         if t % 10 == 0:
#             print(f"  预热步数: {t}, 当前方差: {current_variance:.6f}")
    
#     if not stability_achieved:
#         print(f"警告: 在 {max_warmup_steps} 步内未达到稳定状态，使用最后状态")
#         stable_start_step = max_warmup_steps - 1
    
#     # 第二阶段：从稳定状态开始生成所需的步数
#     print(f"从稳定状态开始生成 {n_direct} 步...")
    
#     z = torch.zeros((n_direct, B, C, H, W), device=device)
    
#     # 使用稳定状态作为起点
#     z[0] = warmup_z[stable_start_step].clone()
    
#     # 继续生成剩余步数
#     for t in range(1, n_direct):
#         # 生成原始噪声
#         raw_noise = torch.randn((B, C, H, W), device=device)
        
#         # 高斯平滑
#         smoothed_noise = gaussian_smooth_batch(raw_noise, kernel_size, sigma)
        
#         # 重新标准化平滑后的噪声
#         smoothed_noise = (smoothed_noise - smoothed_noise.mean()) / (smoothed_noise.std() + 1e-8)
        
#         # OU过程更新
#         z[t] = alpha.sqrt() * z[t-1] + (1 - alpha).sqrt() * smoothed_noise
    
#     # 第三阶段：最终标准化以符合扩散模型输入要求
#     print("进行最终标准化...")
    
#     # 重塑为批次格式
#     z_reshaped = z.transpose(0, 1).reshape(n_direct * B, C, H, W)
    
#     # 全局标准化：确保整个批次的均值为0，标准差为1
#     z_mean = z_reshaped.mean()
#     z_std = z_reshaped.std()
#     z_normalized = (z_reshaped - z_mean) / (z_std + 1e-8)
    
#     # 验证最终结果
#     final_mean = z_normalized.mean().item()
#     final_std = z_normalized.std().item()
#     final_var = z_normalized.var().item()
    
#     print(f"最终统计特性:")
#     print(f"  均值: {final_mean:.6f}")
#     print(f"  标准差: {final_std:.6f}")
#     print(f"  方差: {final_var:.6f}")
    
#     # 检查是否符合N(0,1)分布
#     mean_ok = abs(final_mean) < 0.1
#     std_ok = abs(final_std - 1.0) < 0.1
#     print(f"  符合N(0,1)分布: 均值{'✓' if mean_ok else '✗'} 标准差{'✓' if std_ok else '✗'}")
    
#     return z_normalized

def get_fourier_latents(latent_shape, n_direct, D=0.3, target_mean=0.0, target_std=1.0, seed=42):
    """
    使用傅立叶域指数核衰减生成标准化噪声序列
    从固定的初始噪声开始，逐步应用衰减：h(ω, t) = e^{-D ||ω||^2 t}
    
    Args:
        latent_shape: 潜在空间形状 (B, C, H, W)
        n_direct: 时间步数
        D: 扩散系数，控制衰减速度
        target_mean: 目标均值
        target_std: 目标标准差
        seed: 随机种子，确保可重复性
    """
    B, C, H, W = latent_shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 设置随机种子确保可重复性
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # 生成固定的初始白噪声 (B, C, H, W)
    initial_white_noise = torch.randn((B, C, H, W), device=device)
    
    # 将初始噪声转换到傅立叶域（只需转换一次）
    # 重新整理维度以便批量处理
    initial_noise_reshaped = initial_white_noise.reshape(B * C, H, W)
    initial_noise_fft = torch.fft.fft2(initial_noise_reshaped)
    
    # 创建频率网格
    freq_h = torch.fft.fftfreq(H, device=device)
    freq_w = torch.fft.fftfreq(W, device=device)
    omega_h, omega_w = torch.meshgrid(freq_h, freq_w, indexing='ij')
    omega_magnitude_sq = omega_h**2 + omega_w**2
    
    # 存储每个时间步的结果
    filtered_sequence = torch.zeros((n_direct, B, C, H, W), device=device)
    
    for t in range(n_direct):
        # 计算当前时间步的衰减核
        # 使用累积衰减：从t=0（无衰减）到t=n_direct-1（最大衰减）
        current_decay = D * omega_magnitude_sq * np.log(t+1)
        decay_kernel = torch.exp(-current_decay)
        
        # 在傅立叶域应用衰减核到同一个初始噪声
        # 扩展衰减核的维度以匹配批量处理
        decay_kernel_expanded = decay_kernel.unsqueeze(0).repeat(B * C, 1, 1)
        filtered_fft = initial_noise_fft * decay_kernel_expanded
        
        # 逆FFT回到空间域
        filtered_noise = torch.fft.ifft2(filtered_fft).real
        
        # 恢复原始维度
        filtered_noise = filtered_noise.reshape(B, C, H, W)
        
        # 标准化当前时间步的噪声
        current_mean = filtered_noise.mean()
        current_std = filtered_noise.std()
        
        if current_std > 1e-8:  # 避免除零
            normalized_noise = (filtered_noise - current_mean) / current_std * target_std + target_mean
        else:
            normalized_noise = filtered_noise + target_mean
        
        filtered_sequence[t] = normalized_noise
    
    # 重新整理维度为 (n_direct * B, C, H, W)
    z = filtered_sequence.transpose(0, 1).reshape(n_direct * B, C, H, W)
    
    return z

def get_latents(latent_shape, n_direct, alpha=1.0):
    """
    Variance preserving function for the noise z. 
    alpha=1.0 means fixed noise, 
    alpha=0.0 means uncorrelated noise
    """
    B, C, H, W = latent_shape

    z = torch.zeros((n_direct, B, C, H, W), device=device)
    z[0] = torch.randn((B, C, H, W), device=device)
    alpha = torch.tensor(alpha, device=device)

    for t in range(1, n_direct):
        noise = torch.randn((B, C, H, W), device=device)
        z[t] = (alpha).sqrt() * z[t - 1] + (1 - alpha).sqrt() * noise

    z = z.transpose(0, 1).reshape(n_direct * B, C, H, W)
    return z

# Predict
for previous, current, time_labels in tqdm(loader):        
    n_samples = current.shape[0]

    with torch.no_grad():
        previous = previous.to(device)
        current = current.view(-1, num_variables, dx, dy).to(device)
        
        # Handle time_labels format
        if len(time_labels.shape) == 1:
            time_labels = time_labels.unsqueeze(0).repeat(n_samples, 1)
        
        direct_time_labels = torch.tensor(np.array([x for x in time_labels[0] if x <= t_iter]), device=device)
        n_iter = time_labels.shape[1] // direct_time_labels.shape[0] if direct_time_labels.shape[0] > 0 else 1
        n_direct = direct_time_labels.shape[0]

        class_labels = previous.repeat_interleave(n_direct * n_ens, dim=0)

        # Handle static fields (if any)
        if num_static_fields > 0:
            static_fields = class_labels[:, -num_static_fields:]
        else:
            static_fields = torch.empty(class_labels.shape[0], 0, dx, dy, device=device)

        latent_shape = (n_samples * n_ens, num_variables, dx, dy)
        direct_time_labels = direct_time_labels.repeat(n_ens * n_samples)

        predicted_combined = torch.zeros((n_samples, n_ens, n_times, num_variables, dx, dy), device=device)

        for i in tqdm(range(n_iter), desc="Iterative prediction"):
            # latents = torch.randn(latent_shape, device=device)
            # latents = get_latents(latent_shape, n_direct, alpha=1.0)
            # latents = get_latents_with_stable_smoothing(latent_shape, n_direct)
            latents = get_fourier_latents(latent_shape, n_direct)
            # latents = latents.repeat_interleave(n_direct, dim=0)
            
            predicted = sampler_fn(model, latents, class_labels, direct_time_labels / max_horizon, 
                                 sigma_max=80, sigma_min=0.03, rho=7, num_steps=20, 
                                 S_churn=2.5, S_min=0.75, S_max=80, S_noise=1.05)

            predicted_combined[:, :, i*n_direct:(i+1)*n_direct] = predicted.view(n_samples, n_ens, n_direct, num_variables, dx, dy)

            predicted = predicted.view(n_samples*n_ens, n_direct, num_variables, dx, dy)
            class_labels = class_labels.view(n_samples*n_ens, n_direct, n_conditions, dx, dy)[:, 0]

            if n_direct == 1:
                class_labels = torch.cat((predicted[:,-1], class_labels[:,:num_variables]), dim=1)
            else:
                class_labels = torch.cat((predicted[:,-1], predicted[:,-2]), dim=1).repeat_interleave(n_direct, dim=0)
            
            if num_static_fields > 0:
                class_labels = torch.cat((class_labels, static_fields), dim=1)

        # Save predictions incrementally to zarr file
        predictions[start_idx:start_idx + n_samples, :, :, :, :, :] = renormalize(predicted_combined).view(n_samples, n_ens, n_times, num_variables, dx, dy).cpu().numpy()

        start_idx += n_samples

    gc.collect()
    torch.cuda.empty_cache()

# Calculate ocean-specific metrics
metrics = zarr.open_group('ocean_evaluation_metrics.zarr', mode='a')

calculate_WCRPS = calculate_AreaWeightedRMSE(lat, lon, device).CRPS
calculate_WScores = calculate_AreaWeightedRMSE(lat, lon, device).skill_and_spread
calculate_WMAE = calculate_AreaWeightedRMSE(lat, lon, device).mae

skill_list = []
spread_list = []
ssr_list = []
CRPS_list = []
dx_same_list = []
dx_different_list = []
dx_truth_list = []

i = 0
with torch.no_grad():
    for previous, current, time_labels in tqdm(loader, desc="Calculating metrics"):
        if len(time_labels.shape) == 1:
            time_labels = time_labels.unsqueeze(0).repeat(current.shape[0], 1)
        
        n_times = time_labels.shape[1]
        n_samples, _, dx, dy = current.shape

        truth = renormalize(current.to(device).view(n_samples, n_times, num_variables, dx, dy))
       
        forecast = predictions[i:i + truth.shape[0]]
        forecast = torch.tensor(forecast, device=device)
        i = i + truth.shape[0]
        
        # Add ocean current magnitude (similar to wind speed for atmosphere)
        current_magnitude_truth = (truth[:,:,0]**2 + truth[:,:,1]**2).sqrt().unsqueeze(2)
        truth = torch.cat((truth, current_magnitude_truth), dim=2)
        current_magnitude_forecast = (forecast[:,:,:,0]**2 + forecast[:,:,:,1]**2).sqrt().unsqueeze(3)
        forecast = torch.cat((forecast, current_magnitude_forecast), dim=3)        
        
        # Calculate metrics
        skill, spread, ssr = calculate_WScores(forecast, truth)
        CRPS = calculate_WCRPS(forecast, truth)
        dx_same = calculate_WMAE(forecast[:, :, 1:, :], forecast[:, :, :-1, :])
        dx_different = calculate_WMAE(forecast[:, 1:, 1:, :], forecast[:, :-1, :-1, :])
        dx_truth = calculate_WMAE(truth[:, 1:, :].unsqueeze(1), truth[:, :-1, :].unsqueeze(1))

        # Append to list
        skill_list.append(skill)
        spread_list.append(spread)
        ssr_list.append(ssr)
        CRPS_list.append(CRPS)
        dx_same_list.append(dx_same)
        dx_different_list.append(dx_different)
        dx_truth_list.append(dx_truth)

skill = torch.tensor(np.array(skill_list)).mean(axis=0).cpu().numpy()
spread = torch.tensor(np.array(spread_list)).mean(axis=0).cpu().numpy()
ssr = torch.tensor(np.array(ssr_list)).mean(axis=0).cpu().numpy()
CRPS = torch.tensor(np.array(CRPS_list)).mean(axis=0).cpu().numpy()
dx_same = torch.tensor(np.array(dx_same_list)).mean(axis=0).cpu().numpy()
dx_different = torch.tensor(np.array(dx_different_list)).mean(axis=0).cpu().numpy()
dx_truth = torch.tensor(np.array(dx_truth_list)).mean(axis=0).cpu().numpy()

# Check if group for eval_name exists, else create it
if name not in metrics:
    metrics.create_group(name)

# Store the metrics in the corresponding group
group = metrics[name]
group.array('RMSE', skill, overwrite=True)
group.array('spread', spread, overwrite=True)
group.array('SSR', ssr, overwrite=True)
group.array('CRPS', CRPS, overwrite=True)
group.array('dx_same', dx_same, overwrite=True)
group.array('dx_different', dx_different, overwrite=True)
group.array('dx_truth', dx_truth, overwrite=True)
group.array('times', forecasting_times, overwrite=True)

print(f"Ocean metrics saved under {name}")
print(f"Results saved to: {result_path}")