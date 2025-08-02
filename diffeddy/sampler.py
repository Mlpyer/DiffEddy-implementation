# Code adapted from:
# Elucidating the Design Space of Diffusion-Based Generative Models (EDM)
# Tero Karras, Miika Aittala, Timo Aila, Samuli Laine
# https://github.com/NVlabs/edm

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""
import numpy as np
import torch
import torch.nn.functional as F
#----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
    net, latents, class_labels=None, time_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

        # Euler step.
        denoised = net(x_hat, t_hat, class_labels, time_labels)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels, time_labels)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next

#----------------------------------------------------------------------------
# Proposed Heun sampler (Algorithm 1).
def heun_sampler(
    net, latents, class_labels=None, time_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])]) # t_N = 0

    # Main sampling loop.
    x_next = latents * t_steps[0]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        denoised = net(x_cur, t_cur, class_labels, time_labels)
        d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels, time_labels)
            d_prime = (x_next - denoised) / t_next
            x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# def fluid_physics_edm_sampler(
#     net, latents, class_labels=None, time_labels=None, randn_like=torch.randn_like,
#     num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
#     S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
#     # 流体物理参数
#     fluid_coupling=0.01,          # 空间耦合强度
#     divergence_free=True,         # 强制无散度（不可压缩）
#     vorticity_preserve=True,      # 保持涡度
#     pressure_correction=True,     # 压力修正
#     apply_physics_steps=6,        # 只在前N步应用物理约束
# ):
#     """
#     基于流体物理的EDM采样器
    
#     核心思想：
#     1. 让初始噪声场满足流体的空间相关性
#     2. 在高噪声阶段强制无散度约束（不可压缩流体）
#     3. 保持涡度结构，增强空间连贯性
#     4. 简化操作，删除不必要的统计修正
#     """
    
#     # 标准EDM设置
#     sigma_min = max(sigma_min, net.sigma_min)
#     sigma_max = min(sigma_max, net.sigma_max)
    
#     step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
#     t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * 
#               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
#     t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
#     # 初始化：让初始噪声满足流体约束
#     x_next = initialize_fluid_noise(latents, t_steps[0], 
#                                   divergence_free=divergence_free,
#                                   vorticity_preserve=vorticity_preserve)
    
#     for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
#         x_cur = x_next
        
#         # ===========================================
#         # 1. 标准EDM步骤
#         # ===========================================
#         gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
#         t_hat = net.round_sigma(t_cur + gamma * t_cur)
#         x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
#         denoised = net(x_hat, t_hat, class_labels, time_labels)
#         d_cur = (x_hat - denoised) / t_hat
        
#         # ===========================================
#         # 2. 流体物理约束（仅在高噪声阶段）
#         # ===========================================
#         if i < apply_physics_steps:
#             d_final = apply_fluid_constraints(
#                 d_cur, fluid_coupling, 
#                 divergence_free=divergence_free,
#                 vorticity_preserve=vorticity_preserve,
#                 pressure_correction=pressure_correction,
#                 noise_level=t_cur / sigma_max
#             )
#         else:
#             d_final = d_cur
        
#         # ===========================================
#         # 3. 标准更新
#         # ===========================================
#         x_next = x_hat + (t_next - t_hat) * d_final
        
#         # ===========================================
#         # 4. 二阶修正（保持原始逻辑）
#         # ===========================================
#         if i < num_steps - 1:
#             denoised = net(x_next, t_next, class_labels, time_labels)
#             d_prime = (x_next - denoised) / t_next
            
#             # 对二阶项也应用流体约束（强度减半）
#             if i < apply_physics_steps:
#                 d_prime_final = apply_fluid_constraints(
#                     d_prime, fluid_coupling * 0.5,
#                     divergence_free=divergence_free,
#                     vorticity_preserve=vorticity_preserve,
#                     pressure_correction=False,  # 二阶项不需要压力修正
#                     noise_level=t_next / sigma_max
#                 )
#             else:
#                 d_prime_final = d_prime
            
#             x_next = x_hat + (t_next - t_hat) * (0.5 * d_final + 0.5 * d_prime_final)
    
#     return x_next


# def initialize_fluid_noise(latents, initial_sigma, divergence_free=True, vorticity_preserve=True):
#     """
#     初始化满足流体约束的噪声场
#     """
#     # 基础噪声
#     fluid_noise = latents * initial_sigma
    
#     if not (divergence_free or vorticity_preserve):
#         return fluid_noise
    
#     B, C, H, W = fluid_noise.shape
    
#     # 将前两个通道视为2D速度场 (u, v)
#     if C >= 2 and divergence_free:
#         # 通过流函数生成无散度场
#         # ψ -> (∂ψ/∂y, -∂ψ/∂x) 自动满足 ∇·v = 0
#         stream_function = torch.randn(B, 1, H, W, device=fluid_noise.device) * initial_sigma
        
#         # 计算梯度（有限差分）
#         u_vel = compute_gradient_y(stream_function)    # ∂ψ/∂y
#         v_vel = -compute_gradient_x(stream_function)   # -∂ψ/∂x
        
#         # 替换前两个通道
#         fluid_noise[:, 0:1] = u_vel
#         fluid_noise[:, 1:2] = v_vel
    
#     if C >= 2 and vorticity_preserve:
#         # 增强涡度结构
#         u_vel = fluid_noise[:, 0:1]
#         v_vel = fluid_noise[:, 1:2]
        
#         # 计算涡度 ω = ∂v/∂x - ∂u/∂y
#         vorticity = compute_gradient_x(v_vel) - compute_gradient_y(u_vel)
        
#         # 增强涡度结构（添加一些有组织的涡）
#         enhanced_vorticity = add_vortex_structures(vorticity)
        
#         # 从增强涡度重构速度场（保持无散度）
#         if divergence_free:
#             new_stream = solve_poisson(enhanced_vorticity)
#             fluid_noise[:, 0:1] = compute_gradient_y(new_stream)
#             fluid_noise[:, 1:2] = -compute_gradient_x(new_stream)
    
#     return fluid_noise


# def apply_fluid_constraints(velocity_field, coupling_strength, 
#                           divergence_free=True, vorticity_preserve=True, 
#                           pressure_correction=True, noise_level=1.0):
#     """
#     对速度场应用流体约束
#     """
#     if coupling_strength <= 0:
#         return velocity_field
    
#     # 动态调整约束强度
#     adaptive_strength = coupling_strength * noise_level
    
#     B, C, H, W = velocity_field.shape
#     result = velocity_field.clone()
    
#     # 只对前两个通道（视为2D速度场）应用流体约束
#     if C >= 2:
#         u_vel = result[:, 0:1]
#         v_vel = result[:, 1:2]
        
#         # 1. 无散度约束（不可压缩性）
#         if divergence_free:
#             # 计算散度
#             divergence = compute_gradient_x(u_vel) + compute_gradient_y(v_vel)
            
#             # 通过压力梯度消除散度
#             if pressure_correction:
#                 pressure = solve_poisson(divergence) * adaptive_strength
                
#                 # 减去压力梯度
#                 u_corrected = u_vel - compute_gradient_x(pressure)
#                 v_corrected = v_vel - compute_gradient_y(pressure)
                
#                 result[:, 0:1] = u_corrected
#                 result[:, 1:2] = v_corrected
        
#         # 2. 涡度保持（增强空间连贯性）
#         if vorticity_preserve:
#             # 计算当前涡度
#             current_vorticity = compute_gradient_x(result[:, 1:2]) - compute_gradient_y(result[:, 0:1])
            
#             # 轻微增强涡度结构
#             enhanced_vorticity = enhance_vorticity_structures(current_vorticity, adaptive_strength * 0.5)
            
#             # 从涡度重构速度场的修正项
#             if torch.abs(enhanced_vorticity - current_vorticity).mean() > 1e-6:
#                 vorticity_correction = solve_poisson(enhanced_vorticity - current_vorticity) * adaptive_strength
                
#                 result[:, 0:1] += compute_gradient_y(vorticity_correction) * 0.1
#                 result[:, 1:2] -= compute_gradient_x(vorticity_correction) * 0.1
    
#     # 3. 空间耦合（相邻像素的相关性）
#     spatial_coupling = apply_spatial_coupling(result, adaptive_strength * 0.3)
#     result = (1 - adaptive_strength * 0.2) * result + adaptive_strength * 0.2 * spatial_coupling
    
#     return result


# def compute_gradient_x(field):
#     """计算x方向梯度"""
#     # 使用中心差分
#     grad_x = torch.zeros_like(field)
#     grad_x[:, :, :, 1:-1] = (field[:, :, :, 2:] - field[:, :, :, :-2]) / 2.0
#     grad_x[:, :, :, 0] = field[:, :, :, 1] - field[:, :, :, 0]
#     grad_x[:, :, :, -1] = field[:, :, :, -1] - field[:, :, :, -2]
#     return grad_x


# def compute_gradient_y(field):
#     """计算y方向梯度"""
#     grad_y = torch.zeros_like(field)
#     grad_y[:, :, 1:-1, :] = (field[:, :, 2:, :] - field[:, :, :-2, :]) / 2.0
#     grad_y[:, :, 0, :] = field[:, :, 1, :] - field[:, :, 0, :]
#     grad_y[:, :, -1, :] = field[:, :, -1, :] - field[:, :, -2, :]
#     return grad_y


# def solve_poisson(source):
#     """
#     求解泊松方程 ∇²φ = source
#     使用简化的迭代方法
#     """
#     B, C, H, W = source.shape
#     phi = torch.zeros_like(source)
    
#     # 简化的Jacobi迭代（几次迭代就足够）
#     for _ in range(3):
#         phi_new = torch.zeros_like(phi)
        
#         # 内部点
#         phi_new[:, :, 1:-1, 1:-1] = 0.25 * (
#             phi[:, :, :-2, 1:-1] + phi[:, :, 2:, 1:-1] +
#             phi[:, :, 1:-1, :-2] + phi[:, :, 1:-1, 2:] -
#             source[:, :, 1:-1, 1:-1]
#         )
        
#         # 边界条件（零边界）
#         phi_new[:, :, 0, :] = 0
#         phi_new[:, :, -1, :] = 0
#         phi_new[:, :, :, 0] = 0
#         phi_new[:, :, :, -1] = 0
        
#         phi = phi_new
    
#     return phi


# def add_vortex_structures(vorticity):
#     """在涡度场中添加有组织的涡结构"""
#     B, C, H, W = vorticity.shape
#     enhanced = vorticity.clone()
    
#     # 添加一些随机的涡结构
#     for _ in range(2):  # 添加2个涡
#         # 随机位置和强度
#         center_x = torch.randint(W//4, 3*W//4, (B,))
#         center_y = torch.randint(H//4, 3*H//4, (B,))
#         strength = torch.randn(B, device=vorticity.device) * 0.1
        
#         # 高斯涡
#         y_coords = torch.arange(H, device=vorticity.device).float()[:, None]
#         x_coords = torch.arange(W, device=vorticity.device).float()[None, :]
        
#         for b in range(B):
#             r_squared = (x_coords - center_x[b])**2 + (y_coords - center_y[b])**2
#             vortex = strength[b] * torch.exp(-r_squared / (H*W*0.01))
#             enhanced[b, 0] += vortex
    
#     return enhanced


# def enhance_vorticity_structures(vorticity, strength):
#     """增强现有的涡度结构"""
#     if strength <= 0:
#         return vorticity
    
#     # 使用轻微的高斯滤波器增强连贯性
#     kernel_size = 3
#     sigma = 0.8
#     kernel = create_gaussian_kernel(kernel_size, sigma, vorticity.device, vorticity.dtype)
    
#     enhanced = torch.zeros_like(vorticity)
#     for c in range(vorticity.shape[1]):
#         enhanced[:, c:c+1] = F.conv2d(vorticity[:, c:c+1], kernel, padding=kernel_size//2)
    
#     # 轻微增强
#     return vorticity + strength * (enhanced - vorticity)


# def apply_spatial_coupling(field, coupling_strength):
#     """应用空间耦合，增强相邻像素的相关性"""
#     if coupling_strength <= 0:
#         return field
    
#     # 使用各向异性的耦合核
#     # 水平方向耦合
#     h_kernel = torch.tensor([[[[-0.1, 0.2, -0.1]]]], 
#                            device=field.device, dtype=field.dtype)
#     # 垂直方向耦合
#     v_kernel = torch.tensor([[[[-0.1], [0.2], [-0.1]]]], 
#                            device=field.device, dtype=field.dtype)
    
#     coupled = torch.zeros_like(field)
#     for c in range(field.shape[1]):
#         h_coupled = F.conv2d(field[:, c:c+1], h_kernel, padding=(0, 1))
#         v_coupled = F.conv2d(field[:, c:c+1], v_kernel, padding=(1, 0))
#         coupled[:, c:c+1] = field[:, c:c+1] + coupling_strength * (h_coupled + v_coupled)
    
#     return coupled


# def create_gaussian_kernel(kernel_size, sigma, device, dtype):
#     """创建高斯核"""
#     coords = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
#     g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
#     g /= g.sum()
#     kernel_2d = g[:, None] * g[None, :]
#     return kernel_2d.unsqueeze(0).unsqueeze(0)


# # 简化版本：只关注核心流体约束
# def streamlined_fluid_edm_sampler(
#     net, latents, class_labels=None, time_labels=None, randn_like=torch.randn_like,
#     num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
#     S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
#     divergence_strength=0.005,  # 无散度约束强度
#     physics_steps=4,            # 只在前4步应用
# ):
#     """
#     简化的流体物理EDM采样器
#     只保留最核心的无散度约束
#     """
#     # 标准EDM设置
#     sigma_min = max(sigma_min, net.sigma_min)
#     sigma_max = min(sigma_max, net.sigma_max)
    
#     step_indices = torch.arange(num_steps, dtype=torch.float32, device=latents.device)
#     t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * 
#               (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
#     t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
#     # 简单的流体初始化
#     x_next = latents * t_steps[0]
#     if divergence_strength > 0 and latents.shape[1] >= 2:
#         # 让前两个通道满足无散度约束
#         x_next = make_divergence_free(x_next, divergence_strength)
    
#     for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
#         x_cur = x_next
        
#         # 标准EDM
#         gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
#         t_hat = net.round_sigma(t_cur + gamma * t_cur)
#         x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        
#         denoised = net(x_hat, t_hat, class_labels, time_labels)
#         d_cur = (x_hat - denoised) / t_hat
        
#         # 简化的流体约束
#         if i < physics_steps and divergence_strength > 0:
#             d_final = make_divergence_free(d_cur, 
#                                          divergence_strength * (t_cur / sigma_max))
#         else:
#             d_final = d_cur
        
#         x_next = x_hat + (t_next - t_hat) * d_final
        
#         # 二阶修正
#         if i < num_steps - 1:
#             denoised = net(x_next, t_next, class_labels, time_labels)
#             d_prime = (x_next - denoised) / t_next
#             x_next = x_hat + (t_next - t_hat) * (0.5 * d_final + 0.5 * d_prime)
    
#     return x_next


# def make_divergence_free(field, strength):
#     """简化的无散度约束"""
#     if strength <= 0 or field.shape[1] < 2:
#         return field
    
#     B, C, H, W = field.shape
#     result = field.clone()
    
#     # 对前两个通道应用无散度约束
#     u = result[:, 0:1]
#     v = result[:, 1:2]
    
#     # 计算散度
#     div = compute_gradient_x(u) + compute_gradient_y(v)
    
#     # 简单的压力修正
#     pressure = solve_poisson(div) * strength
    
#     result[:, 0:1] = u - compute_gradient_x(pressure)
#     result[:, 1:2] = v - compute_gradient_y(pressure)
    
#     return result

import torch
import torch.nn.functional as F
import numpy as np

def get_physics_informed_latents(latent_shape, n_direct, alpha=1.0, 
                                physics_params=None, method='divergence_free',
                                physics_strength=0.1):
    """
    Physics-informed variance preserving function for fluid dynamics noise initialization.
    Maintains proper diffusion model noise statistics while adding physics constraints.
    
    Args:
        latent_shape: (B, C, H, W) shape of latent space
        n_direct: number of time steps
        alpha: correlation parameter (1.0=fixed, 0.0=uncorrelated)
        physics_params: dict with physical parameters
        method: physics constraint method ('divergence_free', 'vorticity', 'helmholtz', 'navier_stokes')
        physics_strength: strength of physics constraints (0.0=no physics, 1.0=full physics)
    """
    B, C, H, W = latent_shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Default physics parameters
    if physics_params is None:
        physics_params = {
            'viscosity': 0.1,
            'density': 1.0,
            'dt': 0.01,
            'reynolds_number': 100.0,
            'turbulence_intensity': 0.1
        }
    
    z = torch.zeros((n_direct, B, C, H, W), device=device)
    
    # Initialize first timestep - start with standard Gaussian noise
    z[0] = torch.randn((B, C, H, W), device=device)
    
    # Apply light physics constraints to first timestep if requested
    if physics_strength > 0 and method != 'none':
        physics_correction = generate_physics_correction(z[0], method, physics_params, device)
        z[0] = (1 - physics_strength) * z[0] + physics_strength * physics_correction
        # Renormalize to maintain unit variance
        z[0] = normalize_to_unit_variance(z[0])
    
    alpha = torch.tensor(alpha, device=device)
    
    # Time evolution maintaining variance preservation
    for t in range(1, n_direct):
        # Generate standard Gaussian noise
        base_noise = torch.randn((B, C, H, W), device=device)
        
        # Apply standard temporal correlation
        z_temp = (alpha).sqrt() * z[t - 1] + (1 - alpha).sqrt() * base_noise
        
        # Apply light physics constraints if requested
        if physics_strength > 0 and method != 'none':
            physics_correction = generate_physics_correction(z_temp, method, physics_params, device)
            z[t] = (1 - physics_strength) * z_temp + physics_strength * physics_correction
            # Renormalize to maintain proper variance
            z[t] = normalize_to_unit_variance(z[t])
        else:
            z[t] = z_temp
    
    z = z.transpose(0, 1).reshape(n_direct * B, C, H, W)
    return z


def normalize_to_unit_variance(tensor):
    """Normalize tensor to have unit variance while preserving mean"""
    # Compute current std along all spatial dimensions
    std = tensor.std(dim=(-2, -1), keepdim=True)
    # Avoid division by zero
    std = torch.clamp(std, min=1e-8)
    # Normalize to unit std
    return tensor / std


def generate_physics_correction(noise, method, physics_params, device):
    """
    Generate physics-informed correction that maintains noise statistics.
    This is a correction term, not a replacement for the base noise.
    """
    B, C, H, W = noise.shape
    correction = torch.zeros_like(noise)
    
    if method == 'divergence_free' and C >= 2:
        # Apply divergence-free constraint only to velocity components
        correction = apply_lightweight_divergence_free(noise[:, :2], physics_params)
        # Keep other channels unchanged
        if C > 2:
            result = torch.zeros_like(noise)
            result[:, :2] = correction
            result[:, 2:] = noise[:, 2:]
            correction = result
        else:
            result = torch.zeros_like(noise)
            result[:, :2] = correction
            correction = result
            
    elif method == 'vorticity' and C >= 2:
        correction = apply_lightweight_vorticity_structure(noise[:, :2], physics_params)
        if C > 2:
            result = torch.zeros_like(noise)
            result[:, :2] = correction
            result[:, 2:] = noise[:, 2:]
            correction = result
        else:
            result = torch.zeros_like(noise)
            result[:, :2] = correction
            correction = result
            
    elif method == 'helmholtz' and C >= 2:
        correction = apply_lightweight_helmholtz_structure(noise[:, :2], physics_params)
        if C > 2:
            result = torch.zeros_like(noise)
            result[:, :2] = correction
            result[:, 2:] = noise[:, 2:]
            correction = result
        else:
            result = torch.zeros_like(noise)
            result[:, :2] = correction
            correction = result
    else:
        correction = noise
    
    return correction


def apply_lightweight_divergence_free(velocity_noise, physics_params):
    """
    Apply lightweight divergence-free constraint that preserves noise statistics.
    Uses projection method but with careful normalization.
    """
    B, C, H, W = velocity_noise.shape
    if C < 2:
        return velocity_noise
    
    u, v = velocity_noise[:, 0], velocity_noise[:, 1]
    
    # Compute divergence with padding to handle boundaries
    u_padded = F.pad(u, (1, 1, 1, 1), mode='circular')
    v_padded = F.pad(v, (1, 1, 1, 1), mode='circular')
    
    # Central differences
    du_dx = (u_padded[:, 1:-1, 2:] - u_padded[:, 1:-1, :-2]) / 2.0
    dv_dy = (v_padded[:, 2:, 1:-1] - v_padded[:, :-2, 1:-1]) / 2.0
    
    divergence = du_dx + dv_dy
    
    # Apply light smoothing to divergence for stability
    div_smooth = gaussian_smooth_lightweight(divergence.unsqueeze(1), sigma=1.0).squeeze(1)
    
    # Compute correction (much smaller than original implementation)
    correction_scale = 0.1  # Reduce impact to preserve noise statistics
    
    # Compute potential correction
    phi = solve_poisson_lightweight(div_smooth)
    
    # Gradient of phi gives correction
    phi_padded = F.pad(phi, (1, 1, 1, 1), mode='circular')
    dphi_dx = (phi_padded[:, 1:-1, 2:] - phi_padded[:, 1:-1, :-2]) / 2.0
    dphi_dy = (phi_padded[:, 2:, 1:-1] - phi_padded[:, :-2, 1:-1]) / 2.0
    
    # Apply small correction
    u_corrected = u - correction_scale * dphi_dx
    v_corrected = v - correction_scale * dphi_dy
    
    result = torch.zeros_like(velocity_noise)
    result[:, 0] = u_corrected
    result[:, 1] = v_corrected
    
    return result


def apply_lightweight_vorticity_structure(velocity_noise, physics_params):
    """Apply lightweight vorticity-based structure"""
    B, C, H, W = velocity_noise.shape
    if C < 2:
        return velocity_noise
    
    # Add small vorticity-based correlation
    u, v = velocity_noise[:, 0], velocity_noise[:, 1]
    
    # Compute vorticity
    u_padded = F.pad(u, (1, 1, 1, 1), mode='circular')
    v_padded = F.pad(v, (1, 1, 1, 1), mode='circular')
    
    du_dy = (u_padded[:, 2:, 1:-1] - u_padded[:, :-2, 1:-1]) / 2.0
    dv_dx = (v_padded[:, 1:-1, 2:] - v_padded[:, 1:-1, :-2]) / 2.0
    
    vorticity = dv_dx - du_dy
    
    # Add small vorticity-based modulation
    vort_scale = 0.05
    vort_smooth = gaussian_smooth_lightweight(vorticity.unsqueeze(1), sigma=0.5).squeeze(1)
    
    u_mod = u + vort_scale * vort_smooth * torch.randn_like(u) * 0.1
    v_mod = v + vort_scale * vort_smooth * torch.randn_like(v) * 0.1
    
    result = torch.zeros_like(velocity_noise)
    result[:, 0] = u_mod
    result[:, 1] = v_mod
    
    return result


def apply_lightweight_helmholtz_structure(velocity_noise, physics_params):
    """Apply lightweight Helmholtz decomposition structure"""
    # For simplicity, combine divergence-free and small irrotational component
    div_free = apply_lightweight_divergence_free(velocity_noise, physics_params)
    
    # Add small irrotational component
    B, C, H, W = velocity_noise.shape
    phi = torch.randn(B, 1, H, W, device=velocity_noise.device) * 0.1
    phi_smooth = gaussian_smooth_lightweight(phi, sigma=1.0)
    
    phi_padded = F.pad(phi_smooth.squeeze(1), (1, 1, 1, 1), mode='circular')
    dphi_dx = (phi_padded[:, 1:-1, 2:] - phi_padded[:, 1:-1, :-2]) / 2.0
    dphi_dy = (phi_padded[:, 2:, 1:-1] - phi_padded[:, :-2, 1:-1]) / 2.0
    
    irrot_scale = 0.1
    result = div_free.clone()
    result[:, 0] += irrot_scale * dphi_dx
    result[:, 1] += irrot_scale * dphi_dy
    
    return result


def gaussian_smooth_lightweight(tensor, sigma=1.0):
    """Lightweight Gaussian smoothing with small kernel"""
    kernel_size = 3  # Fixed small kernel
    
    # Simple 3x3 Gaussian kernel
    kernel = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], 
                         dtype=tensor.dtype, device=tensor.device) / 16.0
    kernel = kernel.view(1, 1, 3, 3)
    
    # Apply convolution
    return F.conv2d(tensor, kernel, padding=1)


def solve_poisson_lightweight(divergence):
    """Lightweight Poisson solver using simple iteration"""
    # Very simple Jacobi iteration for Poisson equation
    phi = torch.zeros_like(divergence)
    
    # Just a few iterations to get approximate solution
    for _ in range(3):
        phi_padded = F.pad(phi, (1, 1, 1, 1), mode='circular')
        phi_new = 0.25 * (
            phi_padded[:, 2:, 1:-1] + phi_padded[:, :-2, 1:-1] +
            phi_padded[:, 1:-1, 2:] + phi_padded[:, 1:-1, :-2] +
            divergence
        )
        phi = phi_new
    
    return phi


def generate_divergence_free_noise(latent_shape, physics_params, device):
    """Generate divergence-free velocity field noise (∇·v = 0)"""
    B, C, H, W = latent_shape
    
    # Generate stream function ψ
    psi = torch.randn(B, 1, H, W, device=device)
    
    # Apply Gaussian smoothing to ensure differentiability
    psi = gaussian_smooth(psi, sigma=2.0)
    
    if C >= 2:  # Assume first two channels are velocity components
        # Compute velocity from stream function: v_x = ∂ψ/∂y, v_y = -∂ψ/∂x
        v_x = torch.gradient(psi, dim=2)[0]  # ∂ψ/∂y
        v_y = -torch.gradient(psi, dim=3)[0]  # -∂ψ/∂x
        
        noise = torch.randn(B, C, H, W, device=device)
        noise[:, 0] = v_x.squeeze(1)
        noise[:, 1] = v_y.squeeze(1)
        
        # Scale by turbulence intensity
        noise[:, :2] *= physics_params.get('turbulence_intensity', 0.1)
    else:
        noise = torch.randn(B, C, H, W, device=device)
    
    return noise


def generate_vorticity_based_noise(latent_shape, physics_params, device):
    """Generate noise based on vorticity dynamics"""
    B, C, H, W = latent_shape
    
    # Generate vorticity field
    omega = torch.randn(B, 1, H, W, device=device)
    omega = gaussian_smooth(omega, sigma=1.5)
    
    # Scale by Reynolds number influence
    Re = physics_params.get('reynolds_number', 100.0)
    vorticity_scale = np.sqrt(Re) / 100.0
    omega *= vorticity_scale
    
    if C >= 2:
        # Convert vorticity to velocity field using Poisson solver
        velocity = vorticity_to_velocity(omega, device)
        
        noise = torch.randn(B, C, H, W, device=device)
        noise[:, 0] = velocity[:, 0]
        noise[:, 1] = velocity[:, 1]
    else:
        noise = torch.randn(B, C, H, W, device=device)
    
    return noise


def generate_helmholtz_decomposed_noise(latent_shape, physics_params, device):
    """Generate noise using Helmholtz decomposition (solenoidal + irrotational)"""
    B, C, H, W = latent_shape
    
    # Solenoidal component (divergence-free)
    solenoidal = generate_divergence_free_noise(latent_shape, physics_params, device)
    
    # Irrotational component (curl-free)
    phi = torch.randn(B, 1, H, W, device=device)
    phi = gaussian_smooth(phi, sigma=1.0)
    
    if C >= 2:
        # Velocity from potential: v = ∇φ
        v_x_irrot = torch.gradient(phi, dim=3)[0]  # ∂φ/∂x
        v_y_irrot = torch.gradient(phi, dim=2)[0]  # ∂φ/∂y
        
        irrotational = torch.randn(B, C, H, W, device=device)
        irrotational[:, 0] = v_x_irrot.squeeze(1)
        irrotational[:, 1] = v_y_irrot.squeeze(1)
        
        # Combine components with physical weighting
        solenoidal_weight = 0.8  # Favor incompressible flow
        irrotational_weight = 0.2
        
        noise = solenoidal_weight * solenoidal + irrotational_weight * irrotational
    else:
        noise = solenoidal
    
    return noise


def generate_navier_stokes_informed_noise(latent_shape, physics_params, device):
    """Generate noise informed by Navier-Stokes equation structure"""
    B, C, H, W = latent_shape
    
    # Start with divergence-free base
    noise = generate_divergence_free_noise(latent_shape, physics_params, device)
    
    if C >= 2:
        # Add pressure gradient and viscous effects
        nu = physics_params.get('viscosity', 0.1)
        dt = physics_params.get('dt', 0.01)
        
        # Simulate viscous diffusion effect
        noise[:, :2] = apply_viscous_diffusion(noise[:, :2], nu, dt)
        
        # Add pressure-like field if available
        if C >= 3:
            pressure = torch.randn(B, 1, H, W, device=device)
            pressure = gaussian_smooth(pressure, sigma=2.0)
            noise[:, 2] = pressure.squeeze(1)
    
    return noise


def apply_divergence_free_constraint(noise, physics_params):
    """Project noise onto divergence-free space"""
    B, C, H, W = noise.shape
    
    if C >= 2:
        # Apply divergence-free projection
        u, v = noise[:, 0], noise[:, 1]
        
        # Compute divergence
        du_dx = torch.gradient(u, dim=2)[0]
        dv_dy = torch.gradient(v, dim=1)[0]
        div = du_dx + dv_dy
        
        # Remove divergent component (simplified projection)
        div_smooth = gaussian_smooth(div.unsqueeze(1), sigma=1.0).squeeze(1)
        
        # Subtract divergent part
        correction_x = torch.gradient(div_smooth, dim=2)[0]
        correction_y = torch.gradient(div_smooth, dim=1)[0]
        
        noise[:, 0] -= 0.5 * correction_x
        noise[:, 1] -= 0.5 * correction_y
    
    return noise


def apply_vorticity_evolution(noise, prev_z, physics_params):
    """Apply vorticity-based evolution"""
    if prev_z.shape[1] >= 2:
        # Compute vorticity from previous state
        u_prev, v_prev = prev_z[:, 0], prev_z[:, 1]
        
        du_dy = torch.gradient(u_prev, dim=1)[0]
        dv_dx = torch.gradient(v_prev, dim=2)[0]
        omega_prev = dv_dx - du_dy
        
        # Apply vorticity advection effect to noise
        Re = physics_params.get('reynolds_number', 100.0)
        vorticity_influence = torch.tanh(omega_prev / Re) * 0.1
        
        noise[:, 0] += vorticity_influence * 0.5
        noise[:, 1] += vorticity_influence * 0.5
    
    return apply_divergence_free_constraint(noise, physics_params)


def apply_helmholtz_evolution(noise, prev_z, physics_params):
    """Apply Helmholtz decomposition evolution"""
    # Maintain the solenoidal-irrotational balance
    return apply_divergence_free_constraint(noise, physics_params)


def apply_navier_stokes_evolution(noise, prev_z, physics_params):
    """Apply Navier-Stokes informed evolution"""
    nu = physics_params.get('viscosity', 0.1)
    dt = physics_params.get('dt', 0.01)
    
    # Apply viscous effects
    if prev_z.shape[1] >= 2:
        noise[:, :2] = apply_viscous_diffusion(noise[:, :2], nu, dt)
    
    return apply_divergence_free_constraint(noise, physics_params)


def apply_physics_constraints(z, method, physics_params):
    """Apply physics constraints to maintain consistency"""
    if method in ['divergence_free', 'helmholtz', 'navier_stokes']:
        z = apply_divergence_free_constraint(z, physics_params)
    
    # Apply boundary conditions (no-slip for walls)
    z = apply_boundary_conditions(z, physics_params)
    
    return z


def apply_viscous_diffusion(velocity, nu, dt):
    """Apply viscous diffusion using finite difference"""
    # Simple 2D diffusion: v_new = v + nu * dt * ∇²v
    laplacian = compute_laplacian(velocity)
    return velocity + nu * dt * laplacian


def apply_boundary_conditions(z, physics_params):
    """Apply boundary conditions (e.g., no-slip walls)"""
    # Example: zero velocity at boundaries
    if z.shape[1] >= 2:
        z[:, :2, 0, :] = 0  # top boundary
        z[:, :2, -1, :] = 0  # bottom boundary
        z[:, :2, :, 0] = 0  # left boundary
        z[:, :2, :, -1] = 0  # right boundary
    
    return z


def compute_laplacian(field):
    """Compute 2D Laplacian using finite differences"""
    # Simple 5-point stencil
    laplacian = torch.zeros_like(field)
    
    laplacian[:, :, 1:-1, 1:-1] = (
        field[:, :, 2:, 1:-1] + field[:, :, :-2, 1:-1] +
        field[:, :, 1:-1, 2:] + field[:, :, 1:-1, :-2] -
        4 * field[:, :, 1:-1, 1:-1]
    )
    
    return laplacian


def gaussian_smooth(tensor, sigma=1.0):
    """Apply Gaussian smoothing"""
    kernel_size = int(2 * sigma * 3) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create Gaussian kernel
    kernel = torch.zeros(kernel_size, kernel_size, device=tensor.device)
    center = kernel_size // 2
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            x, y = i - center, j - center
            kernel[i, j] = torch.exp(torch.tensor(-(x**2 + y**2) / (2 * sigma**2)))
    
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size)
    
    # Apply convolution
    return F.conv2d(tensor, kernel, padding=kernel_size//2)


def vorticity_to_velocity(omega, device):
    """Convert vorticity to velocity using simplified Poisson solver"""
    B, _, H, W = omega.shape
    
    # Simplified: assume velocity is proportional to vorticity gradient
    domega_dy = torch.gradient(omega, dim=2)[0]
    domega_dx = torch.gradient(omega, dim=3)[0]
    
    # Velocity components (simplified relationship)
    u = domega_dy
    v = -domega_dx
    
    velocity = torch.cat([u, v], dim=1)
    return velocity


# Example usage and comparison
if __name__ == "__main__":
    # Example parameters
    latent_shape = (4, 3, 64, 64)  # B, C, H, W
    n_direct = 10
    
    physics_params = {
        'viscosity': 0.01,
        'density': 1.0,
        'dt': 0.01,
        'reynolds_number': 1000.0,
        'turbulence_intensity': 0.2
    }
    
    # Generate original noise (for comparison)
    z_original = get_latents(latent_shape, n_direct, alpha=0.8)
    
    # Generate physics-informed noise with different strengths
    z_light_physics = get_physics_informed_latents(
        latent_shape, n_direct, alpha=0.8,
        physics_params=physics_params,
        method='divergence_free',
        physics_strength=0.1  # Light physics constraint
    )
    
    z_medium_physics = get_physics_informed_latents(
        latent_shape, n_direct, alpha=0.8,
        physics_params=physics_params,
        method='divergence_free',
        physics_strength=0.3  # Medium physics constraint
    )
    
    # Check statistics
    print("=== Noise Statistics Comparison ===")
    print(f"Original noise - Mean: {z_original.mean():.4f}, Std: {z_original.std():.4f}")
    print(f"Light physics - Mean: {z_light_physics.mean():.4f}, Std: {z_light_physics.std():.4f}")
    print(f"Medium physics - Mean: {z_medium_physics.mean():.4f}, Std: {z_medium_physics.std():.4f}")
    
    # Check divergence (for velocity channels)
    def compute_divergence(velocity_field):
        B, C, H, W = velocity_field.shape
        if C >= 2:
            u, v = velocity_field[:, 0], velocity_field[:, 1]
            u_padded = F.pad(u, (1, 1, 1, 1), mode='circular')
            v_padded = F.pad(v, (1, 1, 1, 1), mode='circular')
            du_dx = (u_padded[:, 1:-1, 2:] - u_padded[:, 1:-1, :-2]) / 2.0
            dv_dy = (v_padded[:, 2:, 1:-1] - v_padded[:, :-2, 1:-1]) / 2.0
            return (du_dx + dv_dy).abs().mean()
        return 0.0
    
    # Reshape for divergence computation
    z_orig_reshaped = z_original.reshape(n_direct, latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])
    z_light_reshaped = z_light_physics.reshape(n_direct, latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])
    z_medium_reshaped = z_medium_physics.reshape(n_direct, latent_shape[0], latent_shape[1], latent_shape[2], latent_shape[3])
    
    div_orig = compute_divergence(z_orig_reshaped[0])
    div_light = compute_divergence(z_light_reshaped[0])
    div_medium = compute_divergence(z_medium_reshaped[0])
    
    print(f"\n=== Divergence Comparison (lower is more physics-consistent) ===")
    print(f"Original noise divergence: {div_orig:.6f}")
    print(f"Light physics divergence: {div_light:.6f}")
    print(f"Medium physics divergence: {div_medium:.6f}")


# Alternative: Simpler physics-aware initialization
def get_simple_physics_latents(latent_shape, n_direct, alpha=1.0, 
                              apply_divergence_free=False, constraint_strength=0.1):
    """
    Simplified physics-aware noise that maintains diffusion model requirements.
    
    Args:
        latent_shape: (B, C, H, W) 
        n_direct: number of time steps
        alpha: correlation parameter
        apply_divergence_free: whether to apply divergence-free constraint
        constraint_strength: strength of physics constraint (0.0-1.0)
    """
    B, C, H, W = latent_shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    z = torch.zeros((n_direct, B, C, H, W), device=device)
    z[0] = torch.randn((B, C, H, W), device=device)
    
    alpha = torch.tensor(alpha, device=device)
    
    for t in range(1, n_direct):
        base_noise = torch.randn((B, C, H, W), device=device)
        z[t] = (alpha).sqrt() * z[t - 1] + (1 - alpha).sqrt() * base_noise
        
        # Apply light physics constraint if requested
        if apply_divergence_free and C >= 2:
            z[t] = apply_simple_divergence_constraint(z[t], constraint_strength)
    
    z = z.transpose(0, 1).reshape(n_direct * B, C, H, W)
    return z


def apply_simple_divergence_constraint(noise, strength):
    """Apply simple divergence constraint without changing noise statistics significantly"""
    B, C, H, W = noise.shape
    
    if C >= 2:
        u, v = noise[:, 0], noise[:, 1]
        
        # Compute divergence
        du_dx = torch.gradient(u, dim=2)[0]
        dv_dy = torch.gradient(v, dim=1)[0]
        div = du_dx + dv_dy
        
        # Apply very light correction
        correction = strength * 0.1 * div
        
        # Distribute correction
        noise[:, 0] -= correction * 0.5
        noise[:, 1] -= correction * 0.5
        
        # Renormalize to maintain statistics
        noise = normalize_to_unit_variance(noise)
    
    return noise