import torch
import torch.nn as nn
import numpy as np

import math
from diffusion_networks import *

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "GenCast: Diffusion-based
# ensemble forecasting for medium-range weather".
class WGCLoss:
    def __init__(self, lat, lon, device, sigma_min=0.02, sigma_max=88, rho=7, sigma_data=1, time_noise=0, precomputed_std=None):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.rho = rho
        self.sigma_data = sigma_data
        self.time_noise = time_noise
        self.area_weights = torch.tensor(comp_area_weights_simple(lat, lon), device=device, dtype=torch.float32)
    
        self.precomputed_std = precomputed_std

    def residual_scaling(self, x):
        if x.ndim == 0:
            x = x.unsqueeze(0)  
        indices = (len(self.precomputed_std)*x).to(dtype=int) - 1
        
        return self.precomputed_std[indices].view(x.shape[0], -1, 1, 1)
    
    def __call__(self, net, images, class_labels=None, time_labels=None):
        # Time Augmentation
        if self.time_noise > 0:
            time_labels = time_labels + torch.randn_like(time_labels, device=images.device, dtype=torch.float32) * self.time_noise
        
        # Sample from F inverse
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)

        rnd_uniform = 1 - rnd_uniform

        rho_inv = 1 / self.rho
        sigma_max_rho = self.sigma_max ** rho_inv
        sigma_min_rho = self.sigma_min ** rho_inv
        
        sigma = (sigma_max_rho + rnd_uniform * (sigma_min_rho - sigma_max_rho)) ** self.rho
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, class_labels, time_labels)
        loss = self.area_weights * weight * ((D_yn - y) ** 2)

        if self.precomputed_std != None:
            loss = loss / self.residual_scaling(time_labels) # Scale by residual weight

        loss = loss.sum().mul(1/(images.shape[0]*images.shape[1]))
        return loss, D_yn

#----------------------------------------------------------------------------
# Area weighted loss function from the codebase 
# diffusion-models-for-weather-prediction

class calculate_WeightedRMSE:
    def __init__(self, weights, device):
        self.weights = torch.tensor(weights, device=device, dtype=torch.float32)   

    def diff(self, input: torch.tensor, target: torch.tensor):
        return (self.weights * (input - target) ** 2)
    
    def loss_fn(self, input: torch.tensor, target: torch.tensor):
        return self.diff(input, target).mean().sqrt()

    def skill_and_spread(self, input: torch.tensor, target: torch.tensor):
        ens_mean = input.mean(dim=1, keepdim=True)
        dims_to_include = (-1, -2)

        skill = self.diff(ens_mean, target.unsqueeze(1)).mean(dim=dims_to_include).sqrt()

        N = input.size(1)
        spread = ((self.weights*(ens_mean - input)**2).sum(dim=1)/(N - 1)).mean(dim=dims_to_include).sqrt()
        
        ssr = np.sqrt((N+1)/N) * (spread / skill).mean(dim=0).cpu().detach().numpy()[0]
        skill = skill.mean(dim=0).cpu().detach().numpy()[0]
        spread = spread.mean(dim=0).cpu().detach().numpy()

        return skill, spread, ssr

    def CRPS(self, input: torch.tensor, target: torch.tensor):
        dims_to_include = (-1, -2)

        a = (input - target.unsqueeze(1)).abs().mean(dim=1)
        b = (input.unsqueeze(2) - input.unsqueeze(1)).abs().mean(dim=(1,2)) * 0.5
        c = (self.weights*(a - b)).mean(dim=dims_to_include).mean(dim=0)
        return c.cpu().detach().numpy()
    
    
    def mae(self, input: torch.tensor, target: torch.tensor):
        dims_to_include = (-1, -2)

        c = (self.weights*(input - target)).abs().mean(dim=dims_to_include).mean(dim=(0,1))
        return c.cpu().detach().numpy()


class calculate_AreaWeightedRMSE(calculate_WeightedRMSE):
    def __init__(self, lat, lon, device):
        super().__init__(weights=comp_area_weights_simple(lat, lon), device=device)
    

# ----

def comp_area_weights_simple(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Calculate the normalized area weights.

    Args:
        lat (np.ndarray): Array of latitudes of grid center points
        lon (np.ndarray): Array of longitudes of grid center points

    Returns:
        np.ndarray: 2D array of relative area sizes.
    """
    area_weights = np.cos(lat * np.pi / 180).reshape(-1, 1)
    area_weights = np.repeat(area_weights, lon.size, axis=1)
    area_weights *= (lat.size * lon.size / np.sum(area_weights))
    return area_weights

# Physics-Constrained Loss Functions for Ocean Diffusion Model

class PhysicsLoss(nn.Module):
    """
    Physical constraint loss functions for ocean dynamics
    """
    def __init__(self, lat_coords, lon_coords, depth=None, device='cuda'):
        super().__init__()
        self.device = device
        
        # Convert coordinates to tensors and handle potential shape issues
        if isinstance(lat_coords, np.ndarray):
            self.lat_coords = torch.tensor(lat_coords, device=device, dtype=torch.float32)
        else:
            self.lat_coords = torch.tensor(lat_coords, device=device, dtype=torch.float32)
            
        if isinstance(lon_coords, np.ndarray):
            self.lon_coords = torch.tensor(lon_coords, device=device, dtype=torch.float32)
        else:
            self.lon_coords = torch.tensor(lon_coords, device=device, dtype=torch.float32)
        
        # Ensure 1D arrays
        self.lat_coords = self.lat_coords.flatten()
        self.lon_coords = self.lon_coords.flatten()
        
        # Earth parameters
        self.earth_radius = 6.371e6  # meters
        self.omega = 7.2921e-5  # Earth's rotation rate (rad/s)
        
        # Convert to radians and compute grid spacing
        self.lat_rad = self.lat_coords * np.pi / 180
        self.lon_rad = self.lon_coords * np.pi / 180
        
        # Compute Coriolis parameter
        self.f = 2 * self.omega * torch.sin(self.lat_rad)  # [n_lat]
        
        # Grid spacing (assuming regular grid)
        if len(self.lat_rad) > 1:
            self.dlat = torch.abs(self.lat_rad[1] - self.lat_rad[0])
        else:
            self.dlat = torch.tensor(1.0, device=device)  # Default value
            
        if len(self.lon_rad) > 1:
            self.dlon = torch.abs(self.lon_rad[1] - self.lon_rad[0])
        else:
            self.dlon = torch.tensor(1.0, device=device)  # Default value
        
        print(f"PhysicsLoss initialized:")
        print(f"  lat_coords shape: {self.lat_coords.shape}")
        print(f"  lon_coords shape: {self.lon_coords.shape}")
        print(f"  f (Coriolis) shape: {self.f.shape}")
        
    def compute_spatial_derivatives(self, field):
        """
        Simplified spatial derivatives using finite differences
        Args:
            field: [batch, channels, lat, lon]
        Returns:
            dfdy, dfdx: derivatives in lat and lon directions
        """
        try:
            batch_size, channels, n_lat, n_lon = field.shape
            
            # Initialize output tensors
            dfdy = torch.zeros_like(field)
            dfdx = torch.zeros_like(field)
            
            # Simple finite differences without coordinate scaling for robustness
            # Latitude derivative (y-direction)
            if n_lat > 2:
                dfdy[:, :, 1:-1, :] = (field[:, :, 2:, :] - field[:, :, :-2, :]) / 2.0
                dfdy[:, :, 0, :] = field[:, :, 1, :] - field[:, :, 0, :]
                dfdy[:, :, -1, :] = field[:, :, -1, :] - field[:, :, -2, :]
            elif n_lat == 2:
                dfdy[:, :, 0, :] = field[:, :, 1, :] - field[:, :, 0, :]
                dfdy[:, :, 1, :] = field[:, :, 1, :] - field[:, :, 0, :]
            elif n_lat == 1:
                dfdy[:, :, 0, :] = torch.zeros_like(field[:, :, 0, :])
            
            # Longitude derivative (x-direction) with circular boundary
            if n_lon > 2:
                dfdx[:, :, :, 1:-1] = (field[:, :, :, 2:] - field[:, :, :, :-2]) / 2.0
                # Circular boundary conditions for longitude
                dfdx[:, :, :, 0] = (field[:, :, :, 1] - field[:, :, :, -1]) / 2.0
                dfdx[:, :, :, -1] = (field[:, :, :, 0] - field[:, :, :, -2]) / 2.0
            elif n_lon == 2:
                dfdx[:, :, :, 0] = field[:, :, :, 1] - field[:, :, :, 0]
                dfdx[:, :, :, 1] = field[:, :, :, 0] - field[:, :, :, 1]  # Circular
            elif n_lon == 1:
                dfdx[:, :, :, 0] = torch.zeros_like(field[:, :, :, 0])
            
            return dfdy, dfdx
            
        except Exception as e:
            print(f"Error in compute_spatial_derivatives: {e}")
            # Return zero derivatives as fallback
            dfdy = torch.zeros_like(field)
            dfdx = torch.zeros_like(field)
            return dfdy, dfdx
    
    def continuity_loss(self, u, v):
        """
        Continuity equation for incompressible flow: ∇·v = ∂u/∂x + ∂v/∂y = 0
        """
        dudy, dudx = self.compute_spatial_derivatives(u)
        dvdy, dvdx = self.compute_spatial_derivatives(v)
        
        divergence = dudx + dvdy
        return torch.mean(divergence**2)
    
    def geostrophic_balance_loss(self, u, v, ssh=None):
        """
        Simplified geostrophic balance loss
        """
        try:
            # Simplified: assume relative vorticity is small compared to planetary vorticity
            # ∂v/∂x - ∂u/∂y ≈ small for large-scale geostrophic flow
            dudy, dudx = self.compute_spatial_derivatives(u)
            dvdy, dvdx = self.compute_spatial_derivatives(v)
            
            relative_vorticity = dvdx - dudy
            # Use L1 loss instead of L2 for better stability
            return torch.mean(torch.abs(relative_vorticity))
        except Exception as e:
            print(f"Geostrophic balance computation failed: {e}")
            return torch.tensor(0.0, device=self.device)
    
    def energy_conservation_loss(self, u_pred, v_pred, u_true, v_true):
        """
        Kinetic energy should be conserved approximately
        """
        ke_pred = 0.5 * (u_pred**2 + v_pred**2)
        ke_true = 0.5 * (u_true**2 + v_true**2)
        
        # L2 loss on kinetic energy
        return torch.mean((ke_pred - ke_true)**2)
    
    def temporal_consistency_loss(self, current_pred, previous_true, dt=1.0):
        """
        Temporal derivative should be smooth (simplified)
        """
        # Compute temporal derivative
        if current_pred.shape == previous_true.shape:
            temporal_diff = (current_pred - previous_true) / dt
            # Penalize large temporal changes (smoothness)
            return torch.mean(temporal_diff**2)
        return torch.tensor(0.0, device=self.device)


class PhysicsConstrainedLoss(nn.Module):
    """
    Combined loss function with physics constraints
    """
    def __init__(self, lat_coords, lon_coords, device='cuda', 
                 weights=None, precomputed_std=None, original_loss_fn=None):
        super().__init__()
        
        # Initialize physics loss
        self.physics_loss = PhysicsLoss(lat_coords, lon_coords, device=device)
        
        # Loss weights
        default_weights = {
            'data': 1.0,
            'continuity': 0.1,
            'geostrophic': 0.1,
            'energy': 0.05,
            'temporal': 0.02
        }
        self.weights = weights if weights is not None else default_weights
        
        # Use original loss function if provided, otherwise fallback
        if original_loss_fn is not None:
            self.data_loss_fn = original_loss_fn
        elif precomputed_std is not None:
            try:
                self.data_loss_fn = WGCLoss(lat_coords, lon_coords, device, precomputed_std=precomputed_std)
            except ImportError:
                self.data_loss_fn = nn.MSELoss()
        else:
            self.data_loss_fn = nn.MSELoss()
        
    def forward(self, model, target, previous_input, time_label, **kwargs):
        """
        Combined forward pass with physics constraints
        """
        # Compute data loss using the original loss function interface
        if isinstance(self.data_loss_fn, WGCLoss):
            # WGCLoss takes (model, target, class_labels, time_labels)
            data_loss, predicted = self.data_loss_fn(model, target, previous_input, time_label)
        elif hasattr(self.data_loss_fn, '__call__') and hasattr(self.data_loss_fn, '__code__') and len(self.data_loss_fn.__code__.co_varnames) > 4:
            # Custom loss function that takes (model, target, previous_input, time_label)
            data_loss = self.data_loss_fn(model, target, previous_input, time_label)
            predicted = model(previous_input, time_label)
        else:
            # MSE or similar that takes (predicted, target)
            predicted = model(previous_input, time_label)
            data_loss = self.data_loss_fn(predicted, target)
        
        # Extract u and v components (assuming first 2 channels are u,v)
        # Handle case where predicted might have more channels than target
        num_vars = min(predicted.shape[1], target.shape[1], 2)
        u_pred = predicted[:, 0:1, :, :]  # First channel: u-component
        v_pred = predicted[:, 1:2, :, :] if num_vars > 1 else predicted[:, 0:1, :, :]  # Second channel: v-component
        u_true = target[:, 0:1, :, :]
        v_true = target[:, 1:2, :, :] if target.shape[1] > 1 else target[:, 0:1, :, :]
        
        # Compute individual losses
        losses = {}
        losses['data'] = data_loss
        
        # Physics constraint losses
        try:
            losses['continuity'] = self.physics_loss.continuity_loss(u_pred, v_pred)
            losses['geostrophic'] = self.physics_loss.geostrophic_balance_loss(u_pred, v_pred)
            losses['energy'] = self.physics_loss.energy_conservation_loss(u_pred, v_pred, u_true, v_true)
            
            # Temporal consistency (if previous timestep available)
            if 'previous_target' in kwargs:
                losses['temporal'] = self.physics_loss.temporal_consistency_loss(
                    predicted, kwargs['previous_target']
                )
            else:
                losses['temporal'] = torch.tensor(0.0, device=predicted.device)
                
        except Exception as e:
            print(f"Warning: Physics loss computation failed: {e}")
            # Fallback to zero physics losses if computation fails
            losses['continuity'] = torch.tensor(0.0, device=predicted.device)
            losses['geostrophic'] = torch.tensor(0.0, device=predicted.device)
            losses['energy'] = torch.tensor(0.0, device=predicted.device)
            losses['temporal'] = torch.tensor(0.0, device=predicted.device)
        
        # Combine losses
        total_loss = sum(self.weights[key] * losses[key] for key in losses.keys())
        
        # Return total loss and individual components for monitoring
        return total_loss, losses



# Simple wrapper to replace original loss function without changing training loop much
def wrap_with_physics_constraints(original_loss_fn, lat_coords, lon_coords, device, 
                                physics_weights=None, enable_physics=True):
    """
    Simple wrapper to add physics constraints to existing loss function
    """
    if not enable_physics:
        return original_loss_fn
        
    if physics_weights is None:
        physics_weights = {
            'data': 1.0,
            'continuity': 0.1,
            'geostrophic': 0.1,
            'energy': 0.05,
            'temporal': 0.02
        }
    
    return PhysicsConstrainedLoss(
        lat_coords, lon_coords, device,
        weights=physics_weights,
        original_loss_fn=original_loss_fn
    )


# Configuration for physics-constrained training
PHYSICS_CONFIG = {
    'enable_physics': True,
    'physics_weights': {
        'data': 1.0,
        'continuity': 0.1,
        'geostrophic': 0.1, 
        'energy': 0.05,
        'temporal': 0.02
    },
    'physics_loss_schedule': {
        'warmup_epochs': 10,  # Start with data loss only
        'ramp_up_epochs': 20, # Gradually increase physics loss weight
    }
}