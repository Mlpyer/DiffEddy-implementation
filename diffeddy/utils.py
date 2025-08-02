import numpy as np
import torch
from torch.utils.data import Sampler

class ERA5Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path,     # str: Path to the dataset files.
                 dataset_mode,     # str: Dataset dataset_mode ('train', 'val', 'test').
                 sample_counts,    # tuple: Total, training, and validation sample counts (total_samples, train_samples, val_samples).
                 dimensions,        # tuple: Dimensions of the dataset (variables, latitude, longitude).
                 lead_time,        # int: Current lead time for forecasting.
                 max_horizon,    # int: Maximum lead time we want to forecast. Used for not going outside dataset
                 norm_factors,     # tuple: Mean and standard deviation for normalization (mean, std_dev).
                 device,           # torch.device: Device on which tensors will be loaded.
                 lead_time_range,  # Range of lead time
                 spinup = 0,       # int: Number of samples to discard at the start for stability.
                 spacing = 1,      # int: Sample selection interval for data reduction.
                 dtype='float32',   # str: Data type of the dataset (default 'float32').
                 conditioning_times=[0,], # list: Times to condition on for forecasting.
                 static_data_path = None, # str: Path to the static data file.
                 random_lead_time = 0, # bool: Whether to use random lead time

                ):
        """
        Initialize a custom Dataset for lazily loading WB samples from a memory-mapped file,
        which allows for efficient data handling without loading the entire dataset into memory.
        """
        self.dataset_path = dataset_path
        self.data_dtype = dtype
        self.device = device

        self.dataset_mode = dataset_mode
        self.n_samples, self.n_train, self.n_val = sample_counts
        self.num_variables, self.n_lat, self.n_lon = dimensions
        self.max_horizon = max_horizon
        self.lead_time = lead_time
        self.spinup = spinup + 24 # Change this if we ever look back more than 24h
        self.spacing = spacing
        self.mean, self.std_dev = norm_factors
        self.t_min, self.t_max, self.delta_t = lead_time_range

        self.static_data_path = static_data_path
        self.static_fields = None

        self.static_vars = 0
        if static_data_path != None:
            self.static_fields = self.load_static_data()      
            self.static_vars = self.static_fields.shape[1]
        
        self.conditioning_times = conditioning_times
        self.input_times = self.num_variables * len(self.conditioning_times)
        self.output_times = self.num_variables * (len(self.lead_time) if isinstance(lead_time, (list, tuple, np.ndarray)) else 1)

        self.index_array = self._generate_indices()

        self.mmap = self.create_mmap()

        self.random_lead_time = random_lead_time


    def create_mmap(self):
        """Creates a memory-mapped array for the dataset to facilitate large data handling."""
        return np.memmap(self.dataset_path, dtype=np.float32, mode='r', shape=(self.n_samples, self.num_variables, self.n_lat, self.n_lon))

    def load_static_data(self):
        """Load and normalize static fields."""
        static_fields = np.load(self.static_data_path)

        min_vals = np.min(static_fields, axis=(1,2))
        max_vals = np.max(static_fields, axis=(1,2))

        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Replace zero range with one to avoid division by zero

        # Apply min-max scaling: (x - min) / (max - min)
        scaled_static_fields = (static_fields - min_vals[:, None, None]) / range_vals[:, None, None]
        return scaled_static_fields

    def _generate_indices(self):
        """Generates indices for dataset partitioning according to the specified dataset_mode."""
        if self.dataset_mode == 'train':
            start, stop = self.spinup, self.n_train
        elif self.dataset_mode == 'val':
            start, stop = self.spinup + self.n_train, self.n_train + self.n_val
        elif self.dataset_mode == 'test':
            start, stop = self.spinup + self.n_train + self.n_val, self.n_samples

        return np.arange(start, stop - self.max_horizon)[::self.spacing]

    def set_lead_time(self, lead_time):
        """ Updates the lead time lead_time for generating future or past indices."""
        self.lead_time = lead_time

    def set_lead_time_range(self, lead_time_range):
        self.t_min, self.t_max, self.delta_t = lead_time_range

    def get_lead_time(self):
        if self.random_lead_time:
            return self.t_min + self.delta_t * torch.randint(0, 1 + (self.t_max - self.t_min) // self.delta_t, (1,), device=self.device)[0]
        return self.lead_time

    def __len__(self):
        """Returns the number of samples available in the dataset based on the computed indices."""
        return self.index_array.shape[0]

    def __getitem__(self, idx):
        """Retrieves a sample and its corresponding future or past state from the dataset."""
        start_index = self.index_array[idx]
        lead_times = self.get_lead_time()

        x_index = start_index + self.conditioning_times
        y_index = start_index + lead_times

        X_sample = self.mmap[x_index, :].astype(self.data_dtype)
        Y_sample = self.mmap[y_index, :].astype(self.data_dtype)

        X_sample = (X_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]
        Y_sample = (Y_sample - self.mean[None, :, None, None]) / self.std_dev[None, :, None, None]

        X_sample = torch.tensor(X_sample, dtype=torch.float32).view(self.input_times, self.n_lat, self.n_lon)
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32).view(self.output_times, self.n_lat, self.n_lon)

        if self.static_vars != 0:
            X_sample = np.concatenate([X_sample, self.static_fields], axis=0)

        return X_sample, Y_sample, lead_times


def get_uniform_t_dist_fn(t_min, t_max, delta_t):
    """ Create the update function """

    def uniform_t_dist(dataset):
        new_lead_time = t_min + delta_t * np.random.randint(0, 1 + (t_max - t_min) // delta_t)
        dataset.set_lead_time(new_lead_time)
    
    return uniform_t_dist

class DynamicKBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last, t_update_callback, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.t_update_callback = t_update_callback
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        # Shuffle indices at the beginning of each epoch if required
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        batch = []
        for idx in self.indices:
            if len(batch) == self.batch_size:
                self.t_update_callback(self.dataset)  # Update `lead_time` before yielding the batch
                yield batch
                batch = []
            batch.append(idx)
        if batch and not self.drop_last:
            self.t_update_callback(self.dataset)  # Update `lead_time` for the last batch if not dropping it
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size


import os
import torch
import numpy as np
import xarray as xr
from typing import Tuple, List, Union, Optional
from datetime import datetime, timedelta
import pandas as pd

class OceanDatasetERA5Style(torch.utils.data.Dataset):
    def __init__(self,
                 dataset_path: str,           # Path to the netcdf files directory or single file
                 dataset_mode: str,           # Dataset mode ('train', 'val', 'test')
                 dimensions: Tuple[int, int, int],     # (variables, latitude, longitude)
                 lead_time: Union[int, List[int]],     # Current lead time for forecasting
                 max_horizon: int,            # Maximum lead time for forecasting
                 norm_factors: Optional[Tuple[np.ndarray, np.ndarray]] = None,  # (mean, std_dev)
                 device: torch.device = torch.device('cpu'),
                 lead_time_range: Tuple[int, int, int] = (1, 7, 1),  # (t_min, t_max, delta_t)
                 spinup: int = 0,             # Number of samples to discard at start
                 spacing: int = 1,            # Sample selection interval
                 dtype: str = 'float32',      # Data type
                 conditioning_times: List[int] = [0],  # Times to condition on
                 variables: List[str] = ['uo_oras', 'vo_oras'],  # Variables to load
                 random_lead_time: bool = False,  # Whether to use random lead time
                 depth_level: Optional[int] = None,   # Specific depth level to extract (None for all)
                 # 时间分割参数
                 time_split_mode: str = 'index',  # 'index' or 'date'
                 # 索引分割模式参数 (仅在 time_split_mode='index' 时使用)
                 sample_counts: Optional[Tuple[int, int, int]] = None,  # (total_samples, train_samples, val_samples)
                 # 日期分割模式参数 (仅在 time_split_mode='date' 时使用)
                 train_date_range: Optional[Tuple[str, str]] = None,  # ('YYYY-MM-DD', 'YYYY-MM-DD')
                 val_date_range: Optional[Tuple[str, str]] = None,    # ('YYYY-MM-DD', 'YYYY-MM-DD')
                 test_date_range: Optional[Tuple[str, str]] = None,   # ('YYYY-MM-DD', 'YYYY-MM-DD')
                ):
        """
        Initialize Ocean Dataset using ERA5-style memory mapping approach.
        
        Args:
            dataset_path: Path to netcdf files or directory containing them
            dataset_mode: 'train', 'val', or 'test'
            dimensions: (variables, latitude, longitude)
            lead_time: Lead time(s) for forecasting
            max_horizon: Maximum forecasting horizon
            norm_factors: Pre-computed normalization factors (mean, std)
            device: Device to load tensors on
            lead_time_range: (min_lead_time, max_lead_time, step)
            spinup: Initial samples to skip
            spacing: Sampling interval
            dtype: Data type for tensors
            conditioning_times: Input time steps relative to current time
            variables: List of variable names to load
            random_lead_time: Whether to randomize lead time
            depth_level: Specific depth index to extract (0 for surface)
            time_split_mode: 'index' for traditional index-based split, 'date' for date-based split
            sample_counts: (total_samples, train_samples, val_samples) - only required for 'index' mode
            train_date_range: Training data date range - only required for 'date' mode
            val_date_range: Validation data date range - only required for 'date' mode
            test_date_range: Test data date range - only required for 'date' mode
        """
        self.dataset_path = dataset_path
        self.data_dtype = dtype
        self.device = device
        self.variables = variables
        self.depth_level = depth_level

        self.dataset_mode = dataset_mode
        self.num_variables, self.n_lat, self.n_lon = dimensions
        self.max_horizon = max_horizon
        self.lead_time = lead_time
        self.spinup = spinup
        self.spacing = spacing
        self.t_min, self.t_max, self.delta_t = lead_time_range

        self.conditioning_times = conditioning_times
        self.input_times = self.num_variables * len(self.conditioning_times)
        self.output_times = self.num_variables * (len(lead_time) if isinstance(lead_time, (list, tuple, np.ndarray)) else 1)
        
        self.random_lead_time = random_lead_time
        
        # 时间分割相关参数验证和设置
        self.time_split_mode = time_split_mode
        
        if self.time_split_mode == 'index':
            if sample_counts is None:
                raise ValueError("sample_counts must be provided when using 'index' time_split_mode")
            self.n_samples, self.n_train, self.n_val = sample_counts
            self.train_date_range = None
            self.val_date_range = None 
            self.test_date_range = None
        elif self.time_split_mode == 'date':
            if None in [train_date_range, val_date_range, test_date_range]:
                raise ValueError("All date ranges (train_date_range, val_date_range, test_date_range) must be provided when using 'date' time_split_mode")
            self.train_date_range = train_date_range
            self.val_date_range = val_date_range
            self.test_date_range = test_date_range
            # sample_counts will be computed after loading data
            self.n_samples = None
            self.n_train = None
            self.n_val = None
        else:
            raise ValueError(f"time_split_mode must be 'index' or 'date', got {time_split_mode}")

        # Load data and compute normalization factors
        self.data_array, self.mean, self.std_dev, self.time_coords = self._load_and_prepare_data(norm_factors)
        
        # Generate indices for the specific dataset mode
        self.index_array = self._generate_indices()

    def _load_and_prepare_data(self, norm_factors: Optional[Tuple[np.ndarray, np.ndarray]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load ocean data from netcdf files and prepare for memory mapping style access.
        """
        # Load xarray dataset
        if os.path.isdir(self.dataset_path):
            # Multiple files in directory
            nc_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.nc')]
            if not nc_files:
                raise ValueError(f"No .nc files found in {self.dataset_path}")
            file_paths = [os.path.join(self.dataset_path, f) for f in nc_files]
            ds = xr.open_mfdataset(file_paths, combine='by_coords')
        else:
            # Single file
            ds = xr.open_dataset(self.dataset_path)

        # 提取时间坐标信息
        time_coords = pd.to_datetime(ds.time.values)
        
        # Extract variables and combine them
        data_arrays = []
        for var in self.variables:
            if var not in ds.data_vars:
                raise ValueError(f"Variable {var} not found in dataset")
            
            var_data = ds[var].values  # Shape: (time, depth, lat, lon)
            
            # Extract specific depth level if specified
            if self.depth_level is not None:
                if self.depth_level >= var_data.shape[1]:
                    raise ValueError(f"Depth level {self.depth_level} exceeds available depths {var_data.shape[1]}")
                var_data = var_data[:, self.depth_level:self.depth_level+1, :, :]
            
            # Reshape to (time, depth*var, lat, lon) - treating depth as part of variable dimension
            var_data = var_data.reshape(var_data.shape[0], -1, var_data.shape[2], var_data.shape[3])
            data_arrays.append(var_data)

        # Concatenate all variables along the variable dimension
        # Final shape: (time, variables*depth, lat, lon)
        combined_data = np.concatenate(data_arrays, axis=1)
        
        # Update dimensions based on actual data
        actual_samples = combined_data.shape[0]
        if self.time_split_mode == 'index':
            # 验证提供的 sample_counts 是否合理
            if self.n_samples > actual_samples:
                print(f"Warning: Specified n_samples ({self.n_samples}) exceeds actual samples ({actual_samples}). Using actual samples.")
                self.n_samples = actual_samples
        else:
            # 对于 date 模式，设置总样本数
            self.n_samples = actual_samples
            
        actual_vars = combined_data.shape[1]
        if actual_vars != self.num_variables:
            print(f"Warning: Expected {self.num_variables} variables, got {actual_vars}. Updating dimensions.")
            self.num_variables = actual_vars
            self.input_times = self.num_variables * len(self.conditioning_times)
            self.output_times = self.num_variables * (len(self.lead_time) if isinstance(self.lead_time, (list, tuple, np.ndarray)) else 1)

        # Compute or use provided normalization factors
        if norm_factors is None:
            # Compute mean and std across time and spatial dimensions, keep variable dimension
            mean = np.mean(combined_data, axis=(0, 2, 3), keepdims=True)  # Shape: (1, vars, 1, 1)
            std_dev = np.std(combined_data, axis=(0, 2, 3), keepdims=True)  # Shape: (1, vars, 1, 1)
            # Avoid division by zero
            std_dev = np.where(std_dev == 0, 1, std_dev)
        else:
            mean, std_dev = norm_factors

        return combined_data, mean, std_dev, time_coords

    def _generate_indices(self) -> np.ndarray:
        """Generate indices for dataset partitioning according to the specified dataset_mode and time_split_mode."""
        
        if self.time_split_mode == 'date':
            return self._generate_date_based_indices()
        else:
            return self._generate_index_based_indices()
    
    def _generate_index_based_indices(self) -> np.ndarray:
        """Generate indices using traditional index-based splitting."""
        if self.dataset_mode == 'train':
            start, stop = self.spinup, min(self.n_train, self.n_samples)
        elif self.dataset_mode == 'val':
            start = self.spinup + self.n_train
            stop = min(start + self.n_val, self.n_samples)
        elif self.dataset_mode == 'test':
            start = self.spinup + self.n_train + self.n_val
            stop = self.n_samples
        else:
            raise ValueError(f"Invalid dataset_mode: {self.dataset_mode}")

        # Ensure we don't go beyond available samples and account for max_horizon
        start = max(start, self.spinup)
        stop = min(stop, self.n_samples - self.max_horizon)
        
        if start >= stop:
            raise ValueError(f"No valid samples for {self.dataset_mode} mode. Check your sample_counts and data size.")
        
        return np.arange(start, stop)[::self.spacing]
    
    def _generate_date_based_indices(self) -> np.ndarray:
        """Generate indices using date-based splitting."""
        if self.dataset_mode == 'train':
            date_range = self.train_date_range
        elif self.dataset_mode == 'val':
            date_range = self.val_date_range
        elif self.dataset_mode == 'test':
            date_range = self.test_date_range
        else:
            raise ValueError(f"Invalid dataset_mode: {self.dataset_mode}")
        
        if date_range is None:
            raise ValueError(f"Date range not specified for {self.dataset_mode} mode")
        
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Find indices corresponding to the date range
        mask = (self.time_coords >= start_date) & (self.time_coords <= end_date)
        valid_indices = np.where(mask)[0]
        
        # Apply spinup and ensure we don't exceed max_horizon
        valid_indices = valid_indices[valid_indices >= self.spinup]
        valid_indices = valid_indices[valid_indices < len(self.time_coords) - self.max_horizon]
        
        # Apply spacing
        return valid_indices[::self.spacing]

    def set_lead_time(self, lead_time: Union[int, List[int]]):
        """Updates the lead time for generating future indices."""
        self.lead_time = lead_time
        self.output_times = self.num_variables * (len(lead_time) if isinstance(lead_time, (list, tuple, np.ndarray)) else 1)

    def set_lead_time_range(self, lead_time_range: Tuple[int, int, int]):
        """Update the lead time range for random sampling."""
        self.t_min, self.t_max, self.delta_t = lead_time_range

    def get_lead_time(self) -> Union[int, torch.Tensor]:
        """Get lead time - either fixed or random."""
        if self.random_lead_time:
            return self.t_min + self.delta_t * torch.randint(
                0, 1 + (self.t_max - self.t_min) // self.delta_t, 
                (1,), device=self.device
            )[0]
        return self.lead_time

    def __len__(self) -> int:
        """Returns the number of samples available in the dataset."""
        return len(self.index_array)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Union[int, torch.Tensor], dict]:
        """
        Retrieves a sample and its corresponding future state from the dataset.
        
        Returns:
            X_sample: Input tensor of shape (input_times, n_lat, n_lon)
            Y_sample: Target tensor of shape (output_times, n_lat, n_lon)  
            lead_times: The lead time(s) used for this sample
            metadata: Dictionary containing additional information about the sample
        """
        start_index = self.index_array[idx]
        lead_times = self.get_lead_time()

        # Get input indices
        x_indices = start_index + np.array(self.conditioning_times)
        
        # Get output indices - 修复这里的处理逻辑
        if isinstance(lead_times, torch.Tensor):
            # 如果是 torch.Tensor，转换为 numpy 数组
            lead_times_np = lead_times.cpu().numpy()
            if lead_times_np.ndim == 0:  # 如果是标量张量
                y_indices = np.array([start_index + int(lead_times_np)])
            else:
                y_indices = start_index + lead_times_np
        else:
            # 如果是其他类型
            if isinstance(lead_times, (list, tuple, np.ndarray)):
                y_indices = start_index + np.array(lead_times)
            else:
                # 标量情况
                y_indices = np.array([start_index + lead_times])

        # Ensure indices are within bounds
        x_indices = np.clip(x_indices, 0, len(self.time_coords) - 1)
        y_indices = np.clip(y_indices, 0, len(self.time_coords) - 1)

        # Extract data
        X_sample = self.data_array[x_indices, :].astype(self.data_dtype)  # (len(conditioning_times), vars, lat, lon)
        Y_sample = self.data_array[y_indices, :].astype(self.data_dtype)  # (len(lead_times), vars, lat, lon)

        # Normalize
        X_sample = (X_sample - self.mean) / self.std_dev
        Y_sample = (Y_sample - self.mean) / self.std_dev

        # Reshape: flatten time and variable dimensions
        X_sample = X_sample.reshape(-1, self.n_lat, self.n_lon)  # (input_times, lat, lon)
        Y_sample = Y_sample.reshape(-1, self.n_lat, self.n_lon)  # (output_times, lat, lon)

        # Convert to tensors
        X_sample = torch.tensor(X_sample, dtype=torch.float32, device=self.device)
        Y_sample = torch.tensor(Y_sample, dtype=torch.float32, device=self.device)

        # # 构建元数据信息 - 确保索引是数组
        # metadata = {
        #     'input_dates': [self.time_coords[i].strftime('%Y-%m-%d %H:%M:%S') for i in x_indices],
        #     'output_dates': [self.time_coords[i].strftime('%Y-%m-%d %H:%M:%S') for i in y_indices],
        #     'start_index': start_index,
        #     'dataset_mode': self.dataset_mode,
        #     'time_split_mode': self.time_split_mode,
        #     'sample_idx': idx
        # }

        return X_sample, Y_sample, lead_times

    def get_normalization_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the normalization factors (mean, std) for external use."""
        return self.mean, self.std_dev

    def get_data_info(self) -> dict:
        """Return information about the loaded data."""
        info = {
            'variables': self.variables,
            'shape': (self.n_samples, self.num_variables, self.n_lat, self.n_lon),
            'dataset_mode': self.dataset_mode,
            'available_samples': len(self.index_array),
            'normalization_factors_shape': (self.mean.shape, self.std_dev.shape),
            'time_split_mode': self.time_split_mode,
            'time_range': {
                'start': self.time_coords[0].strftime('%Y-%m-%d %H:%M:%S'),
                'end': self.time_coords[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'total_timesteps': len(self.time_coords)
            }
        }
        
        if self.time_split_mode == 'index':
            info['sample_counts'] = {
                'total': self.n_samples,
                'train': self.n_train,
                'val': self.n_val,
                'test': self.n_samples - self.n_train - self.n_val
            }
        elif self.time_split_mode == 'date':
            info['date_ranges'] = {
                'train': self.train_date_range,
                'val': self.val_date_range,
                'test': self.test_date_range
            }
            
            # 添加实际使用的时间范围
            if len(self.index_array) > 0:
                actual_start = self.time_coords[self.index_array[0]]
                actual_end = self.time_coords[self.index_array[-1]]
                info['actual_time_range'] = {
                    'start': actual_start.strftime('%Y-%m-%d %H:%M:%S'),
                    'end': actual_end.strftime('%Y-%m-%d %H:%M:%S')
                }
        
        return info
    
    def get_time_info(self, idx: int) -> dict:
        """Get detailed time information for a specific sample index."""
        if idx >= len(self.index_array):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self.index_array)} samples")
        
        start_index = self.index_array[idx]
        lead_times = self.get_lead_time()
        
        x_indices = start_index + np.array(self.conditioning_times)
        
        # 修复这里的处理逻辑
        if isinstance(lead_times, torch.Tensor):
            lead_times_np = lead_times.cpu().numpy()
            if lead_times_np.ndim == 0:
                y_indices = np.array([start_index + int(lead_times_np)])
            else:
                y_indices = start_index + lead_times_np
        else:
            if isinstance(lead_times, (list, tuple, np.ndarray)):
                y_indices = start_index + np.array(lead_times)
            else:
                y_indices = np.array([start_index + lead_times])
        
        return {
            'sample_idx': idx,
            'start_index': start_index,
            'input_indices': x_indices.tolist(),
            'output_indices': y_indices.tolist(),
            'input_dates': [self.time_coords[i].strftime('%Y-%m-%d %H:%M:%S') for i in x_indices],
            'output_dates': [self.time_coords[i].strftime('%Y-%m-%d %H:%M:%S') for i in y_indices],
            'lead_times': lead_times
        }


# Example usage
if __name__ == "__main__":
    # Example configuration for your ocean data
    dataset = OceanDatasetERA5Style(
        dataset_path="/root/autodl-tmp/test_sourth_ocean_uv",  # or directory with multiple .nc files
        dataset_mode='test',
        time_split_mode='date',
        train_date_range=('2000-01-01', '2021-12-31'),  # 必需
        val_date_range=('2022-01-01', '2022-12-31'),    # 必需
        test_date_range=('2023-01-01', '2023-12-31'),
        # sample_counts=(365, 292, 73),  # Assuming 80% train, 20% val for 365 days
        dimensions=(2, 288, 196),  # 2 variables (uo_oras, vo_oras), 32 lat, 64 lon  
        lead_time=1,  # Predict 1 day ahead
        max_horizon=30,  # Maximum 7 days forecasting
        lead_time_range=(1, 7, 1),  # Lead time from 1 to 7 days
        conditioning_times=[0],  # Use current time step as input
        variables=['uo', 'vo'],
        depth_level=0,  # Use surface level only (index 0)
        random_lead_time=False,
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    )
    
    print("Dataset info:", dataset.get_data_info())
    print("Dataset length:", len(dataset))
    
    # Test getting a sample
    x, y, lead_time = dataset[0]
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Lead time: {lead_time}")

