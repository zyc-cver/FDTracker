import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from scipy.spatial.transform import Rotation as R
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix


def _gaussian_kernel1d(sigma, order, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    
    Args:
        sigma: Standard deviation for Gaussian kernel
        order: Order of the kernel (0 for smoothing, 1 for derivative)
        radius: Radius of the kernel (typically 4*sigma)
        
    Returns:
        1D Gaussian kernel
    """
    if order < 0:
        raise ValueError('order must be non-negative')
    radius = int(radius)
    x = np.arange(-radius, radius + 1)
    sigma2 = sigma * sigma
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()

    if order == 0:
        return phi_x
    else:
        # Gaussian derivative functions
        return (-(x/sigma2) * phi_x)


def gaussian_smooth_1d(x, sigma=3, dim=-1):
    """
    Apply Gaussian smoothing along a specific dimension for general tensors
    
    Args:
        x: Input tensor
        sigma: Standard deviation for Gaussian kernel
        dim: Dimension along which to smooth
        
    Returns:
        Smoothed tensor
    """
    kernel_smooth = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
    kernel_smooth = torch.from_numpy(kernel_smooth).float()[None, None].to(x.device)  # (1, 1, K)
    rad = kernel_smooth.size(-1) // 2

    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, "... f -> (...) 1 f")  # (NB, 1, f)
    x = F.pad(x[None], (rad, rad, 0, 0), mode="replicate")[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)  # (..., f)
    x = x.transpose(-1, dim)
    return x


def smooth_rotation_matrices_slerp(rot_matrices, sigma=3):
    """
    Smooth rotation matrices using spherical linear interpolation (SLERP)
    
    Args:
        rot_matrices: Rotation matrices [B, T, 3, 3] or [T, 3, 3]
        sigma: Smoothing strength (higher = more smoothing)
        
    Returns:
        Smoothed rotation matrices with same shape
    """
    original_shape = rot_matrices.shape
    if len(original_shape) == 3:
        # Add batch dimension if not present
        rot_matrices = rot_matrices.unsqueeze(0)  # [1, T, 3, 3]
    
    B, T, _, _ = rot_matrices.shape
    device = rot_matrices.device
    
    # Convert to scipy format for SLERP
    smoothed_matrices = torch.zeros_like(rot_matrices)
    
    for b in range(B):
        # Convert to scipy Rotation objects
        matrices_np = rot_matrices[b].detach().cpu().numpy()  # [T, 3, 3]
        rotations = R.from_matrix(matrices_np)
        
        # Create time indices
        times = np.arange(T)
        
        # Apply Gaussian weights for smoothing
        kernel = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
        half_window = len(kernel) // 2
        
        smoothed_rots = []
        for t in range(T):
            # Get weights for current time
            start_idx = max(0, t - half_window)
            end_idx = min(T, t + half_window + 1)
            
            # Adjust kernel indices
            kernel_start = max(0, half_window - t)
            kernel_end = kernel_start + (end_idx - start_idx)
            
            weights = kernel[kernel_start:kernel_end]
            weights = weights / weights.sum()  # Normalize
            
            # Weighted average using SLERP
            if len(weights) == 1:
                smoothed_rots.append(rotations[t])
            else:
                # Use first rotation as reference
                ref_rot = rotations[start_idx]
                weighted_rot = ref_rot
                
                for i, w in enumerate(weights):
                    if i == 0:
                        continue
                    curr_rot = rotations[start_idx + i]
                    # SLERP between current result and new rotation
                    weighted_rot = weighted_rot.slerp(curr_rot, w / (w + weights[:i+1].sum()))
                
                smoothed_rots.append(weighted_rot)
        
        # Convert back to matrices
        smoothed_rotation = R.from_quat([r.as_quat() for r in smoothed_rots])
        smoothed_matrices[b] = torch.from_numpy(smoothed_rotation.as_matrix()).float().to(device)
    
    # Restore original shape
    if len(original_shape) == 3:
        smoothed_matrices = smoothed_matrices.squeeze(0)
    
    return smoothed_matrices


def smooth_rotation_matrices_quat(rot_matrices, sigma=3):
    """
    Smooth rotation matrices by converting to quaternions and smoothing
    
    Args:
        rot_matrices: Rotation matrices [B, T, 3, 3] or [T, 3, 3]
        sigma: Smoothing strength
        
    Returns:
        Smoothed rotation matrices
    """
    original_shape = rot_matrices.shape
    if len(original_shape) == 3:
        rot_matrices = rot_matrices.unsqueeze(0)  # [1, T, 3, 3]
    
    B, T, _, _ = rot_matrices.shape
    
    # Convert to 6D representation for smoothing
    rot_6d = matrix_to_rotation_6d(rot_matrices.reshape(-1, 3, 3))  # [B*T, 6]
    rot_6d = rot_6d.reshape(B, T, 6)  # [B, T, 6]
    
    # Smooth each 6D component
    smoothed_6d = gaussian_smooth_1d(rot_6d, sigma=sigma, dim=1)  # [B, T, 6]
    
    # Convert back to rotation matrices
    smoothed_matrices = rotation_6d_to_matrix(smoothed_6d.reshape(-1, 6))  # [B*T, 3, 3]
    smoothed_matrices = smoothed_matrices.reshape(B, T, 3, 3)  # [B, T, 3, 3]
    
    # Restore original shape
    if len(original_shape) == 3:
        smoothed_matrices = smoothed_matrices.squeeze(0)
    
    return smoothed_matrices


def smooth_pose_6d(pose_6d, sigma=3):
    """
    Smooth 6D pose representations
    
    Args:
        pose_6d: 6D pose tensor [B, T, J*6] or [B, T, J, 6]
        sigma: Smoothing strength
        
    Returns:
        Smoothed 6D poses
    """
    return gaussian_smooth_1d(pose_6d, sigma=sigma, dim=1)


def smooth_translation(translation, sigma=3):
    """
    Smooth translation vectors
    
    Args:
        translation: Translation tensor [B, T, 3]
        sigma: Smoothing strength
        
    Returns:
        Smoothed translations
    """
    return gaussian_smooth_1d(translation, sigma=sigma, dim=1)


def smooth_betas(betas, sigma=3):
    """
    Smooth shape parameters (betas)
    
    Args:
        betas: Shape parameters [B, T, 10]
        sigma: Smoothing strength
        
    Returns:
        Smoothed betas
    """
    return gaussian_smooth_1d(betas, sigma=sigma, dim=1)


# Backward compatibility
def gaussian_smooth(x, sigma=3, dim=-1):
    """
    Backward compatibility function
    """
    return gaussian_smooth_1d(x, sigma=sigma, dim=dim)
