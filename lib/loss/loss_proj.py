import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from core.config import cfg


def get_camera_intrinsics_matrix():
    """
    Create camera intrinsics matrix from config values
    
    Returns:
        K: Camera intrinsics matrix as numpy array [3, 3]
    """
    K = np.array([[cfg.CAMERA.fx, 0, cfg.CAMERA.cx],
                  [0, cfg.CAMERA.fy, cfg.CAMERA.cy],
                  [0, 0, 1.]])
    return K


def compute_human_vertices_loss(pred_pose6d, pred_betas, gt_pose6d, gt_betas, smpl_layer, criterion):
    """
    Compute human vertices loss by forward passing through SMPL model
    
    Args:
        pred_pose6d: Predicted pose in 6D rotation format [B*T, 22, 6]
        pred_betas: Predicted shape parameters [B*T, 10]
        gt_pose6d: Ground truth pose in 6D rotation format [B*T, 22, 6]
        gt_betas: Ground truth shape parameters [B*T, 10]
        smpl_layer: SMPL model layer
        criterion: Loss criterion (e.g., L1Loss)
        
    Returns:
        vertices_loss: L1 loss between predicted and ground truth vertices
    """
    # Convert 6D rotation to axis-angle for SMPL
    h_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(pred_pose6d.reshape(-1, 22, 6))).reshape(-1, 66)
    gt_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(gt_pose6d.reshape(-1, 22, 6))).reshape(-1, 66)
    
    # Reshape betas
    h_betas = pred_betas.reshape(-1, 10)
    gt_betas_reshaped = gt_betas.reshape(-1, 10)
    
    # Forward pass through SMPL to get vertices
    h_verts = smpl_layer.forward(
        betas=h_betas,
        body_pose=h_pose_aa[:, 3:],
        global_orient=h_pose_aa[:, :3]
    ).vertices
    
    gt_verts = smpl_layer.forward(
        betas=gt_betas_reshaped,
        body_pose=gt_pose_aa[:, 3:],
        global_orient=gt_pose_aa[:, :3]
    ).vertices
    
    # Compute L1 loss
    vertices_loss = criterion(h_verts, gt_verts)
    
    return vertices_loss


def compute_human_vertices_loss_from_verts(pred_verts, gt_verts, criterion):
    """
    Compute human vertices loss from pre-computed vertices
    
    Args:
        pred_verts: Predicted human vertices [B*T, V, 3]
        gt_verts: Ground truth human vertices [B*T, V, 3]
        criterion: Loss criterion (e.g., L1Loss)
        
    Returns:
        vertices_loss: Loss between predicted and ground truth vertices
    """
    return criterion(pred_verts, gt_verts)

def project_keypoints_to_2d(
    keypoints_3d,     
    cam_trans,       
    K = None,        
    img_size = None,
    eps = 1e-6
):
    BT, J, _ = keypoints_3d.shape

    if img_size is None:
        img_w, img_h = cfg.CAMERA.original_img_size
    else:
        img_w, img_h = img_size

    if K is None:
        fx = cfg.CAMERA.fx
        fy = cfg.CAMERA.fy
        cx = cfg.CAMERA.cx
        cy = cfg.CAMERA.cy

        K = keypoints_3d.new_zeros((BT, 3, 3))
        K[:, 0, 0] = fx
        K[:, 0, 2] = cx
        K[:, 1, 1] = fy
        K[:, 1, 2] = cy
        K[:, 2, 2] = 1.0
    else:
        if K.ndim == 2 and K.shape == (3, 3):
            K = K.unsqueeze(0).expand(BT, -1, -1).contiguous()

    K = K.to(dtype=keypoints_3d.dtype, device=keypoints_3d.device)

    keypoints_cam = keypoints_3d + cam_trans.unsqueeze(1)  # [BT, J, 3]

    Z = keypoints_cam[..., 2].clamp_min(eps)               # [BT, J]
    x = keypoints_cam[..., 0] / Z                          # [BT, J]
    y = keypoints_cam[..., 1] / Z                          # [BT, J]

    ones = torch.ones_like(x)
    xy1 = torch.stack([x, y, ones], dim=-1)                # [BT, J, 3]
    uvw = torch.einsum('bij,bkj->bki', K, xy1)             # [BT, J, 3]
    u = uvw[..., 0]
    v = uvw[..., 1]

    u_norm = u / float(img_w)
    v_norm = v / float(img_h)

    projected_2d = torch.stack([u_norm, v_norm], dim=-1)   # [BT, J, 2]
    return projected_2d


def get_keypoint_weights(num_joints, end_effector_weight=3.0):
    """
    Get keypoint weights with higher weights for end-effectors (hands and feet)
    
    Args:
        num_joints: Total number of joints from SMPL
        end_effector_weight: Weight multiplier for end-effector joints
        
    Returns:
        weights: Tensor of weights for each joint [num_joints]
    """
    # Initialize all weights to 1.0
    weights = torch.ones(num_joints)
    
    # Define end-effector joint indices for SMPL (hands and feet)
    # Based on standard SMPL joint ordering
    if num_joints >= 24:
        # Standard SMPL joints (24 joints)
        end_effector_indices = [
            7,   # left_ankle
            8,   # right_ankle
            20,  # left_wrist
            21,  # right_wrist
        ]
        
        # If we have more joints, also include toe and finger tips
        if num_joints >= 45:
            # Extended SMPL-H joints (include hands)
            hand_indices = [
                22, 23,  # left_index1, left_index2
                24, 25,  # left_index3, left_middle1
                26, 27,  # left_middle2, left_middle3
                28, 29,  # left_pinky1, left_pinky2
                30, 31,  # left_pinky3, left_ring1
                32, 33,  # left_ring2, left_ring3
                34, 35,  # left_thumb1, left_thumb2
                36,      # left_thumb3
                37, 38,  # right_index1, right_index2
                39, 40,  # right_index3, right_middle1
                41, 42,  # right_middle2, right_middle3
                43, 44,  # right_pinky1, right_pinky2
            ]
            # Add finger tips with medium weight
            end_effector_indices.extend([25, 27, 31, 33, 36, 40, 42, 44])  # finger tips
            
        # Set higher weights for end-effectors
        for idx in end_effector_indices:
            if idx < num_joints:
                weights[idx] = end_effector_weight
    
    return weights


def compute_human_keypoint_projection_loss(pred_pose6d, pred_betas, pred_trans, gt_keypoints_2d, 
                                          smpl_layer, criterion, cam_intrinsics, valid_joints=None,
                                          end_effector_weight=3.0):
    """
    Compute 2D keypoint projection loss for human pose alignment using all SMPL keypoints
    with higher weights for end-effectors (hands and feet)
    
    Args:
        pred_pose6d: Predicted pose in 6D rotation format [B, T, 132]
        pred_betas: Predicted shape parameters [B, T, 10]
        pred_trans: Predicted camera translation [B, T, 3]
        gt_keypoints_2d: Ground truth 2D keypoints [B, T, J, 2] or [B, T, J, 3] with confidence
        smpl_layer: SMPL model layer
        criterion: Loss criterion (e.g., MSELoss)
        use_intrinsics: Whether to use camera intrinsics from config
        valid_joints: Optional valid joint indices (if None, use all joints)
        end_effector_weight: Weight multiplier for end-effector joints (hands and feet)
        
    Returns:
        keypoint_loss: Weighted MSE loss between projected 2D keypoints
    """
    B, T = pred_pose6d.shape[:2]
    
    # Reshape for SMPL forward pass
    pred_pose6d_flat = pred_pose6d.reshape(-1, 22, 6)
    pred_betas_flat = pred_betas.reshape(-1, 10)
    pred_trans_flat = pred_trans.reshape(-1, 3)
    
    # Convert 6D rotation to axis-angle for SMPL
    pred_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(pred_pose6d_flat)).reshape(-1, 66)
    
    # Forward pass through SMPL to get 3D keypoints (joints)
    smpl_output = smpl_layer.forward(
        betas=pred_betas_flat,
        body_pose=pred_pose_aa[:, 3:],
        global_orient=pred_pose_aa[:, :3]
    )
    pred_joints_3d = smpl_output.joints  # [B*T, J, 3] - Use all SMPL joints
    
    # Get the number of joints from SMPL output
    num_joints = pred_joints_3d.shape[1]
    
    # Get keypoint weights with higher weights for end-effectors
    keypoint_weights = get_keypoint_weights(num_joints, end_effector_weight).to(pred_joints_3d.device)
    
    # Apply valid joints mask if provided, otherwise use all joints
    if valid_joints is not None:
        pred_joints_3d = pred_joints_3d[:, valid_joints, :]
        keypoint_weights = keypoint_weights[valid_joints]
        num_joints = len(valid_joints)
    
    # Project 3D keypoints to 2D
    pred_keypoints_2d = project_keypoints_to_2d(pred_joints_3d, pred_trans_flat, cam_intrinsics)
    
    # Reshape back to [B, T, num_joints, 2]
    pred_keypoints_2d = pred_keypoints_2d.reshape(B, T, num_joints, 2)
    
    # Handle ground truth keypoints
    if gt_keypoints_2d.shape[-1] == 3:
        gt_kpts_2d = gt_keypoints_2d[..., :2]
        confidence = gt_keypoints_2d[..., 2:]
    else:
        gt_kpts_2d = gt_keypoints_2d
        confidence = None
    
    # Ensure GT keypoints match the number of predicted joints
    gt_num_joints = gt_kpts_2d.shape[2]
    if gt_num_joints != num_joints:
        if gt_num_joints > num_joints:
            # GT has more joints, select the same subset
            if valid_joints is not None:
                gt_kpts_2d = gt_kpts_2d[:, :, valid_joints, :]
                if confidence is not None:
                    confidence = confidence[:, :, valid_joints, :]
            else:
                # Truncate to match predicted joints
                gt_kpts_2d = gt_kpts_2d[:, :, :num_joints, :]
                if confidence is not None:
                    confidence = confidence[:, :, :num_joints, :]
        else:
            # GT has fewer joints, pad with zeros
            pad_size = num_joints - gt_num_joints
            gt_kpts_2d = F.pad(gt_kpts_2d, (0, 0, 0, pad_size), mode='constant', value=0)
            if confidence is not None:
                confidence = F.pad(confidence, (0, 0, 0, pad_size), mode='constant', value=0)
    
    # Apply confidence weights if available
    if confidence is not None:
        pred_keypoints_2d = pred_keypoints_2d * confidence
        gt_kpts_2d = gt_kpts_2d * confidence
    
    # Compute per-keypoint MSE loss without reduction
    keypoint_errors = (pred_keypoints_2d - gt_kpts_2d) ** 2  # [B, T, num_joints, 2]
    keypoint_errors = keypoint_errors.mean(dim=-1)  # [B, T, num_joints] - average over x,y
    
    # Apply keypoint weights - reshape weights to match [B, T, num_joints]
    keypoint_weights = keypoint_weights.view(1, 1, -1).expand(B, T, -1)
    weighted_errors = keypoint_errors * keypoint_weights  # [B, T, num_joints]
    
    # Compute final weighted loss
    keypoint_loss = weighted_errors.mean()
    
    return keypoint_loss


def compute_human_projection_loss(pred_pose6d, pred_betas, pred_trans, gt_pose6d, gt_betas, gt_trans, 
                                 smpl_layer, criterion, cam_intrinsics, focal_length=None, img_size=None, use_intrinsics=True,
                                 end_effector_weight=3.0):
    """
    Compute 2D projection loss using all SMPL keypoints for alignment
    with higher weights for end-effectors (hands and feet)
    
    Projects both predicted and GT SMPL poses to 2D and computes alignment loss
    """
    B, T = pred_pose6d.shape[:2]
    
    # Reshape GT data for SMPL forward pass
    gt_pose6d_flat = gt_pose6d.reshape(-1, 22, 6)
    gt_betas_flat = gt_betas.reshape(-1, 10)
    gt_trans_flat = gt_trans.reshape(-1, 3)
    
    # Convert 6D rotation to axis-angle for SMPL
    gt_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(gt_pose6d_flat)).reshape(-1, 66)
    
    # Forward pass through SMPL to get GT 3D keypoints
    gt_smpl_output = smpl_layer.forward(
        betas=gt_betas_flat,
        body_pose=gt_pose_aa[:, 3:],
        global_orient=gt_pose_aa[:, :3]
    )
    gt_joints_3d = gt_smpl_output.joints  # [B*T, J, 3] - Use all SMPL joints
    
    # Get the number of joints from SMPL output
    num_joints = gt_joints_3d.shape[1]
    
    # Project GT 3D keypoints to 2D
    gt_keypoints_2d = project_keypoints_to_2d(gt_joints_3d, gt_trans_flat, cam_intrinsics)
    gt_keypoints_2d = gt_keypoints_2d.reshape(B, T, num_joints, 2)
    
    # Now compute keypoint projection loss using all available keypoints with end-effector weighting
    return compute_human_keypoint_projection_loss(
        pred_pose6d, pred_betas, pred_trans, gt_keypoints_2d,
        smpl_layer, criterion, cam_intrinsics, valid_joints=None,
        end_effector_weight=end_effector_weight
    )
    

