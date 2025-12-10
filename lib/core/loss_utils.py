import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.config import cfg, model_verts
from models.templates import smplh
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
import pickle
import sys
sys.path.append('lib')
from utils.data_utils import inverse_trans_obj_id


class ClsLoss(nn.Module):
    def __init__(self):
        super(ClsLoss, self).__init__()        
        self.criterion = nn.BCELoss(reduction='mean')

    def forward(self, pred, gt, valid=None):
        gt = gt.to(pred.device)
        batch_size = gt.shape[0]
        
        if valid is not None:
            valid = valid.bool()
            pred, gt = pred[valid], gt[valid]

        return self.criterion(pred, gt)


class ParamLoss(nn.Module):
    def __init__(self, type='l1'):
        super(ParamLoss, self).__init__()        
        if type == 'l1': self.criterion = nn.L1Loss(reduction='mean')
        elif type == 'l2': self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, param_out, param_gt, valid=None):
        param_out = param_out.reshape(param_out.shape[0], -1)
        param_gt = param_gt.reshape(param_gt.shape[0], -1).to(param_out.device)

        if valid is not None:
            valid = valid.reshape(-1).to(param_out.device)
            param_out, param_gt = param_out * valid[:,None], param_gt * valid[:,None]
        return self.criterion(param_out, param_gt)


class CoordLoss(nn.Module):
    def __init__(self, type='l1'):
        super(CoordLoss, self).__init__()
        if type == 'l1': self.criterion = nn.L1Loss(reduction='mean')
        elif type == 'l2': self.criterion = nn.MSELoss(reduction='mean')

    def forward(self, pred, target, valid=None):
        target = target.to(pred.device)
        if valid is None:
            if pred.shape[-1] != target.shape[-1]:
                target, valid = target[...,:-1], target[...,-1]
            else:
                return self.criterion(pred, target)
        else:
            valid = valid.to(pred.device)

        pred, target = pred * valid[...,None], target * valid[...,None]
        return self.criterion(pred, target)


class EdgeLengthLoss(nn.Module):
    def __init__(self, face):
        super(EdgeLengthLoss, self).__init__()
        self.face = torch.LongTensor(face.astype(int))

    def forward(self, coord_out, coord_gt):
        coord_gt = coord_gt.to(coord_out.device)
        face = self.face.to(coord_out.device)

        d1_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 0], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_out = torch.sqrt(
            torch.sum((coord_out[:, face[:, 1], :] - coord_out[:, face[:, 2], :]) ** 2, 2, keepdim=True))

        d1_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 1], :]) ** 2, 2, keepdim=True))
        d2_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 0], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
        d3_gt = torch.sqrt(torch.sum((coord_gt[:, face[:, 1], :] - coord_gt[:, face[:, 2], :]) ** 2, 2, keepdim=True))
 
        diff1 = torch.abs(d1_out - d1_gt)
        diff2 = torch.abs(d2_out - d2_gt)
        diff3 = torch.abs(d3_out - d3_gt)
        loss = torch.cat((diff1, diff2, diff3), 1)
        return loss.mean()


# New distance loss for relative position
class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, pos1, pos2, gt_pos1, gt_pos2):
        # Calculate relative distances
        pred_dist = pos1 - pos2
        gt_dist = gt_pos1 - gt_pos2
        
        return self.criterion(pred_dist, gt_dist)


# Add projection loss to handle 3D-to-2D projections
class ProjectionLoss(nn.Module):
    def __init__(self):
        super(ProjectionLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, proj_points, target_points, valid=None):
        target_points = target_points.to(proj_points.device)
        
        if valid is not None:
            valid = valid.to(proj_points.device)
            proj_points = proj_points * valid[..., None]
            target_points = target_points * valid[..., None]
            
        return self.criterion(proj_points, target_points)


# Add a new loss class for velocity smoothness
class VelocitySmoothnessLoss(nn.Module):
    def __init__(self):
        super(VelocitySmoothnessLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        
    def forward(self, positions, valid_mask=None):
        """
        Compute velocity smoothness loss on a trajectory
        
        Args:
            positions: Tensor of shape (B, T, D) where B is batch size, T is sequence length, D is dimension
            valid_mask: Optional tensor of shape (B, T) with 1s for valid frames and 0s for invalid frames
            
        Returns:
            Loss value measuring the smoothness of velocities
        """
        # Calculate velocities (first differences)
        velocities = positions[:, 1:] - positions[:, :-1]
        
        # Calculate accelerations (second differences)
        accelerations = velocities[:, 1:] - velocities[:, :-1]
        
        # If we have a valid mask, adjust it for acceleration calculation (need 3 consecutive valid frames)
        if valid_mask is not None:
            valid_mask = valid_mask.bool()
            # For velocities, we need 2 consecutive valid frames
            vel_mask = valid_mask[:, :-1] & valid_mask[:, 1:]
            # For accelerations, we need 3 consecutive valid frames
            acc_mask = vel_mask[:, :-1] & vel_mask[:, 1:]
            
            # Apply the mask
            accelerations = accelerations * acc_mask.unsqueeze(-1)
            
            # Mean over valid values only
            smoothness_loss = (accelerations ** 2).sum() / (acc_mask.sum() * accelerations.shape[-1] + 1e-8)
        else:
            # Mean over all values
            smoothness_loss = (accelerations ** 2).mean()
            
        return smoothness_loss


def get_loss():
    """
    Get dictionary of loss functions for original loss components
    """
    loss = {}
    loss['contact'] = ClsLoss()
    loss['vert'] = CoordLoss(type='l1')
    loss['edge'] = EdgeLengthLoss(smplh.faces)
    loss['param'] = ParamLoss(type='l1')
    loss['coord'] = CoordLoss(type='l1')
    loss['hand_bbox'] = CoordLoss(type='l1')  
    return loss


def get_loss_weights(trial=None):
    """
    Get loss weights from config or optimize with optuna
    
    Args:
        trial: Optional optuna trial for hyperparameter optimization
        
    Returns:
        Dictionary of loss weights
    """
    if trial is not None:
        # For hyperparameter tuning with optuna
        weights = {
            'h_trans': trial.suggest_float('h_trans', 0.1, 10.0, log=True),
            'h_pose': trial.suggest_float('h_pose', 0.1, 10.0, log=True),
            'h_shape': trial.suggest_float('h_shape', 0.1, 10.0, log=True),
            'h_verts': trial.suggest_float('h_verts', 0.1, 10.0, log=True),
            'o_rot': trial.suggest_float('o_rot', 0.1, 10.0, log=True),
            'o_trans': trial.suggest_float('o_trans', 0.1, 10.0, log=True),
            'o_verts': trial.suggest_float('o_verts', 0.1, 10.0, log=True),
            'proj': trial.suggest_float('proj', 0.1, 10.0, log=True),
            'distance': trial.suggest_float('distance', 0.1, 10.0, log=True),
            'occlusion': trial.suggest_float('occlusion', 0.0, 5.0, log=True),
        }
    else:
        # Use weights from config
        weights = {
            # Human losses
            'h_pose': cfg.TRAIN.h_pose_loss_weight,
            'h_shape': cfg.TRAIN.h_shape_loss_weight,
            'h_verts': cfg.TRAIN.h_verts_loss_weight,
            'proj': cfg.TRAIN.proj_loss_weight,
            
            # Object losses
            'o_rot': cfg.TRAIN.o_rot_loss_weight,
            'o_verts': cfg.TRAIN.o_verts_loss_weight,
            
            # Trajectory losses
            'h_trans': cfg.TRAIN.h_trans_loss_weight,
            'obj_trans': cfg.TRAIN.o_trans_loss_weight,
            'distance': cfg.TRAIN.distance_loss_weight,
            
            # Other losses
            'occlusion': cfg.TRAIN.occlusion_loss_weight
            }
    
    return weights


def compute_loss(preds, batch, gender, weights, smpl_layers):
    """
    Compute losses for human-object interaction
    
    Args:
        preds: Dictionary of model predictions
        batch: Dictionary of ground truth data
        gender: Gender information for SMPL model
        weights: Dictionary of loss weights
        smpl_layers: SMPL model layers
        
    Returns:
        total_loss: Combined weighted loss
        loss_dict: Dictionary of individual loss components
    """
    # Initialize total loss and loss dictionary
    total_loss = torch.tensor(0.0, device=preds['hum_trans_cam'].device)
    loss_dict = {}
    
    # Initialize loss modules
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    bce_criterion = nn.BCELoss()
    velocity_criterion = VelocitySmoothnessLoss()
    B, T = preds['hum_pose6d'].shape[:2]
    # 1. Human pose loss (MSE loss)
    if weights['h_pose'] > 0:
        loss_h_pose = mse_criterion(preds['hum_pose6d'], batch['gt_pose6d'].reshape(B,T,-1))
        total_loss += weights['h_pose'] * loss_h_pose
        loss_dict['h_pose'] = loss_h_pose.item()
    
    # 2. Human shape loss (MSE loss)
    if weights['h_shape'] > 0:
        loss_h_shape = mse_criterion(preds['hum_betas'], batch['gt_betas'].reshape(B,T,-1))
        total_loss += weights['h_shape'] * loss_h_shape
        loss_dict['h_shape'] = loss_h_shape.item()
    
    # 3. Human trajectory loss (L1 loss)
    if weights['h_trans'] > 0:
        loss_h_trans = l1_criterion(preds['hum_trans_cam'], batch['gt_trans_cam'])
        total_loss += weights['h_trans'] * loss_h_trans
        loss_dict['h_trans'] = loss_h_trans.item()

    # 4. Human vertices loss (L1 loss) - needs SMPL forward pass
    if weights['h_verts'] > 0:
        h_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(preds['hum_pose6d'].reshape(-1, 22, 6))).reshape(-1, 66)
        h_betas = preds['hum_betas'].reshape(-1, 10)
        gt_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(batch['gt_pose6d'].reshape(-1, 22, 6))).reshape(-1, 66)
        gt_betas = batch['gt_betas'].reshape(-1, 10)
        h_verts = smpl_layers['NEUTRAL'].forward(
            betas=h_betas,
            body_pose=h_pose_aa[:, 3:],
            global_orient=h_pose_aa[:, :3],
            gender=gender).vertices
        gt_verts = smpl_layers['NEUTRAL'].forward(
            betas=gt_betas,
            body_pose=gt_pose_aa[:, 3:],
            global_orient=gt_pose_aa[:, :3],
            gender=gender).vertices
        loss_h_verts = l1_criterion(h_verts, gt_verts)
        total_loss += weights['h_verts'] * loss_h_verts
        loss_dict['h_verts'] = loss_h_verts.item()

    # 5. Object rotation loss (MSE loss)
    if weights['o_rot'] > 0:
        loss_obj_rot = mse_criterion(preds['obj_rot_mat'], batch['gt_obj_rot_mat'])
        total_loss += weights['o_rot'] * loss_obj_rot
        loss_dict['o_rot'] = loss_obj_rot.item()
    
    # 6. Object translation loss (L1 loss)
    if weights['obj_trans'] > 0:
        loss_obj_trans = l1_criterion(preds['obj_trans_cam'], batch['gt_obj_trans'])
        total_loss += weights['obj_trans'] * loss_obj_trans
        loss_dict['obj_trans'] = loss_obj_trans.item()
    
    # 7. Object vertices loss (L1 loss)
    if weights['o_verts'] > 0:
        # Load object templates from config path
        obj_templates = pickle.load(open(cfg.OBJ.template_path, 'rb'))[cfg.OBJ.template_key]
        # Get batch size
        batch_size = preds['obj_rot_mat'].shape[0]
        
        # Extract rotations and translations
        o_pose_mat = preds['obj_rot_mat'].reshape(-1, 3, 3)  # (B*T, 3, 3)
        o_gt_mat = batch['gt_obj_rot_mat'].reshape(-1, 3, 3)  # (B*T, 3, 3)

        obj_vert_losses = []
        # Process each object in the batch separately
        for b in range(batch_size):
            obj_id = int(batch['obj_id'][b][0].item())
            obj_name = inverse_trans_obj_id(obj_id, dataset=cfg.DATASET.obj_set)
            # Get template vertices
            temp_verts = torch.tensor(obj_templates[obj_name]['verts']).float().to(preds['obj_rot_mat'].device)
            
            # Calculate indices for this batch item across all time steps
            b_indices = torch.arange(b*T, (b+1)*T, device=preds['obj_rot_mat'].device)

            # Get rotations and translations for all time steps of this object
            obj_rots = o_pose_mat[b_indices]  # (T, 3, 3)
            obj_gt_rots = o_gt_mat[b_indices]  # (T, 3, 3)
            obj_trans = preds['obj_trans_cam'][b].unsqueeze(1)  # (T, 1, 3)
            obj_gt_trans = batch['gt_obj_trans'][b].unsqueeze(1)  # (T, 1, 3)
            
            # Expand template vertices for all time steps: (V, 3) -> (T, V, 3)
            temp_verts_expanded = temp_verts.unsqueeze(0).expand(T, -1, -1)
            
            # Apply rotation to vertices for all time steps at once
            pred_verts = torch.bmm(temp_verts_expanded, obj_rots.transpose(1, 2))
            gt_verts = torch.bmm(temp_verts_expanded, obj_gt_rots.transpose(1, 2))
            
            # Compute L1 loss for this object and add to list
            obj_loss = l1_criterion(pred_verts, gt_verts)
            obj_vert_losses.append(obj_loss)
    
        loss_obj_verts = torch.stack(obj_vert_losses).mean()
        total_loss += weights['o_verts'] * loss_obj_verts
        loss_dict['o_verts'] = loss_obj_verts.item()
    
    # 8. Projection loss (MSE loss)
    if weights['proj'] > 0:
        loss_proj = mse_criterion(preds['proj_vertices'], batch['gt_proj_vertices'])
        total_loss += weights['proj'] * loss_proj
        loss_dict['proj'] = loss_proj.item()
    
    # 9. Relative distance loss (MSE loss)
    if weights['distance'] > 0:
        loss_distance = mse_criterion(
            preds['hum_trans_cam'] - preds['obj_trans_cam'],
            batch['gt_trans_cam'] - batch['gt_obj_trans']
        )
        total_loss += weights['distance'] * loss_distance
        loss_dict['distance'] = loss_distance.item()
    
    # 10. Occlusion loss (BCE loss)
    if weights['occlusion'] > 0:
        loss_occlusion = l1_criterion(preds['obj_occlusion'].squeeze(-1), batch['gt_occlusion'])
        total_loss += weights['occlusion'] * loss_occlusion
        loss_dict['occlusion'] = loss_occlusion.item()
    
    # Include total loss in dictionary
    loss_dict['total'] = total_loss.item()
    
    return total_loss, loss_dict


def get_loss_component_tracker():
    """
    Get a dictionary with all possible loss components initialized to 0.0
    for tracking during training and validation
    
    Returns:
        Dictionary with all loss components initialized to 0.0
    """
    # Get loss weights to determine which components to track
    weights = get_loss_weights()
    
    # Initialize tracking dictionary for all components
    components = {
        'total': 0.0,
        'h_trans': 0.0,
        'h_pose': 0.0,
        'h_shape': 0.0,
        'h_verts': 0.0,
        'obj_rot': 0.0,
        'obj_trans': 0.0,
        'obj_verts': 0.0,
        'proj': 0.0,
        'distance': 0.0,
        'occlusion': 0.0,
    }
    
    return components
