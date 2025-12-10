import torch
import torch.nn as nn
import torch.nn.functional as F
from core.config import cfg
import sys
sys.path.append('lib')
from loss.compute_verts import compute_all_vertices
from loss.loss_proj import compute_human_projection_loss
from loss.interact_loss import InterContactLoss, BilateralContactDirectionalLoss


def angular_distance_rot(m1, m2, reduction="mean"):
    m = torch.bmm(m1, m2.transpose(1, 2))  # b*3*3
    m_trace = torch.einsum("bii->b", m)  # batch trace
    cos = (m_trace - 1) / 2  # [-1, 1]
    dist = (1 - cos) / 2  # [0, 1]
    if reduction == "mean":
        return dist.mean()


def so3_log(R, eps=1e-8):
    tr = R[...,0,0] + R[...,1,1] + R[...,2,2]
    cos_theta = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    V = torch.stack([
        R[...,2,1] - R[...,1,2],
        R[...,0,2] - R[...,2,0],
        R[...,1,0] - R[...,0,1]
    ], dim=-1)
    small = theta < 1e-3
    w_small = 0.5 * V
    denom = (2.0 * torch.sin(theta)).unsqueeze(-1) + eps
    scale = (theta.unsqueeze(-1) / denom)
    w_general = scale * V
    return torch.where(small.unsqueeze(-1), w_small, w_general)

def finite_diff_same_len(x, order=1):
    if order == 1:
        dx = x[:, 1:] - x[:, :-1]
        return torch.cat([dx[:, :1], dx], dim=1)
    elif order == 2:
        d1 = finite_diff_same_len(x, 1)
        d2 = finite_diff_same_len(d1, 1)
        return d2


def _second_diff(x: torch.Tensor, dt: float = 1.0):
    """
    x: (B, T, N, 3)
    return: (B, T-2, N, 3)
    """
    return (x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]) / (dt * dt)


def accel_loss_error_weighted(pred: torch.Tensor,
                              gt: torch.Tensor,
                              c: float = 1e-2,
                              dt: float = 1.0):
    a = _second_diff(pred, dt)
    a_norm_sq = (a ** 2).sum(dim=-1)
    err = torch.linalg.norm(pred - gt, dim=-1)   # (B, T, N)
    w = err[:, 1:-1] / (err[:, 1:-1] + c)        # (B, T-2, N)
    return (w * a_norm_sq).mean()


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
            'h_shape': trial.suggest_float('h_shape', 0.001, 1.0, log=True),
            'h_verts': trial.suggest_float('h_verts', 0.1, 10.0, log=True),
            'h_proj': trial.suggest_float('h_proj', 10.0, 1000.0, log=True),
            'o_rot': trial.suggest_float('o_rot', 0.1, 10.0, log=True),
            'o_trans': trial.suggest_float('o_trans', 0.1, 10.0, log=True),
            'distance': trial.suggest_float('distance', 1.0, 100.0, log=True),
            'interact': trial.suggest_float('interact', 0.1, 10.0, log=True),
        }
    else:
        # Use weights from config
        weights = {
            # Human losses
            'h_pose': cfg.TRAIN.h_pose_loss_weight,
            'h_shape': cfg.TRAIN.h_shape_loss_weight,
            'h_trans': cfg.TRAIN.h_trans_loss_weight,
            'h_verts': cfg.TRAIN.h_verts_loss_weight,
            'h_proj': cfg.TRAIN.h_proj_loss_weight,
            
            # Object losses
            'o_rot': cfg.TRAIN.o_rot_loss_weight,
            'o_trans': cfg.TRAIN.o_trans_loss_weight,
            'o_centroid': cfg.TRAIN.o_centroid_loss_weight,
            'o_z': cfg.TRAIN.o_z_loss_weight,
            'o_points': cfg.TRAIN.o_points_loss_weight,
            'o_proj': cfg.TRAIN.o_proj_loss_weight,
            
            # interact losses
            'interact_contact': cfg.TRAIN.interact_contact_loss_weight,
            'directional_contact': cfg.TRAIN.directional_contact_loss_weight,
            'distance': cfg.TRAIN.distance_loss_weight,

            # 'contact': cfg.TRAIN.contact_loss_weight,

            'acc': cfg.TRAIN.acc_loss_weight,
        }
    
    return weights


def compute_loss(preds, batch, gender, weights, smpl_layers, 
                pred_human_verts=None, pred_object_verts=None, pred_human_joints=None,
                gt_human_verts=None, gt_object_verts=None, gt_human_joints=None):
    """
    Compute losses for human-object interact
    
    Args:
        preds: Dictionary of model predictions (now includes pre-computed vertices)
        batch: Dictionary of ground truth data
        gender: Gender information for SMPL model
        weights: Dictionary of loss weights
        smpl_layers: SMPL model layers
        pred_human_verts: Pre-computed predicted human vertices [B*T, V_h, 3] (optional, from preds)
        pred_object_verts: Pre-computed predicted object vertices [B*T, V_o, 3] (optional, from preds)
        pred_human_joints: Pre-computed predicted human joints [B*T, J, 3] (optional, from preds)
        gt_human_verts: Pre-computed GT human vertices [B*T, V_h, 3] (optional)
        gt_object_verts: Pre-computed GT object vertices [B*T, V_o, 3] (optional)
        gt_human_joints: Pre-computed GT human joints [B*T, J, 3] (optional)
        
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
    # bce_criterion = nn.BCELoss()
    inter_contact_loss = InterContactLoss()
    contact_directional_loss = BilateralContactDirectionalLoss(topk_ratio=0.2, gamma=4)

    B, T = preds['hum_pose6d'].shape[:2]

    downsample_smpl_index = preds['downsample_smpl_index']

    # obj_gt_rot_sym = get_closest_rot_batch(preds['obj_rot_mat'], batch['gt_obj_rot_mat'], batch['obj_id'])
    obj_gt_rot_sym = batch['gt_obj_rot_mat']
    obj_gt_rot = batch['gt_obj_rot_mat']

    # Use pre-computed vertices from model predictions if available
    if 'human_verts' in preds and 'obj_verts' in preds:
        pred_human_verts = preds['human_verts']
        pred_object_verts = preds['obj_verts']
        init_pred_human_verts = preds['init_human_verts']
        init_pred_object_verts = preds['init_obj_verts']
        pred_human_joints = preds.get('human_joints', None)
    elif pred_human_verts is None or pred_object_verts is None:
        # Fallback: compute vertices if not provided 
        pred_human_verts, pred_object_verts, pred_human_joints, _ = compute_all_vertices(
            pred_human_pose6d=preds['hum_pose6d'],
            pred_human_betas=preds['hum_betas'],
            pred_human_trans=preds['hum_trans_cam'],  
            pred_obj_rot=preds['obj_rot_mat'],
            pred_obj_trans=preds['obj_trans_cam'],
            obj_ids=batch['obj_id'],
            smpl_layer=smpl_layers['NEUTRAL'],
            dataset_name=cfg.DATASET.obj_set
        )
    
    if gt_human_verts is None or gt_object_verts is None:
        gt_human_verts, gt_object_verts, gt_human_joints, _ = compute_all_vertices(
            pred_human_pose6d=batch['gt_pose6d'],
            pred_human_betas=batch['gt_betas'],
            pred_human_trans=batch['gt_trans_cam'],  
            pred_obj_rot=obj_gt_rot_sym,
            pred_obj_trans=batch['gt_obj_trans'],
            obj_ids=batch['obj_id'],
            smpl_layer=smpl_layers['NEUTRAL'],
            dataset_name=cfg.DATASET.obj_set
        )

    gt_human_verts_downsampled = torch.gather(gt_human_verts, 1, downsample_smpl_index)
    init_pred_human_verts_downsampled = torch.gather(init_pred_human_verts, 1, downsample_smpl_index)
    pred_human_verts_downsampled = torch.gather(pred_human_verts, 1, downsample_smpl_index)

    # 1. Human pose loss (MSE loss)
    if weights['h_pose'] > 0:
        loss_h_pose = mse_criterion(preds['hum_pose6d'], batch['gt_pose6d'].reshape(B,T,-1))
        total_loss += weights['h_pose'] * loss_h_pose
        loss_dict['h_pose'] = (weights['h_pose'] * loss_h_pose).item()

        loss_h_pose_init = mse_criterion(preds['init_hum_pose6d'], batch['gt_pose6d'].reshape(B,T,-1))
        total_loss += weights['h_pose'] / 5 * loss_h_pose_init
        loss_dict['init_h_pose'] = (weights['h_pose'] / 5 * loss_h_pose_init).item()

    # 2. Human shape loss (MSE loss)
    if weights['h_shape'] > 0:
        loss_h_shape = mse_criterion(preds['hum_betas'], batch['gt_betas'].reshape(B,T,-1))
        total_loss += weights['h_shape'] * loss_h_shape
        loss_dict['h_shape'] = (weights['h_shape'] * loss_h_shape).item()

        loss_h_shape_init = mse_criterion(preds['init_hum_betas'], batch['gt_betas'].reshape(B,T,-1))
        total_loss += weights['h_shape'] / 5 * loss_h_shape_init
        loss_dict['init_h_shape'] = (weights['h_shape'] / 5 * loss_h_shape_init).item()

    # 3. Human trajectory loss (L1 loss)
    if weights['h_trans'] > 0:
        loss_h_trans = l1_criterion(preds['hum_trans_cam'], batch['gt_trans_cam'])
        total_loss += weights['h_trans'] * loss_h_trans
        loss_dict['h_trans'] = (weights['h_trans'] * loss_h_trans).item()

        loss_h_trans_init = mse_criterion(preds['init_hum_trans_cam'], batch['gt_trans_cam'])
        total_loss += weights['h_trans'] / 5 * loss_h_trans_init
        loss_dict['init_h_trans'] = (weights['h_trans'] / 5 * loss_h_trans_init).item()

    # 4. Human vertices loss (L1 loss) - using pre-computed vertices with center normalization
    if weights['h_verts'] > 0:
        pred_human_verts_centered = pred_human_verts - pred_human_verts.mean(dim=1, keepdim=True)
        gt_human_verts_centered = gt_human_verts - gt_human_verts.mean(dim=1, keepdim=True)
        
        loss_h_verts = l1_criterion(pred_human_verts_centered, gt_human_verts_centered)
        total_loss += weights['h_verts'] * loss_h_verts
        loss_dict['h_verts'] = (weights['h_verts'] * loss_h_verts).item()

    # 5. Object rotation loss (MSE loss)
    if weights['o_rot'] > 0:
        loss_obj_rot = angular_distance_rot(preds['obj_rot_mat'].reshape(-1, 3, 3), obj_gt_rot_sym.reshape(-1, 3, 3))
        total_loss += weights['o_rot'] * loss_obj_rot
        loss_dict['o_rot'] = (weights['o_rot'] * loss_obj_rot).item()

        loss_obj_rot_init = angular_distance_rot(preds['init_obj_rot_mat'].reshape(-1, 3, 3), obj_gt_rot_sym.reshape(-1, 3, 3))
        total_loss += weights['o_rot'] / 5 * loss_obj_rot_init
        loss_dict['init_o_rot'] = (weights['o_rot'] / 5 * loss_obj_rot_init).item()
    
    # 6. Human projection loss
    if weights['h_proj'] > 0:
        loss_h_proj = compute_human_projection_loss(
            pred_pose6d=preds['hum_pose6d'],
            pred_betas=preds['hum_betas'],
            pred_trans=preds['hum_trans_cam'],
            gt_pose6d=batch['gt_pose6d'],
            gt_betas=batch['gt_betas'],
            gt_trans=batch['gt_trans_cam'],
            smpl_layer=smpl_layers['NEUTRAL'],
            criterion=mse_criterion,
            cam_intrinsics=batch['roi_cam'].reshape(-1, 3, 3),
            end_effector_weight=3.0
        )
        total_loss += weights['h_proj'] * loss_h_proj
        loss_dict['h_proj'] = (weights['h_proj'] * loss_h_proj).item()

    if weights['interact_contact'] > 0:
        loss_interact_contact = inter_contact_loss(init_pred_human_verts_downsampled, init_pred_object_verts, gt_human_verts_downsampled, gt_object_verts)
        total_loss += weights['interact_contact'] * loss_interact_contact
        loss_dict['interact_contact'] = (weights['interact_contact'] * loss_interact_contact).item()

    if weights['directional_contact'] > 0:
        loss_directional_contact = contact_directional_loss(pred_human_verts_downsampled, pred_object_verts, gt_human_verts_downsampled, gt_object_verts)
        total_loss += weights['directional_contact'] * loss_directional_contact
        loss_dict['directional_contact'] = (weights['directional_contact'] * loss_directional_contact).item()


    # 7. centroid loss
    loss_centroid = nn.L1Loss(reduction="mean")(preds['obj_bbox_centroid'], batch['gt_bbox_centroid'])
    total_loss += weights['o_centroid'] * loss_centroid
    loss_dict['o_centroid'] = (weights['o_centroid'] * loss_centroid).item()

    loss_centroid_init = nn.L1Loss(reduction="mean")(preds['init_obj_bbox_centroid'], batch['gt_bbox_centroid'])
    total_loss += weights['o_centroid'] / 5 * loss_centroid_init
    loss_dict['init_o_centroid'] = (weights['o_centroid'] / 5 * loss_centroid_init).item()

    # 8. obj z loss
    loss_z = nn.L1Loss(reduction="mean")(preds['obj_bbox_z'], batch['gt_bbox_z'])
    total_loss += weights['o_z'] * loss_z
    loss_dict['o_z'] = (weights['o_z'] * loss_z).item()

    loss_z_init = nn.L1Loss(reduction="mean")(preds['init_obj_bbox_z'], batch['gt_bbox_z'])
    total_loss += weights['o_z'] / 5 * loss_z_init
    loss_dict['init_o_z'] = (weights['o_z'] / 5 * loss_z_init).item()

    pred_object_verts_centered = pred_object_verts - pred_object_verts.mean(dim=1, keepdim=True)
    gt_object_verts_centered = gt_object_verts - gt_object_verts.mean(dim=1, keepdim=True)
    loss_point = nn.L1Loss(reduction="mean")(pred_object_verts_centered, gt_object_verts_centered)
    total_loss += weights['o_points'] * loss_point
    loss_dict['o_points'] = (weights['o_points'] * loss_point).item()

    init_pred_object_verts_centered = init_pred_object_verts - init_pred_object_verts.mean(dim=1, keepdim=True)
    gt_object_verts_centered = gt_object_verts - gt_object_verts.mean(dim=1, keepdim=True)
    loss_point_init = nn.L1Loss(reduction="mean")(init_pred_object_verts_centered, gt_object_verts_centered)
    total_loss += weights['o_points'] / 5 * loss_point_init
    loss_dict['init_o_points'] = (weights['o_points'] / 5 * loss_point_init).item()

    # 9. obj trans loss
    if weights['o_trans'] > 0:
        loss_obj_trans_xy = nn.L1Loss(reduction="mean")(preds['obj_trans_cam'][:, :2], batch['gt_obj_trans'][:, :2])
        loss_obj_trans_z = nn.L1Loss(reduction="mean")(preds['obj_trans_cam'][:, 2], batch['gt_obj_trans'][:, 2])
        total_loss += weights['o_trans'] * loss_obj_trans_xy
        total_loss += weights['o_trans'] * loss_obj_trans_z
        loss_dict['o_trans_xy'] = (weights['o_trans'] * loss_obj_trans_xy).item()
        loss_dict['o_trans_z'] = (weights['o_trans'] * loss_obj_trans_z).item()

        loss_obj_trans_xy_init = nn.L1Loss(reduction="mean")(preds['init_obj_trans_cam'][:, :2], batch['gt_obj_trans'][:, :2])
        loss_obj_trans_z_init = nn.L1Loss(reduction="mean")(preds['init_obj_trans_cam'][:, 2], batch['gt_obj_trans'][:, 2])
        total_loss += weights['o_trans'] / 5 * loss_obj_trans_xy_init
        total_loss += weights['o_trans'] / 5 * loss_obj_trans_z_init
        loss_dict['init_o_trans_xy'] = (weights['o_trans'] / 5 * loss_obj_trans_xy_init).item()
        loss_dict['init_o_trans_z'] = (weights['o_trans'] / 5 * loss_obj_trans_z_init).item()

    # 10. Acceleration loss for object vertices
    if weights['acc'] > 0:
        acc_loss = accel_loss_error_weighted(
            pred_object_verts_centered.reshape(B,T,64,3),
            gt_object_verts_centered.reshape(B,T,64,3),
            c=1e-2,
            dt=1.0
        )
        total_loss += weights['acc'] * acc_loss
        loss_dict['acc'] = (weights['acc'] * acc_loss).item()

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
    components = {
        'total': 0.0,
        'h_trans': 0.0,
        'h_pose': 0.0,
        'h_shape': 0.0,
        'h_verts': 0.0,
        'o_rot': 0.0,
        'o_trans': 0.0,
        'h_proj': 0.0,
        'distance': 0.0,
        'interact_contact': 0.0,
        'directional_contact': 0.0,
        'o_z': 0.0,
        'o_trans_xy': 0.0,
        'o_trans_z': 0.0,
        'o_points': 0.0,
        'o_centroid': 0.0,
        'o_proj': 0.0,
        'contact': 0.0,
        

        'init_h_trans': 0.0,
        'init_h_pose': 0.0,
        'init_h_shape': 0.0,
        'init_o_rot': 0.0,
        'init_o_z': 0.0,
        'init_o_points': 0.0,
        'init_o_centroid': 0.0,
        "init_o_trans_xy": 0.0,
        "init_o_trans_z": 0.0,

        'acc': 0.0
    }
    return components

