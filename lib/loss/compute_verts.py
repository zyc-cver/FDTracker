import os
import json
import trimesh
import torch
import torch.nn as nn
import numpy as np
import pickle
import core.config as cfg
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
from core.config import cfg
from utils.data_utils import inverse_trans_obj_id

obj_info = None
obj_data = None

def compute_human_vertices(pred_pose6d, pred_betas, pred_trans, smpl_layer):
    """
    Compute human vertices from pose, shape and translation parameters
    
    Args:
        pred_pose6d: Predicted pose in 6D rotation format [B, T, 132] or [B*T, 22, 6]
        pred_betas: Predicted shape parameters [B, T, 10] or [B*T, 10]
        pred_trans: Predicted translation [B, T, 3] or [B*T, 3]
        smpl_layer: SMPL model layer
        
    Returns:
        human_verts: Human vertices [B*T, V, 3]
        human_joints: Human joints [B*T, J, 3]
    """
    # Handle different input shapes
    if len(pred_pose6d.shape) == 3:  # [B, T, 132]
        B, T = pred_pose6d.shape[:2]
        pred_pose6d = pred_pose6d.reshape(-1, 22, 6)
        pred_betas = pred_betas.reshape(-1, 10)
        pred_trans = pred_trans.reshape(-1, 3)
    else:  # Already flattened
        pred_pose6d = pred_pose6d.reshape(-1, 22, 6)
        pred_betas = pred_betas.reshape(-1, 10)
        pred_trans = pred_trans.reshape(-1, 3)
    
    # Convert 6D rotation to axis-angle for SMPL
    pred_pose_aa = matrix_to_axis_angle(rotation_6d_to_matrix(pred_pose6d)).reshape(-1, 66)
    
    # Forward pass through SMPL
    smpl_output = smpl_layer.forward(
        betas=pred_betas,
        body_pose=pred_pose_aa[:, 3:],
        global_orient=pred_pose_aa[:, :3]
    )
    
    # Add translation to get vertices in camera coordinate system
    human_verts_cam = smpl_output.vertices + pred_trans.unsqueeze(1)  # [B*T, V, 3]
    human_joints_cam = smpl_output.joints + pred_trans.unsqueeze(1)   # [B*T, J, 3]
    
    return human_verts_cam, human_joints_cam


def compute_object_vertices(pred_rot, pred_trans, obj_ids, dataset_name):
    """
    Compute object vertices from rotation, translation and object templates
    
    Args:
        pred_rot: Predicted object rotation in 6D format [B, T, 6] or [B*T, 3, 3]
        pred_trans: Predicted object translation [B, T, 3] or [B*T, 3]
        obj_ids: Object IDs for each batch item [B, T] or [B*T]
        dataset_name: Dataset name for template loading
        
    Returns:
        object_verts: Object vertices [B*T, V, 3]
        template_info: Dictionary with template vertices for each object
    """
    global obj_info, obj_data
    if obj_info is None or obj_data is None:
        with open(cfg.OBJ.template_sparse_path + '/_info.json') as f:
            obj_info = json.load(f)
        obj_data = {}
        for k, v in obj_info.items():
            obj_name = v['path'].split('/')[-1].replace('.obj', '')
            verts64 = np.array(v['kps']).astype(np.float32)
            obj_data[obj_name] = {
                'verts64': torch.from_numpy(verts64).cuda()
            }
    
    # Handle different input shapes
    if len(pred_rot.shape) == 4:  # [B, T, 3, 3]
        B, T = pred_rot.shape[:2]
        pred_rot = pred_rot.reshape(-1, 3, 3)
        pred_trans = pred_trans.reshape(-1, 3)
        obj_ids = obj_ids.reshape(-1)
    else:  # Already flattened
        pred_rot = pred_rot.reshape(-1, 3, 3)
        pred_trans = pred_trans.reshape(-1, 3)
        obj_ids = obj_ids.reshape(-1)
    
    # Convert 6D rotation to rotation matrix
    rot_matrices = pred_rot  # [B*T, 3, 3]
    
    # Initialize list to store vertices for each frame
    all_object_verts = []
    template_info = {}
    
    # Process each frame
    for i in range(len(obj_ids)):
        obj_id = int(obj_ids[i].item())
        obj_name = inverse_trans_obj_id(obj_id, dataset=dataset_name).replace('obj', '')
        
        # Get template vertices from obj_data
        if obj_name not in template_info:
            if obj_name in obj_data:
                template_info[obj_name] = obj_data[obj_name]['verts64']
            else:
                raise ValueError(f"Object template not found for: {obj_name}")
        
        temp_verts = template_info[obj_name]  # [64, 3]
        
        # Apply rotation: vertices @ R^T (since we want R * vertices^T)^T = vertices @ R^T
        rotated_verts = torch.mm(temp_verts, rot_matrices[i].T)  # [64, 3]
        
        # Apply translation
        transformed_verts = rotated_verts + pred_trans[i].unsqueeze(0)  # [64, 3]
        
        all_object_verts.append(transformed_verts)
    
    # Stack all vertices
    object_verts = torch.stack(all_object_verts, dim=0)  # [B*T, 64, 3]
    
    return object_verts, template_info


def compute_all_vertices(pred_human_pose6d, pred_human_betas, pred_human_trans, pred_obj_rot, pred_obj_trans, 
                        obj_ids, smpl_layer, dataset_name):
    """
    Compute both human and object vertices in one function call
    
    Args:
        pred_human_pose6d: Human pose [B, T, 132]
        pred_human_betas: Human shape [B, T, 10]
        pred_human_trans: Human translation [B, T, 3]
        pred_obj_rot: Object rotation [B, T, 6]
        pred_obj_trans: Object translation [B, T, 3]
        obj_ids: Object IDs [B, T]
        smpl_layer: SMPL model layer
        dataset_name: Dataset name for object templates
        
    Returns:
        human_verts: Human vertices (first 49 points) [B*T, 49, 3]
        object_verts: Object vertices [B*T, V_o, 3]
        human_joints: Human joints [B*T, J, 3]
        template_info: Object template information
    """
    # Compute human vertices with translation
    human_verts_full, human_joints = compute_human_vertices(
        pred_human_pose6d, pred_human_betas, pred_human_trans, smpl_layer
    )
    
    # Take only the first 49 vertices to reduce redundancy
    human_verts = human_verts_full[:, :, :]  # [B*T, 49, 3]
    
    # Compute object vertices
    object_verts, template_info = compute_object_vertices(
        pred_obj_rot, pred_obj_trans, obj_ids, dataset_name
    )
    
    return human_verts, object_verts, human_joints, template_info
