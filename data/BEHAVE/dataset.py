import numpy as np
import os
import os.path as osp
import torch
import pickle
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
import sys
sys.path.append('lib.utils')
from core.config import cfg
from torch.utils.data import Dataset
from utils.data_utils import trans_obj_id


def batch_project_points_to_image(rot_mats, trans_vecs, cam_intrinsics):
    """
    Projects a batch of 3D points onto the image plane using rotation matrices, trajectories, and camera intrinsics.

    Args:
        rot_matrices (torch.Tensor): Rotation matrices of shape (T, 3, 3).
        trajectories (torch.Tensor): 3D trajectories of shape (T, 3).
        camera_intrinsics (torch.Tensor): Camera intrinsic matrices of shape (T, 3, 3).

    Returns:
        torch.Tensor: Projected 2D points of shape (T, 2) on the image plane.
    """
    # Transform the 3D points using the rotation matrices
    T = rot_mats.shape[0]
    device = rot_mats.device
    points_3d_local = torch.zeros((T, 3, 1), device=device)
    points_cam = trans_vecs.unsqueeze(-1)  # (T, 3, 1)

    projected_homo = torch.bmm(cam_intrinsics, points_cam).squeeze(-1)  # (T, 3)

    x = projected_homo[:, 0] / projected_homo[:, 2]
    y = projected_homo[:, 1] / projected_homo[:, 2]
    projected_points_2d = torch.stack([x, y], dim=1)  # (T, 2)

    return projected_points_2d

def compute_bbox_center(bbox_sequence):
    """
    Computes the center of bounding boxes.

    Args:
        bbox_sequence (torch.Tensor): Bounding box sequence of shape (T, 4), where each bbox is represented as [x_min, y_min, x_max, y_max].

    Returns:
        torch.Tensor: Bounding box centers of shape (T, 2), where each center is represented as [x_center, y_center].
    """
    x_center = bbox_sequence[:, 0] + bbox_sequence[:, 2] / 2
    y_center = bbox_sequence[:, 1] + bbox_sequence[:, 3] / 2
    return torch.stack([x_center, y_center], dim=1)

def compute_center_differences(obj_center, bbox_center):
    """
    Computes the difference between object centers and bounding box centers.

    Args:
        obj_center (torch.Tensor): Object centers of shape (T, 2).
        bbox_center (torch.Tensor): Bounding box centers of shape (T, 2).

    Returns:
        torch.Tensor: Differences of shape (T, 2).
    """
    return obj_center - bbox_center



class BEHAVE(Dataset):
    def __init__(self, data_path, motion_frames=64, limit_size=None):
        self.data_path = data_path
        self.motion_frames = motion_frames
        self.limit_size = limit_size
        self.motion_files = {}
        self.seqs = []
        self.idx2meta = []
        self.limit_size = limit_size
        self._load_dataset()
        self._get_idx2meta()       

    def _load_dataset(self):
        gt_data = torch.load(osp.join(self.data_path,'gt.pt'), map_location='cpu')
        meta_data = torch.load(osp.join(self.data_path,'metadata.pt'), map_location='cpu')
        bbox_hum = torch.load(osp.join(self.data_path,'hum_bbox.pt'), map_location='cpu')

        for vid_name in gt_data.keys():
            hum_features = torch.load(osp.join(self.data_path, f'hum_feat/{vid_name}.pt'), map_location='cpu')['hum_features']
            num_frames = hum_features.shape[0]

            smplh_poses_reshaped = gt_data[vid_name]['smplh_poses'].reshape(gt_data[vid_name]['smplh_poses'].shape[0], -1, 3)
            body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(smplh_poses_reshaped))[:,:22]
            gt_betas = gt_data[vid_name]["smplh_betas"]
            obj_rot_mat = gt_data[vid_name]['obj_rot'].float()
            obj_id = torch.tensor((trans_obj_id(meta_data[vid_name]['obj_name'], cfg.DATASET.name)), dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1)

            self.motion_files[vid_name] = {
                'hum_features': hum_features,
                'obj_id': obj_id,
                'gt_pose6d': body_pose_r6d,
                'gt_betas': gt_betas,
                'gt_trans_cam': gt_data[vid_name]['smplh_trans'].float(),
                'gt_obj_trans': gt_data[vid_name]['obj_trans'].float(),
                'gt_obj_rot_mat': obj_rot_mat.float(),
                'bbox_xys_hum': bbox_hum[vid_name]['xys'],
                'gender': 'NEUTRAL',
            }

    def _get_idx2meta(self):
        stride = 32
        for vid_name, data in self.motion_files.items():
            num_frames = data['gt_pose6d'].shape[0]
            
            for start_idx in range(0, num_frames - self.motion_frames + 1, stride):
                end_idx = start_idx + self.motion_frames
                self.idx2meta.append((vid_name, start_idx, end_idx))
            
            last_start = num_frames - self.motion_frames
            if last_start >= 0 and (last_start % stride) != 0:
                last_end = num_frames
                self.idx2meta.append((vid_name, last_start, last_end))
            elif num_frames < self.motion_frames:
                self.idx2meta.append((vid_name, 0, num_frames))
                
        if self.limit_size:
            self.idx2meta = self.idx2meta[:self.limit_size]
            
    def _load_data(self, idx):
        vid_name, start_idx, end_idx = self.idx2meta[idx]

        obj_feat = torch.load(osp.join(self.data_path, f'obj_feat/{vid_name}.pt'), map_location='cpu')
        overlap = torch.load(osp.join(self.data_path, f'overlap/{vid_name}.pt'), map_location='cpu')

        cam_intri = obj_feat['cam'].squeeze().float()
        obj_center = batch_project_points_to_image(self.motion_files[vid_name]['gt_obj_rot_mat'], self.motion_files[vid_name]['gt_obj_trans'], cam_intri)
        obj_bbox = obj_feat['bbox']
        obj_bbox_center = obj_feat['roi_center']
        obj_center_diff = compute_center_differences(obj_center, obj_bbox_center)
        obj_bbox_center_diff = obj_center_diff / obj_feat['roi_wh']
        z_ratio = self.motion_files[vid_name]['gt_obj_trans'][:, 2] / obj_feat['resize_ratios']
        gt_bbox_centroid = torch.cat([obj_bbox_center_diff, z_ratio.unsqueeze(-1)], dim=-1)

        data = {}

        data['obj_features'] = obj_feat['feat']
        data['roi_cam'] = cam_intri
        data['roi_center'] = obj_feat['roi_center']
        data['resize_ratios'] = obj_feat['resize_ratios']
        data['roi_whs'] = obj_feat['roi_wh']
        data['gt_bbox_z'] = gt_bbox_centroid[:, 2]
        data['gt_bbox_centroid'] = gt_bbox_centroid[:, :2]
        data['obj_overlap'] = overlap.float()
        data['bbox'] = obj_bbox
        
        data['hum_features'] = self.motion_files[vid_name]['hum_features']
        data['obj_id'] = self.motion_files[vid_name]['obj_id']
        data['gt_pose6d'] = self.motion_files[vid_name]['gt_pose6d']
        data['gt_betas'] = self.motion_files[vid_name]['gt_betas']
        data['gt_trans_cam'] = self.motion_files[vid_name]['gt_trans_cam']
        data['gt_obj_trans'] = self.motion_files[vid_name]['gt_obj_trans']
        data['gt_obj_rot_mat'] = self.motion_files[vid_name]['gt_obj_rot_mat']
        data['bbox_xys_hum'] = self.motion_files[vid_name]['bbox_xys_hum']
        data['gender'] = self.motion_files[vid_name]['gender']

        data_clip = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.shape[0] >= end_idx:
                data_clip[k] = v[start_idx:end_idx]
            elif isinstance(v, torch.Tensor):
                actual_frames = v.shape[0]
                padding_needed = self.motion_frames - actual_frames
                padding = v[0:1].expand(padding_needed, *v.shape[1:])
                data_clip[k] = torch.cat([padding, v], dim=0)
            else:
                data_clip[k] = v
        
        return data_clip, data['gender']

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def __getitem__(self, idx):
        data, gender = self._load_data(idx)
        return data, gender

class BEHAVETest(Dataset):
    def __init__(self, data_path, motion_frames=64, limit_size=None):
        self.data_path = data_path
        self.motion_frames = motion_frames
        self.limit_size = limit_size
        self.motion_files = {}
        self.seqs = []
        self.idx2meta = []
        self.has_gt = False  # Flag to indicate if GT data is available
        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        bbox_hum_path = osp.join(self.data_path, 'hum_bbox.pt')
            
        gt_data_path = osp.join(self.data_path, 'gt.pt')
        meta_data_path = osp.join(self.data_path, 'metadata.pt')

        gt_data = None
        meta_data = None

        if osp.exists(gt_data_path):
            gt_data = torch.load(gt_data_path, map_location='cpu')
            self.has_gt = True
            
        if osp.exists(meta_data_path):
            meta_data = torch.load(meta_data_path, map_location='cpu')
            
        bbox_hum = torch.load(bbox_hum_path, map_location='cpu')
        # Process each video sequence
        for vid_name in meta_data.keys():
            vid_info = {}

            # Always add input features
            hum_features = torch.load(osp.join(self.data_path, f'hum_feat/{vid_name}.pt'), map_location='cpu')['hum_features']
            num_frames = hum_features.shape[0]
            vid_info['hum_features'] = hum_features
            vid_info['bbox_xys_hum'] = bbox_hum[vid_name]['xys']

            if meta_data and vid_name in meta_data:
                obj_id = torch.tensor((trans_obj_id(meta_data[vid_name]['obj_name'], cfg.DATASET.name)), 
                                      dtype=torch.float32).unsqueeze(0).repeat(num_frames, 1)
                vid_info['obj_id'] = obj_id
                vid_info['gender'] = meta_data[vid_name]['gender']
            else:
                # Default object ID if not available
                vid_info['obj_id'] = torch.zeros((num_frames, 1), dtype=torch.float32)
                vid_info['gender'] = 'NEUTRAL'  # Default gender
            
            # Add ground truth if available
            if self.has_gt and gt_data and vid_name in gt_data:
                gt = gt_data[vid_name]
                
                # Extract SMPL pose parameters
                if 'smplh_poses' in gt:
                    smplh_poses_reshaped = gt['smplh_poses'].reshape(gt['smplh_poses'].shape[0], -1, 3)
                    body_pose_r6d = matrix_to_rotation_6d(axis_angle_to_matrix(smplh_poses_reshaped))[:,:22]
                    vid_info['gt_pose6d'] = body_pose_r6d
                
                # Add other GT data
                if 'smplh_betas' in gt:
                    vid_info['gt_betas'] = gt['smplh_betas']
                
                if 'smplh_trans' in gt:
                    vid_info['gt_trans_cam'] = gt['smplh_trans']
                    vid_info['gt_relative_trans'] = gt['smplh_trans'] - gt['smplh_trans']
                
                if 'obj_rot' in gt:
                    vid_info['gt_obj_rot_mat'] = gt['obj_rot']
                
                if 'obj_trans' in gt:
                    vid_info['gt_obj_trans'] = gt['obj_trans']
            
            # Add metadata for visualization
            vid_info['meta'] = {
                'seq_name': vid_name,
                'frame_ids': list(range(num_frames))
            }
            
            self.motion_files[vid_name] = vid_info

    def _get_idx2meta(self):
        self.idx2meta = []
        for vid_name, data in self.motion_files.items():
            num_frames = data['hum_features'].shape[0]
            n_full = num_frames // self.motion_frames
            n_remain = num_frames % self.motion_frames
            # 正常切片
            for i in range(n_full):
                start_idx = i * self.motion_frames
                end_idx = start_idx + self.motion_frames
                self.idx2meta.append({
                    'vid_name': vid_name,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'is_last': False,
                    'orig_len': num_frames
                })

            if n_remain > 0:
                start_idx = num_frames - self.motion_frames
                end_idx = num_frames
                if start_idx < 0:
                    start_idx = 0
                self.idx2meta.append({
                    'vid_name': vid_name,
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'is_last': True,
                    'orig_len': num_frames
                })
            elif n_full > 0:
                self.idx2meta[-1]['is_last'] = True
        if self.limit_size:
            self.idx2meta = self.idx2meta[:self.limit_size]

    def _load_data(self, idx):
        meta = self.idx2meta[idx]
        vid_name = meta['vid_name']
        start_idx = meta['start_idx']
        end_idx = meta['end_idx']

        obj_feat = torch.load(osp.join(self.data_path, f'obj_feat/{vid_name}.pt'), map_location='cpu')
        overlap = torch.load(osp.join(self.data_path, f'overlap/{vid_name}.pt'), map_location='cpu')

        cam_intri = obj_feat['cam'].squeeze().float()
        obj_center = batch_project_points_to_image(self.motion_files[vid_name]['gt_obj_rot_mat'], self.motion_files[vid_name]['gt_obj_trans'], cam_intri)
        obj_bbox = obj_feat['bbox']
        obj_bbox_center = obj_feat['roi_center']
        obj_center_diff = compute_center_differences(obj_center, obj_bbox_center)
        obj_bbox_center_diff = obj_center_diff / obj_feat['roi_wh']
        z_ratio = self.motion_files[vid_name]['gt_obj_trans'][:, 2] / obj_feat['resize_ratios']
        gt_bbox_centroid = torch.cat([obj_bbox_center_diff, z_ratio.unsqueeze(-1)], dim=-1)

        data = {}

        data['obj_features'] = obj_feat['feat']
        data['roi_cam'] = cam_intri
        data['roi_center'] = obj_feat['roi_center']
        data['resize_ratios'] = obj_feat['resize_ratios']
        data['roi_whs'] = obj_feat['roi_wh']
        data['gt_bbox_z'] = gt_bbox_centroid[:, 2]
        data['gt_bbox_centroid'] = gt_bbox_centroid[:, :2]
        data['obj_overlap'] = overlap.float()
        data['bbox'] = obj_bbox

        data['hum_features'] = self.motion_files[vid_name]['hum_features']
        data['obj_id'] = self.motion_files[vid_name]['obj_id']
        data['gt_pose6d'] = self.motion_files[vid_name]['gt_pose6d']
        data['gt_betas'] = self.motion_files[vid_name]['gt_betas']
        data['gt_trans_cam'] = self.motion_files[vid_name]['gt_trans_cam']
        data['gt_obj_trans'] = self.motion_files[vid_name]['gt_obj_trans']
        data['gt_obj_rot_mat'] = self.motion_files[vid_name]['gt_obj_rot_mat']
        data['bbox_xys_hum'] = self.motion_files[vid_name]['bbox_xys_hum']
        data['gender'] = self.motion_files[vid_name]['gender']
        data['meta'] = self.motion_files[vid_name]['meta']
                
        data_clip = {}
        for k, v in data.items():
            if isinstance(v, torch.Tensor) and v.shape[0] >= end_idx:
                clip = v[start_idx:end_idx]
                # 补齐
                actual_len = clip.shape[0]
                if actual_len < self.motion_frames:
                    pad_len = self.motion_frames - actual_len
                    pad = v[0:1].expand(pad_len, *v.shape[1:])
                    clip = torch.cat([pad, clip], dim=0)
                data_clip[k] = clip
            else:
                data_clip[k] = v
                
        if 'meta' not in data_clip:
            data_clip['meta'] = {}
        data_clip['meta']['seq_name'] = vid_name
        data_clip['meta']['start_frame'] = start_idx
        data_clip['meta']['end_frame'] = end_idx
        data_clip['meta']['is_last_segment'] = meta['is_last']
        data_clip['meta']['orig_len'] = meta['orig_len']
        
        if end_idx - start_idx < self.motion_frames:
            pad_len = self.motion_frames - (end_idx - start_idx)
            frame_ids = [-i-1 for i in range(pad_len)] + list(range(start_idx, end_idx))
        else:
            frame_ids = list(range(start_idx, end_idx))
        data_clip['meta']['frame_ids'] = frame_ids
        return data_clip, data.get('gender', 'NEUTRAL')

    def __len__(self):
        if self.limit_size is not None:
            return min(self.limit_size, len(self.idx2meta))
        return len(self.idx2meta)

    def __getitem__(self, idx):
        data, gender = self._load_data(idx)
        return data, gender