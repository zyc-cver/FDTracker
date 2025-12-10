import torch
import numpy as np
import sys
sys.path.append('.')
from lib.core.config import cfg

def compute_transl_full_cam(pred_cam, bbx_xys):
    """
    Convert predicted camera parameters to full camera translation
    
    Args:
        pred_cam: predicted camera parameters with shape (B, T, 3) or (B, 3)
        bbx_xys: bounding box coordinates with shape (B, T, 3) or (B, 3)
        
    Returns:
        cam_t: camera translation with shape (B, T, 3) or (B, 3)
    """
    # Get original tensor shape and reshape if needed
    original_shape = pred_cam.shape
    B = original_shape[0]
    has_time_dim = len(original_shape) > 2
    
    if has_time_dim:
        T = original_shape[1]
        # Reshape to (B*T, 3) for processing
        pred_cam = pred_cam.reshape(-1, 3)
        bbx_xys = bbx_xys.reshape(-1, 3)
    
    # Define camera intrinsics
    k_fullimg = np.array([[912.862, 0, 956.720],
                       [0, 912.676, 554.216],
                       [0, 0, 1.]])
    k_fullimg = torch.from_numpy(k_fullimg).float().to(pred_cam.device)
    k_fullimg = k_fullimg.unsqueeze(0).repeat(pred_cam.shape[0], 1, 1)
    
    # Extract parameters
    s, tx, ty = pred_cam[..., 0], pred_cam[..., 1], pred_cam[..., 2]
    focal_length = k_fullimg[..., 0, 0]

    # Compute camera translation
    icx = k_fullimg[..., 0, 2]
    icy = k_fullimg[..., 1, 2]
    sb = s * bbx_xys[..., 2]
    cx = 2 * (bbx_xys[..., 0] - icx) / (sb + 1e-9)
    cy = 2 * (bbx_xys[..., 1] - icy) / (sb + 1e-9)
    tz = 2 * focal_length / (sb + 1e-9)

    cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)
    
    # Reshape back to original dimensions if needed
    if has_time_dim:
        cam_t = cam_t.reshape(B, T, 3)
    
    return cam_t
    
def get_a_pred_cam(transl, bbx_xys, K_fullimg):
    """Inverse operation of compute_transl_full_cam"""
    assert transl.ndim == bbx_xys.ndim  # (*, L, 3)
    assert K_fullimg.ndim == (bbx_xys.ndim + 1)  # (*, L, 3, 3)
    f = K_fullimg[..., 0, 0]
    cx = K_fullimg[..., 0, 2]
    cy = K_fullimg[..., 1, 2]
    gt_s = 2 * f / (transl[..., 2] * bbx_xys[..., 2])  # (B, L)
    gt_x = transl[..., 0] - transl[..., 2] / f * (bbx_xys[..., 0] - cx)
    gt_y = transl[..., 1] - transl[..., 2] / f * (bbx_xys[..., 1] - cy)
    gt_pred_cam = torch.stack([gt_s, gt_x, gt_y], dim=-1)
    return gt_pred_cam


def get_obj_trans_cam(pred_trans, bbox_xys, bbox_xyxy):
    """Standardize object translation based on bounding box size"""
    # compute_wh
    w, h, s = bbox_xyxy[..., 2] - bbox_xyxy[..., 0], bbox_xyxy[..., 3] - bbox_xyxy[..., 1], bbox_xys[..., 2]
    # Ensure w and h are at least 1.0
    w = torch.clamp(w, min=1.0)
    h = torch.clamp(h, min=1.0)
    s = torch.clamp(s*1.5, min=1.0)
    
    k_fullimg = np.array([[912.862, 0, 956.720],
                       [0, 912.676, 554.216],
                       [0, 0, 1.]])
    k_fullimg = torch.from_numpy(k_fullimg).float().to(pred_trans.device)

    icx = k_fullimg[..., 0, 2]
    icy = k_fullimg[..., 1, 2]
    tx, ty, tz = pred_trans[..., 0], pred_trans[..., 1], pred_trans[..., 2]
    focal_length = k_fullimg[..., 0, 0]

    pred_z = tz * (64.0 / s)
    pred_x = (tx * w + bbox_xys[:,:,0] - icx) * pred_z / focal_length
    pred_y = (ty * h + bbox_xys[:,:,1] - icy) * pred_z / focal_length

    return torch.stack([pred_x, pred_y, pred_z], dim=-1)

def get_bbx_xys(i_j2d, bbx_ratio=[512, 512], do_augment=False, base_enlarge=1.2):
    """Args: (B, L, J, 3) [x,y,c] -> Returns: (B, L, 3)"""
    # Center
    min_x = i_j2d[..., 0].min(-1)[0]
    max_x = i_j2d[..., 0].max(-1)[0]
    min_y = i_j2d[..., 1].min(-1)[0]
    max_y = i_j2d[..., 1].max(-1)[0]
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Size
    h = max_y - min_y  # (B, L)
    w = max_x - min_x  # (B, L)

    if True:  # fit w and h into aspect-ratio
        aspect_ratio = bbx_ratio[0] / bbx_ratio[1]
        mask1 = w > aspect_ratio * h
        h[mask1] = w[mask1] / aspect_ratio
        mask2 = w < aspect_ratio * h
        w[mask2] = h[mask2] * aspect_ratio

    # apply a common factor to enlarge the bounding box
    bbx_size = torch.max(h, w) * base_enlarge

    if do_augment:
        B, L = bbx_size.shape[:2]
        device = bbx_size.device
        if True:
            scaleFactor = torch.rand((B, L), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, L), device=device) * 1.6 - 0.8  # -0.8~0.8
        else:
            scaleFactor = torch.rand((B, 1), device=device) * 0.3 + 1.05  # 1.05~1.35
            txFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8
            tyFactor = torch.rand((B, 1), device=device) * 1.6 - 0.8  # -0.8~0.8

        raw_bbx_size = bbx_size / base_enlarge
        bbx_size = raw_bbx_size * scaleFactor
        center_x += raw_bbx_size / 2 * ((scaleFactor - 1) * txFactor)
        center_y += raw_bbx_size / 2 * ((scaleFactor - 1) * tyFactor)

    return torch.stack([center_x, center_y, bbx_size], dim=-1)

def augment_betas(betas, std=0.1):
    noise = torch.normal(mean=torch.zeros(10), std=torch.ones(10) * std)
    betas_aug = betas + noise[None]
    return betas_aug

def trans_obj_id(obj_id, dataset='intercap'):
    if dataset.lower() == 'intercap':
        name_dict = {
            "obj01": 0,
            "obj02": 1,
            "obj03": 2,
            "obj04": 3,
            "obj05": 4,
            "obj06": 5,
            "obj07": 6,
            "obj08": 7,
            "obj09": 8,
            "obj10": 9,
        }
        return name_dict.get(obj_id, -1)
    elif dataset.lower() == 'behave':
        name_dict = {
            "backpack": 0,
            "basketball": 1, 
            "boxlarge": 2,
            "boxlong": 3,
            "boxmedium": 4,
            "boxsmall": 5,
            "boxtiny": 6,
            "chairblack": 7,
            "chairwood": 8,
            "keyboard": 9,
            "monitor": 10,
            "plasticcontainer": 11,
            "stool": 12,
            "suitcase": 13,
            "tablesmall": 14,
            "tablesquare": 15,
            "toolbox": 16,
            "trashbin": 17,
            "yogaball": 18,
            "yogamat": 19
        }
        return name_dict.get(obj_id, -1)
    else:
        raise ValueError(f"{dataset}, use 'intercap' or 'behave'")

def inverse_trans_obj_id(obj_id, dataset='intercap'):

    if dataset.lower() == 'intercap':
        id_to_name = {
            0: "obj01",
            1: "obj02",
            2: "obj03",
            3: "obj04",
            4: "obj05",
            5: "obj06",
            6: "obj07",
            7: "obj08",
            8: "obj09",
            9: "obj10",
        }
        return id_to_name.get(obj_id, 'unknown')
    elif dataset.lower() == 'behave':
        id_to_name = {
            0: 'backpack',
            1: 'basketball', 
            2: 'boxlarge',
            3: 'boxlong',
            4: 'boxmedium',
            5: 'boxsmall',
            6: 'boxtiny',
            7: 'chairblack',
            8: 'chairwood',
            9: 'keyboard',
            10: 'monitor',
            11: 'plasticcontainer',
            12: 'stool',
            13: 'suitcase',
            14: 'tablesmall',
            15: 'tablesquare',
            16: 'toolbox',
            17: 'trashbin',
            18: 'yogaball',
            19: 'yogamat'
        }
        return id_to_name.get(obj_id, 'unknown')
    else:
        raise ValueError(f"unsupport: {dataset}, use 'intercap' or 'behave'")

def compute_bbox_info(bbx_xys):
    """impl as in BEDLAM
    Args:
        bbx_xys: ((B), N, 3), in pixel space described by K_fullimg
        K_fullimg: ((B), (N), 3, 3)
    Returns:
        bbox_info: ((B), N, 3)
    """
    # Define camera intrinsics
    k_fullimg = np.array([[912.862, 0, 956.720],
                       [0, 912.676, 554.216],
                       [0, 0, 1.]])
    k_fullimg = torch.from_numpy(k_fullimg).float().to(bbx_xys.device)
    fl = k_fullimg[..., 0, 0].unsqueeze(-1)
    icx = k_fullimg[..., 0, 2]
    icy = k_fullimg[..., 1, 2]

    cx, cy, b = bbx_xys[..., 0], bbx_xys[..., 1], bbx_xys[..., 2]
    bbox_info = torch.stack([cx - icx, cy - icy, b], dim=-1)
    bbox_info = bbox_info / fl
    return bbox_info
