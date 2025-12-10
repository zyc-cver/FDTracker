import torch.nn.functional as F
import torch
import numpy as np
import numpy.linalg as LA
import math
from transforms3d.euler import euler2mat
from transforms3d.quaternions import axangle2quat, mat2quat, qmult, quat2mat
from transforms3d.axangles import axangle2mat
from typing import List
_FLOAT_EPS = np.finfo(np.float64).eps

def normalize_vector(v):
    v = F.normalize(v, p=2, dim=1)
    return v


def cross_product(u, v):
    # u, v bxn
    batch = u.shape[0]
    # print (u.shape)
    # print (v.shape)
    i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
    j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
    k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

    out = torch.cat((i.view(batch, 1), j.view(batch, 1), k.view(batch, 1)), 1)  # bx3

    return out

def ortho6d_to_mat_batch(poses):
    B, T, _ = poses.shape  # B: batch size, T: time steps
    # poses bxTx6
    poses = poses.view(-1, 6)  # bx6
    x_raw = poses[:, 0:3]  # bx3
    y_raw = poses[:, 3:6]  # bx3

    x = normalize_vector(x_raw)  # bx3
    z = cross_product(x, y_raw)  # bx3
    z = normalize_vector(z)  # bx3
    y = cross_product(z, x)  # bx3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # bx3x3
    matrix = matrix.view(B, T, 3, 3)  # bxTx3x3
    return matrix



def R_transform(R_src, R_delta, rot_coord="MODEL"):
    """transform R_src use R_delta.

    :param R_src: matrix
    :param R_delta:
    :param rot_coord:
    :return:
    """
    if rot_coord.lower() == "model":
        R_output = np.dot(R_src, R_delta)
    elif rot_coord.lower() == "camera" or rot_coord.lower() == "naive" or rot_coord.lower() == "camera_new":
        R_output = np.dot(R_delta, R_src)
    else:
        raise Exception("Unknown rot_coord in R_transform: {}".format(rot_coord))
    return R_output

def T_transform(T_src, T_delta, T_means, T_stds, rot_coord):
    """
    :param T_src: (x1, y1, z1)
    :param T_delta: (dx, dy, dz), normed
    :return: T_tgt: (x2, y2, z2)
    """
    # print("T_src: {}".format(T_src))
    assert T_src[2] != 0, "T_src: {}".format(T_src)
    T_delta_1 = T_delta * T_stds + T_means
    T_tgt = np.zeros((3,))
    z2 = T_src[2] / np.exp(T_delta_1[2])
    T_tgt[2] = z2
    if rot_coord.lower() == "camera" or rot_coord.lower() == "model":
        # use this
        T_tgt[0] = z2 * (T_delta_1[0] + T_src[0] / T_src[2])
        T_tgt[1] = z2 * (T_delta_1[1] + T_src[1] / T_src[2])
    elif rot_coord.lower() == "camera_new":
        T_tgt[0] = T_src[2] * T_delta_1[0] + T_src[0]
        T_tgt[1] = T_src[2] * T_delta_1[1] + T_src[1]
    else:
        raise Exception("Unknown: {}".format(rot_coord))

    return T_tgt

def se3_mul(RT1, RT2):
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3, 1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3, 1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape(3)
    return RT_new

def RT_transform(pose_src, r, t, T_means, T_stds, rot_coord="MODEL"):
    # r: 4(quat) or 3(euler) number
    # t: 3 number, (delta_x, delta_y, scale)
    r = np.squeeze(r)
    if r.shape[0] == 3:
        Rm_delta = euler2mat(r[0], r[1], r[2])
    elif r.shape[0] == 4:
        # QUAT
        quat = r / LA.norm(r)
        Rm_delta = quat2mat(quat)
    else:
        raise Exception("Unknown r shape: {}".format(r.shape))
    t_delta = np.squeeze(t)

    if rot_coord.lower() == "naive":
        se3_mx = np.zeros((3, 4))
        se3_mx[:, :3] = Rm_delta
        se3_mx[:, 3] = t
        pose_est = se3_mul(se3_mx, pose_src)
    else:
        pose_est = np.zeros((3, 4))
        pose_est[:3, :3] = R_transform(pose_src[:3, :3], Rm_delta, rot_coord)
        pose_est[:3, 3] = T_transform(pose_src[:, 3], t_delta, T_means, T_stds, rot_coord)

    return pose_est


def allocentric_to_egocentric(allo_pose, src_type="mat", dst_type="mat", cam_ray=(0, 0, 1.0)):
    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = np.asarray(cam_ray)
    if src_type == "mat":
        trans = allo_pose[:3, 3]
    elif src_type == "quat":
        trans = allo_pose[4:7]
    else:
        raise ValueError("src_type should be mat or quat, got: {}".format(src_type))
    obj_ray = trans.copy() / np.linalg.norm(trans)
    angle = math.acos(cam_ray.dot(obj_ray))

    # Rotate back by that amount

    if angle > 0:
        if dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, 3] = trans
            rot_mat = axangle2mat(axis=np.cross(cam_ray, obj_ray), angle=angle)
            if src_type == "mat":
                ego_pose[:3, :3] = np.dot(rot_mat, allo_pose[:3, :3])
            elif src_type == "quat":
                ego_pose[:3, :3] = np.dot(rot_mat, quat2mat(allo_pose[:4]))
        elif dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[4:7] = trans
            rot_q = axangle2quat(np.cross(cam_ray, obj_ray), angle)
            if src_type == "quat":
                ego_pose[:4] = qmult(rot_q, allo_pose[:4])
            elif src_type == "mat":
                ego_pose[:4] = qmult(rot_q, mat2quat(allo_pose[:3, :3]))
        else:
            raise ValueError("dst_type should be mat or quat, got: {}".format(dst_type))
    else:  # allo to ego
        if src_type == "mat" and dst_type == "quat":
            ego_pose = np.zeros((7,), dtype=allo_pose.dtype)
            ego_pose[:4] = mat2quat(allo_pose[:3, :3])
            ego_pose[4:7] = allo_pose[:3, 3]
        elif src_type == "quat" and dst_type == "mat":
            ego_pose = np.zeros((3, 4), dtype=allo_pose.dtype)
            ego_pose[:3, :3] = quat2mat(allo_pose[:4])
            ego_pose[:3, 3] = allo_pose[4:7]
        else:
            ego_pose = allo_pose.copy()
    return ego_pose

def pose_from_pred_centroid_z(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    roi_cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-4,
    is_allo=True,
    z_type="REL",
    is_train=True,
):
    if is_train:
        return pose_from_predictions_train(
            pred_rots,
            pred_centroids,
            pred_z_vals,
            roi_cams,
            roi_centers,
            resize_ratios,
            roi_whs,
            eps=eps,
            is_allo=is_allo,
            z_type=z_type,
        )
    else:
        return pose_from_predictions_test(
            pred_rots,
            pred_centroids,
            pred_z_vals,
            roi_cams,
            roi_centers,
            resize_ratios,
            roi_whs,
            eps=eps,
            is_allo=is_allo,
            z_type=z_type,
        )
    
def pose_from_predictions_test(
    pred_rots,
    pred_centroids,
    pred_z_vals,
    roi_cams,
    roi_centers,
    resize_ratios,
    roi_whs,
    eps=1e-4,
    is_allo=True,
    z_type="REL",
):
    """NOTE: for test, non-differentiable"""
    assert roi_cams.dim() == 3, roi_cams.dim()

    # absolute coords
    c = torch.stack(
        [
            (pred_centroids[:, 0] * roi_whs[:, 0]) + roi_centers[:, 0],
            (pred_centroids[:, 1] * roi_whs[:, 1]) + roi_centers[:, 1],
        ],
        dim=1,
    )

    cx = c[:, 0:1]  # [#roi, 1]
    cy = c[:, 1:2]  # [#roi, 1]

    # unnormalize regressed z
    if z_type == "ABS":
        z = pred_z_vals
    elif z_type == "REL":
        # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
        z = pred_z_vals * resize_ratios.view(-1, 1)
    else:
        raise ValueError(f"Unknown z_type: {z_type}")

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """
    translation = torch.cat(
        [z * (cx - roi_cams[:, 0:1, 2]) / roi_cams[:, 0:1, 0], z * (cy - roi_cams[:, 1:2, 2]) / roi_cams[:, 1:2, 1], z],
        dim=1,
    )

    # quat_allo = pred_quats / (torch.norm(pred_quats, dim=1, keepdim=True) + eps)
    # quat_ego = allocentric_to_egocentric_torch(translation, quat_allo, eps=eps)
    # use numpy since it is more accurate
    if pred_rots.shape[-1] == 4 and pred_rots.ndim == 2:
        pred_quats = pred_rots.detach().cpu().numpy()  # allo
        ego_rot_preds = np.zeros((pred_quats.shape[0], 3, 3), dtype=np.float32)
        for i in range(pred_quats.shape[0]):
            # try:
            if is_allo:
                # this allows unnormalized quat
                cur_ego_mat = allocentric_to_egocentric(
                    RT_transform.quat_trans_to_pose_m(pred_quats[i], translation[i].detach().cpu().numpy()),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
            else:
                cur_ego_mat = RT_transform.quat_trans_to_pose_m(pred_quats[i], translation[i].detach().cpu().numpy())
            ego_rot_preds[i] = cur_ego_mat
            # except:

    # rot mat
    if pred_rots.shape[-1] == 3 and pred_rots.ndim == 3:
        pred_rots = pred_rots.detach().cpu().numpy()
        ego_rot_preds = np.zeros_like(pred_rots)
        for i in range(pred_rots.shape[0]):
            if is_allo:
                cur_ego_mat = allocentric_to_egocentric(
                    np.hstack([pred_rots[i], translation[i].detach().cpu().numpy().reshape(3, 1)]),
                    src_type="mat",
                    dst_type="mat",
                )[:3, :3]
            else:
                cur_ego_mat = pred_rots[i]
            ego_rot_preds[i] = cur_ego_mat
    return torch.from_numpy(ego_rot_preds), translation

def quatmul_torch(q1, q2):
    """Computes the multiplication of two quaternions.

    Note, output dims: NxMx4 with N being the batchsize and N the number
    of quaternions or 3D points to be transformed.
    """
    # RoI dimension. Unsqueeze if not fitting.
    a = q1.unsqueeze(0) if q1.dim() == 1 else q1
    b = q2.unsqueeze(0) if q2.dim() == 1 else q2

    # Corner dimension. Unsequeeze if not fitting.
    a = a.unsqueeze(1) if a.dim() == 2 else a
    b = b.unsqueeze(1) if b.dim() == 2 else b

    # Quaternion product
    x = a[:, :, 1] * b[:, :, 0] + a[:, :, 2] * b[:, :, 3] - a[:, :, 3] * b[:, :, 2] + a[:, :, 0] * b[:, :, 1]
    y = -a[:, :, 1] * b[:, :, 3] + a[:, :, 2] * b[:, :, 0] + a[:, :, 3] * b[:, :, 1] + a[:, :, 0] * b[:, :, 2]
    z = a[:, :, 1] * b[:, :, 2] - a[:, :, 2] * b[:, :, 1] + a[:, :, 3] * b[:, :, 0] + a[:, :, 0] * b[:, :, 3]
    w = -a[:, :, 1] * b[:, :, 1] - a[:, :, 2] * b[:, :, 2] - a[:, :, 3] * b[:, :, 3] + a[:, :, 0] * b[:, :, 0]

    return torch.stack((w, x, y, z), dim=2)

def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)

def allocentric_to_egocentric_torch(translation, q_allo, eps=1e-4):
    """Given an allocentric (object-centric) pose, compute new camera-centric
    pose Since we do detection on the image plane and our kernels are
    2D-translationally invariant, we need to ensure that rendered objects
    always look identical, independent of where we render them.

    Since objects further away from the optical center undergo skewing, we try to visually correct by
    rotating back the amount between optical center ray and object centroid ray.
    Another way to solve that might be translational variance (https://arxiv.org/abs/1807.03247)
    Args:
        translation: Nx3
        q_allo: Nx4
    """

    # Compute rotation between ray to object centroid and optical center ray
    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation / (torch.norm(translation, dim=1, keepdim=True) + eps)

    # cam_ray.dot(obj_ray), assume cam_ray: (0, 0, 1)
    angle = obj_ray[:, 2:3].acos()

    # Compute rotation between ray to object centroid and optical center ray
    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)

    # Build quaternion representing the rotation around the computed axis
    # angle-axis => quat
    q_allo_to_ego = cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )

    # Apply quaternion for transformation from allocentric to egocentric.
    q_ego = quatmul_torch(q_allo_to_ego, q_allo)[:, 0]  # Remove added Corner dimension here.
    return q_ego

def quat2mat_torch(quat, eps=1e-8):
    """Convert quaternion coefficients to rotation matrix.

    Args:
        quat: [B*T, 4]
    Returns:
        Rotation matrix: [B*T, 3, 3]
    """
    assert quat.ndim == 2 and quat.shape[1] == 4, f"Expected (B*T,4), got {quat.shape}"

    norm_quat = quat / (quat.norm(p=2, dim=1, keepdim=True) + eps)
    qw, qx, qy, qz = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    R = torch.empty((B, 3, 3), dtype=quat.dtype, device=quat.device)

    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)

    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)

    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)

    return R  # (B*T, 3, 3)

def allo_to_ego_mat_torch(translation, rot_allo, eps=1e-4):
    # translation: B×T×3
    # rot_allo: B×T×3×3
    B, T = translation.shape[:2]

    # Flatten to (B*T, ...)
    translation_flat = translation.view(-1, 3)  # (B*T, 3)
    rot_allo_flat = rot_allo.view(-1, 3, 3)     # (B*T, 3, 3)

    cam_ray = torch.tensor([0, 0, 1.0], dtype=translation.dtype, device=translation.device)  # (3,)
    obj_ray = translation_flat / (torch.norm(translation_flat, dim=1, keepdim=True) + eps)   # (B*T, 3)

    angle = obj_ray[:, 2:3].clamp(-1.0, 1.0).acos()  # numerical stability

    axis = torch.cross(cam_ray.expand_as(obj_ray), obj_ray, dim=1)
    axis = axis / (torch.norm(axis, dim=1, keepdim=True) + eps)  # (B*T, 3)

    # quaternion [B*T, 4]
    q_allo_to_ego = torch.cat(
        [
            torch.cos(angle / 2.0),
            axis[:, 0:1] * torch.sin(angle / 2.0),
            axis[:, 1:2] * torch.sin(angle / 2.0),
            axis[:, 2:3] * torch.sin(angle / 2.0),
        ],
        dim=1,
    )

    rot_allo_to_ego = quat2mat_torch(q_allo_to_ego)  # (B*T, 3, 3)

    # rot_ego = rot_allo_to_ego @ rot_allo
    rot_ego = torch.bmm(rot_allo_to_ego, rot_allo_flat)  # (B*T, 3, 3)

    # Reshape back to (B, T, 3, 3)
    return rot_ego.view(B, T, 3, 3)

def pose_from_predictions_train(
    pred_rots,    # B*T*3*3
    pred_centroids,  # B*T*2
    pred_z_vals,  # B*T*1
    roi_cams,  # B*T*3*3
    roi_centers, # B*T*2
    resize_ratios, # B*T
    roi_whs,  # B*T*2
    eps=1e-4,
):
    """for train
    Args:
        pred_rots:
        pred_centroids:
        pred_z_vals: [B, 1]
        roi_cams: absolute cams
        roi_centers:
        roi_scales:
        roi_whs: (bw,bh) for bboxes
        eps:
        is_allo:
        z_type: REL | ABS | LOG | NEG_LOG

    Returns:

    """
    assert roi_cams.dim() == 4, roi_cams.dim()
    # absolute coords
    c = torch.stack(
        [
            (pred_centroids[..., 0] * roi_whs[..., 0]) + roi_centers[..., 0],
            (pred_centroids[..., 1] * roi_whs[..., 1]) + roi_centers[..., 1],
        ],
        dim=-1,
    )

    cx = c[..., 0:1]  # [#roi, 1]
    cy = c[..., 1:2]  # [#roi, 1]

    # unnormalize regressed z
    # z_1 / z_2 = s_2 / s_1 ==> z_1 = s_2 / s_1 * z_2
    z = pred_z_vals * resize_ratios.unsqueeze(2)

    # backproject regressed centroid with regressed z
    """
    fx * tx + px * tz = z * cx
    fy * ty + py * tz = z * cy
    tz = z
    ==>
    fx * tx / tz = cx - px
    fy * ty / tz = cy - py
    ==>
    tx = (cx - px) * tz / fx
    ty = (cy - py) * tz / fy
    """
    # NOTE: z must be [B,T,1]
    translation = torch.cat(
        [
            z * (cx - roi_cams[..., 0, 2:3]) / roi_cams[..., 0, 0:1],
            z * (cy - roi_cams[..., 1, 2:3]) / roi_cams[..., 1, 1:2],
            z
        ],
        dim=-1,  # 原来是 dim=1，现在 dim=-1 因为我们要拼出 B×T×3
    )

    rot_ego = allo_to_ego_mat_torch(translation, pred_rots, eps=eps)

    return rot_ego, translation