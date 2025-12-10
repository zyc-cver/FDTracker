import torch
import trimesh
import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import os

from core.config import cfg
from models.templates import smplh


def eval_chamfer_distance(pred_human, target_human, pred_object, target_object, object_faces, sample_num=6000):
    pred_human, target_human = pred_human.copy(), target_human.copy()
    pred_object, target_object, object_faces = pred_object.copy(), target_object.copy(), object_faces.copy()
    batch_size = pred_human.shape[0]

    for j in range(batch_size):
        pred_mesh = np.concatenate((pred_human[j], pred_object[j]))
        target_mesh = np.concatenate((target_human[j], target_object[j]))

        pred_mesh = rigid_align(pred_mesh, target_mesh)
        pred_human[j], pred_object[j] = pred_mesh[:len(pred_human[j]),:], pred_mesh[len(pred_human[j]):,:]
        target_human[j], target_object[j] = target_mesh[:len(target_human[j]),:], target_mesh[len(target_human[j]):,:]
    
    human_chamfer_dist = []
    for j in range(batch_size):
        pred_mesh = trimesh.Trimesh(pred_human[j], smplh.faces, process=False, maintain_order=True)
        target_mesh = trimesh.Trimesh(target_human[j], smplh.faces, process=False, maintain_order=True)

        pred_verts, target_verts = pred_mesh.sample(sample_num), target_mesh.sample(sample_num)
        dist = chamfer_distance(target_verts, pred_verts)
        human_chamfer_dist.append(dist)

    object_chamfer_dist = []
    for j in range(batch_size):
        pred_mesh = trimesh.Trimesh(pred_object[j], object_faces[j], process=False, maintain_order=True)
        target_mesh = trimesh.Trimesh(target_object[j], object_faces[j], process=False, maintain_order=True)

        pred_verts, target_verts = pred_mesh.sample(sample_num), target_mesh.sample(sample_num)
        dist = chamfer_distance(target_verts, pred_verts)
        object_chamfer_dist.append(dist)

    return np.array(human_chamfer_dist), np.array(object_chamfer_dist)


def eval_contact_score(pred_human, pred_object, target_h_contacts, metric='l2'):
    pred_human, pred_object = pred_human.copy(), pred_object.copy()
    batch_size = pred_human.shape[0]

    precision_list = []
    recall_list = []
    for j in range(batch_size):
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(pred_object[j])
        min_y_to_x = x_nn.kneighbors(pred_human[j])[0].squeeze()
        pred_contacts = (min_y_to_x < cfg.TEST.contact_thres)
        pred_idxs = np.where(pred_contacts)[0]

        target_contacts = (target_h_contacts[j] == 1).numpy()
        target_idxs = np.where(target_contacts)[0]
        true_positive = (pred_contacts * target_contacts).sum()

        if len(pred_idxs) > 0:
            precision_list.append(true_positive / len(pred_idxs))
        if len(target_idxs) > 0:
            recall_list.append(true_positive / len(target_idxs))

    return np.array(precision_list), np.array(recall_list)


def eval_contact_estimation(h_contacts, o_contacts, target_h_contacts, target_o_contacts):
    h_contacts, o_contacts = h_contacts.copy(), o_contacts.copy()
    batch_size = h_contacts.shape[0]

    precision_list = []
    recall_list = []
    for j in range(batch_size):
        pred_contacts = smplh.upsample(torch.tensor(h_contacts[j])).numpy() > 0.5
        pred_idxs = np.where(pred_contacts)[0]

        target_contacts = (target_h_contacts[j] == 1).numpy()
        target_idxs = np.where(target_contacts)[0]
        true_positive = (pred_contacts * target_contacts).sum()

        if len(pred_idxs) > 0:
            precision_list.append(true_positive / len(pred_idxs))
        if len(target_idxs) > 0:
            recall_list.append(true_positive / len(target_idxs))

    precision_list2 = []
    recall_list2 = []
    for j in range(batch_size):
        pred_contacts = o_contacts[j] > 0.5
        pred_idxs = np.where(pred_contacts)[0]

        target_contacts = (target_o_contacts[j] == 1).numpy()
        target_idxs = np.where(target_contacts)[0]
        true_positive = (pred_contacts * target_contacts).sum()

        if len(pred_idxs) > 0:
            precision_list2.append(true_positive / len(pred_idxs))
        if len(target_idxs) > 0:
            recall_list2.append(true_positive / len(target_idxs))

    return np.array(precision_list), np.array(recall_list), np.array(precision_list2), np.array(recall_list2)


def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default l2
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||_metric}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||_metric}}

        this is the squared root distance, while pytorch3d is the squared distance
    """

    if direction == 'y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == 'x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == 'bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y) # bidirectional errors are accumulated
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist


def rigid_transform_3D(A, B):
    n, dim = A.shape
    centroid_A = np.mean(A, axis = 0)
    centroid_B = np.mean(B, axis = 0)
    H = np.dot(np.transpose(A - centroid_A), B - centroid_B) / n
    U, s, V = np.linalg.svd(H)
    R = np.dot(np.transpose(V), np.transpose(U))
    if np.linalg.det(R) < 0:
        s[-1] = -s[-1]
        V[2] = -V[2]
        R = np.dot(np.transpose(V), np.transpose(U))

    varP = np.var(A, axis=0).sum()
    c = 1/varP * np.sum(s)

    t = -np.dot(c*R, np.transpose(centroid_A)) + np.transpose(centroid_B)
    return c, R, t


def rigid_align(A, B):
    c, R, t = rigid_transform_3D(A, B)
    A2 = np.transpose(np.dot(c*R, np.transpose(A))) + t
    return A2


def reconstruct_video_predictions(model_outputs, dataset):
    videos_pred = defaultdict(dict)
    
    for idx, outputs in enumerate(model_outputs):
        # Retrieve original video and frame information
        meta = dataset.idx2meta[idx]
        vid_name = meta['vid_name']
        start_idx = meta['start_idx']
        end_idx = meta['end_idx']
        is_last = meta['is_last']
        orig_len = meta['orig_len']
        
        # Initialize container for the video if encountered for the first time
        if 'init' not in videos_pred[vid_name]:
            # Create storage containers for each output
            videos_pred[vid_name]['init'] = True
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    # Initialize with appropriately sized empty tensors
                    videos_pred[vid_name][k] = torch.zeros((orig_len,) + v.shape[1:], device='cpu')
        
        # Compute actual frame indices
        if is_last and end_idx - start_idx < dataset.motion_frames:
            # For the last overlapping segment, only take the non-overlapping part
            actual_start = max(0, orig_len - dataset.motion_frames)
            slice_offset = actual_start - start_idx
            frame_indices = range(actual_start, orig_len)
        else:
            # For non-overlapping parts, directly take the indices
            frame_indices = range(start_idx, end_idx)
            slice_offset = 0
            
        # Place the prediction results into the corresponding frame positions
        for t_rel, t_abs in enumerate(frame_indices):
            for k, v in outputs.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    t_idx = t_rel + slice_offset
                    if t_idx < v.shape[0]:  # Ensure the index is within range
                        videos_pred[vid_name][k][t_abs] = v[t_idx].cpu()
    
    # Remove temporary initialization markers
    for vid_name in videos_pred:
        if 'init' in videos_pred[vid_name]:
            del videos_pred[vid_name]['init']
            
    return videos_pred

def compute_video_metrics(videos_pred, videos_gt, metadata=None):
    metrics = {
        'per_video': {},
        'overall': {
            'hum_trans_error': 0.0,
            'obj_trans_error': 0.0,
            'total_frames': 0
        }
    }
    
    for vid_name in videos_pred:
        # Skip metric computation if there is no ground truth
        if vid_name not in videos_gt:
            print(f"Warning: Video {vid_name} has no ground truth, skipping evaluation")
            continue
            
        new_pred = videos_pred[vid_name]
        vid_gt = videos_gt[vid_name]
        n_frames = new_pred['hum_trans_cam'].shape[0]
        
        # Compute human translation error
        if 'hum_trans_cam' in new_pred and 'smplh_trans' in vid_gt:
            hum_trans_error = torch.nn.functional.mse_loss(
                new_pred['hum_trans_cam'], 
                torch.tensor(vid_gt['smplh_trans'], device=new_pred['hum_trans_cam'].device)
            ).item()
        else:
            hum_trans_error = float('nan')
            
        # Compute object translation error
        if 'obj_trans_cam' in new_pred and 'obj_trans' in vid_gt:
            obj_trans_error = torch.nn.functional.mse_loss(
                new_pred['obj_trans_cam'], 
                torch.tensor(vid_gt['obj_trans'], device=new_pred['obj_trans_cam'].device)
            ).item()
        else:
            obj_trans_error = float('nan')
            
        # Store metrics for each video
        metrics['per_video'][vid_name] = {
            'hum_trans_error': hum_trans_error,
            'obj_trans_error': obj_trans_error,
            'n_frames': n_frames
        }
        
        # Accumulate overall metrics
        if not np.isnan(hum_trans_error):
            metrics['overall']['hum_trans_error'] += hum_trans_error * n_frames
        if not np.isnan(obj_trans_error):
            metrics['overall']['obj_trans_error'] += obj_trans_error * n_frames
        metrics['overall']['total_frames'] += n_frames
    
    # Compute weighted averages
    if metrics['overall']['total_frames'] > 0:
        metrics['overall']['hum_trans_error'] /= metrics['overall']['total_frames']
        metrics['overall']['obj_trans_error'] /= metrics['overall']['total_frames']
    
    return metrics

def save_video_predictions(videos_pred, save_dir):
    """
    Save video prediction results
    
    Args:
        videos_pred: Prediction results organized by video
        save_dir: Directory to save the results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save predictions for all videos
    save_path = os.path.join(save_dir, 'all_predictions.pkl')
    torch.save(videos_pred, save_path)
    print(f"Saved all video prediction results to {save_path}")


def convert_predictions_for_evaluation(videos_pred):

    new_pred = {}
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
    
    for vid_name, pred in videos_pred.items():

        new_pred[vid_name] = {}
        
        # human pose 6d to axis-angle
        pose6d = pred['hum_pose6d']
        pose = np.zeros((pose6d.shape[0], 156))
        temp = matrix_to_axis_angle(rotation_6d_to_matrix(pose6d.reshape(pose6d.shape[0], -1, 6))).numpy().reshape(pose6d.shape[0], -1)
        pose[:,:66] = temp
        new_pred[vid_name]['pose'] = pose
        
        # human trans
        new_pred[vid_name]['trans'] = pred['hum_trans_cam'].numpy()
        
        # human betas
        new_pred[vid_name]['betas'] = pred['hum_betas'].numpy()
        
        # obj_rot 6d to matrix
        obj_rot_mat = pred['obj_rot_mat']
        new_pred[vid_name]['obj_rot'] = obj_rot_mat.reshape(-1, 3, 3).numpy()
        
        # obj trans
        new_pred[vid_name]['obj_trans'] = pred['obj_trans_cam'].numpy()
    
    return new_pred

def convert_predictions_for_evaluation_with_init(videos_pred):
    new_pred = {}
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle
    
    for vid_name, pred in videos_pred.items():
        new_pred[vid_name] = {}
        
        pose6d = pred['hum_pose6d']
        pose = np.zeros((pose6d.shape[0], 156))
        temp = matrix_to_axis_angle(rotation_6d_to_matrix(pose6d.reshape(pose6d.shape[0], -1, 6))).numpy().reshape(pose6d.shape[0], -1)
        pose[:,:66] = temp
        new_pred[vid_name]['pose'] = pose
        
        # human trans
        new_pred[vid_name]['trans'] = pred['hum_trans_cam'].numpy()
        
        # human betas
        new_pred[vid_name]['betas'] = pred['hum_betas'].numpy()
        
        # obj_rot 6d to matrix
        obj_rot_mat = pred['obj_rot_mat']
        new_pred[vid_name]['obj_rot'] = obj_rot_mat.reshape(-1, 3, 3).numpy()
        
        # obj trans
        new_pred[vid_name]['obj_trans'] = pred['obj_trans_cam'].numpy()

        init_pose6d = pred['init_hum_pose6d']
        init_pose = np.zeros((init_pose6d.shape[0], 156))
        init_temp = matrix_to_axis_angle(rotation_6d_to_matrix(init_pose6d.reshape(init_pose6d.shape[0], -1, 6))).numpy().reshape(init_pose6d.shape[0], -1)
        init_pose[:,:66] = init_temp
        new_pred[vid_name]['init_pose'] = init_pose

        # human trans
        new_pred[vid_name]['init_trans'] = pred['init_hum_trans_cam'].numpy()
        
        # human betas
        new_pred[vid_name]['init_betas'] = pred['init_hum_betas'].numpy()
        
        # obj_rot 6d to matrix
        init_obj_rot_mat = pred['init_obj_rot_mat']
        new_pred[vid_name]['init_obj_rot'] = init_obj_rot_mat.reshape(-1, 3, 3).numpy()
        
        # obj trans
        new_pred[vid_name]['init_obj_trans'] = pred['init_obj_trans_cam'].numpy()

    return new_pred

def process_predictions_by_video(batch_outputs, dataset, current_batch_idx):
    if not hasattr(process_predictions_by_video, "videos_pred"):
        process_predictions_by_video.videos_pred = {}
        process_predictions_by_video.is_initialized = {}
    
    batch_size = len(batch_outputs)

    start_idx = current_batch_idx * batch_size
    end_idx = min(start_idx + batch_size, len(dataset))

    for batch_item_idx, global_idx in enumerate(range(start_idx, end_idx)):
        if batch_item_idx >= len(batch_outputs):
            break
            
        if not hasattr(dataset, 'idx2meta') or global_idx not in dataset.idx2meta:
            continue
            
        meta = dataset.idx2meta[global_idx]
        vid_name = meta['vid_name']
        start_frame = meta['start_idx']
        end_frame = meta['end_idx']
        is_last = meta.get('is_last', False)
        orig_len = meta.get('orig_len', end_frame)
        
        pred = batch_outputs[batch_item_idx]
        
        if vid_name not in process_predictions_by_video.is_initialized:
            process_predictions_by_video.videos_pred[vid_name] = {}
            process_predictions_by_video.is_initialized[vid_name] = True
            
            for k, v in pred.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    process_predictions_by_video.videos_pred[vid_name][k] = torch.zeros(
                        (orig_len,) + v.shape[1:], 
                        device='cpu',
                        dtype=v.dtype
                    )
        
        frame_indices = range(start_frame, end_frame)
        
        for t_rel, t_abs in enumerate(frame_indices):
            if t_rel >= pred['trans'].shape[0]:
                continue
                
            for k, v in pred.items():
                if isinstance(v, torch.Tensor) and len(v.shape) >= 2:
                    if t_abs < process_predictions_by_video.videos_pred[vid_name][k].shape[0]:
                        process_predictions_by_video.videos_pred[vid_name][k][t_abs] = v[t_rel].cpu()
    
    return process_predictions_by_video.videos_pred

def reset_video_predictions():
    if hasattr(process_predictions_by_video, "videos_pred"):
        delattr(process_predictions_by_video, "videos_pred")
        delattr(process_predictions_by_video, "is_initialized")
        delattr(process_predictions_by_video, "all_indices")