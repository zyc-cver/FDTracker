import os
import cv2
import sys
sys.path.append('lib')
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
import smplx
from core.config import cfg
from renderer import Renderer, get_global_cameras_static, get_ground_params_from_points


def render_global(metadata, obj_templates, video_name, video_pred, save_path):
    
    # Get gender and object name from metadata
    if video_name in metadata:
        gender = metadata[video_name].get('gender', 'neutral')
        obj_name = metadata[video_name].get('obj_name', None)

    smplh_model_path = 'data/base_data/human_models/mano_v1_2'
    # Create SMPL layers with proper configuration
    layer_arg = {'create_global_orient': False, 'create_body_pose': False, 'create_left_hand_pose': False, 'create_right_hand_pose': False, 'create_betas': False, 'create_transl': False, 'use_pca': False}
    smpl_layers = {
        'neutral': smplx.create(smplh_model_path, 'smplh', gender='NEUTRAL', **layer_arg).cuda(), 
        'male': smplx.create(smplh_model_path, 'smplh', gender='MALE', **layer_arg).cuda(), 
        'female': smplx.create(smplh_model_path, 'smplh', gender='FEMALE', **layer_arg).cuda()
    }
    
    # Select the appropriate SMPL layer based on gender
    gender_key = gender.lower() if gender.lower() in ['male', 'female'] else 'neutral'
    smpl_layer = smpl_layers[gender_key]
    smplh_faces = smpl_layer.faces

    
    # Extract predictions for this video
    hum_pose_aa = video_pred['pose'][:]  # [T, 66] - axis-angle format
    hum_betas = video_pred['betas'][:]    # [T, 10]
    hum_trans = video_pred['trans'][:] # [T, 3]
    obj_rot_mat = video_pred['obj_rot'][:]    # [T, 3, 3] - rotation matrix
    obj_trans = video_pred['obj_trans'][:] # [T, 3]
    
    # Use object name from metadata if available
    if obj_name is None:
        obj_id = video_pred['obj_id'][0] if 'obj_id' in video_pred else 0
        from utils.data_utils import inverse_trans_obj_id

        obj_name = inverse_trans_obj_id(obj_id, dataset='intercap')
    
    # Get object template and faces
    if obj_name in obj_templates:
        obj_template = obj_templates[obj_name]['verts']
        obj_faces = obj_templates[obj_name]['faces']
    else:
        print(f"Warning: Object {obj_name} not found in templates")
        first_obj = list(obj_templates.keys())[0]
        obj_template = obj_templates[first_obj]['verts']
        obj_faces = obj_templates[first_obj]['faces']
        print(f"Using fallback object: {first_obj}")
    
    # Generate all human vertices for global view planning
    num_frames = len(hum_pose_aa)
    all_hum_verts = []
    all_obj_verts = []
    all_foot_keypoints = []  # Used to calculate ground height
    
    print("Generating vertices for global camera planning...")
    for i in range(num_frames):
        # Convert poses to SMPL format
        pose_aa_frame = torch.from_numpy(hum_pose_aa[i:i+1]).float().cuda()
        betas_frame = torch.from_numpy(hum_betas[i:i+1]).float().cuda()
        trans_frame = torch.from_numpy(hum_trans[i:i+1]).float().cuda()
        
        # Create zero hand poses
        left_hand_pose = torch.zeros(1, 45).cuda()
        right_hand_pose = torch.zeros(1, 45).cuda()
        
        # SMPL forward pass
        smpl_output = smpl_layer(
            betas=betas_frame,
            body_pose=pose_aa_frame[:, 3:66],
            global_orient=pose_aa_frame[:, :3],
            left_hand_pose=left_hand_pose,
            right_hand_pose=right_hand_pose,
        )
        
        # Adjust coordinate system: make the human upright and adjust transformations
        hum_verts = smpl_output.vertices[0] + trans_frame[0]
        # Coordinate transformation: (x, y, z) -> (-z, -y, x) to make the human upright and facing the correct direction
        hum_verts_corrected = torch.stack([
            -hum_verts[:, 2],  # x = -z 
            -hum_verts[:, 1],  # y = -y (vertical flip)
            hum_verts[:, 0]    # z = x 
        ], dim=1)
        all_hum_verts.append(hum_verts_corrected)
        
        # Get foot keypoints for ground calculation (left ankle 7, right ankle 8)
        joints = smpl_output.joints[0] + trans_frame[0]
        foot_joints = joints[[7, 8]]  # Ankle keypoints
        foot_joints_corrected = torch.stack([
            -foot_joints[:, 2],
            -foot_joints[:, 1],  # Vertical flip
            foot_joints[:, 0]
        ], dim=1)
        all_foot_keypoints.append(foot_joints_corrected)
        
        # Get object vertices with same coordinate correction
        obj_rot_frame = torch.from_numpy(obj_rot_mat[i]).float().cuda()
        obj_trans_frame = torch.from_numpy(obj_trans[i:i+1]).float().cuda()
        obj_verts_template = torch.tensor(obj_template).float().cuda()
        obj_verts = torch.matmul(obj_verts_template, obj_rot_frame.T) + obj_trans_frame[0]
        
        # Apply the same coordinate transformation to the object
        obj_verts_corrected = torch.stack([
            -obj_verts[:, 2],
            -obj_verts[:, 1],  # Vertical flip
            obj_verts[:, 0]
        ], dim=1)
        all_obj_verts.append(obj_verts_corrected)
    
    # Stack all vertices
    all_hum_verts = torch.stack(all_hum_verts)  # [T, V, 3]
    all_obj_verts = torch.stack(all_obj_verts)  # [T, V_obj, 3]
    all_foot_keypoints = torch.stack(all_foot_keypoints)  # [T, 2, 3]
    
    # Calculate ground height: based on the lowest point of foot keypoints, then lower slightly
    ground_height = all_foot_keypoints[:, :, 1].min().item() - 0.05  # Lower by 5cm
    print(f"Ground height set to: {ground_height:.3f}")
    
    # Lift human and object trajectories above the ground by an appropriate height
    lift_height = -ground_height + 0.02  # Slightly lift by 2cm
    all_hum_verts[:, :, 1] += lift_height
    all_obj_verts[:, :, 1] += lift_height
    
    # Combine human and object vertices for camera planning
    combined_verts = torch.cat([all_hum_verts, all_obj_verts], dim=1)  # [T, V+V_obj, 3]
    
    # Set better camera parameters: top-down view, farther distance, moderate height
    global_R, global_T, global_lightsm = get_global_cameras_static(
        combined_verts.cpu(),
        beta=5.0,              # Increase distance multiplier for farther camera
        cam_height_degree=15,  # Lower tilt angle for a lower viewpoint (originally 25)
        target_center_height=0.8,  # Target height set near the waist
        use_long_axis=False,
        vec_rot=135,           # Adjust horizontal angle for a better oblique view
    )
    
    # Setup renderer with global view
    width, height = 1280, 720  # Lower resolution to reduce memory usage (originally 1920x1080)
    # Create camera intrinsics for global view - Adjust focal length for a wider field of view
    focal_length = 35  # Lower focal length for a wider field of view
    K = np.array([[focal_length * width / 36, 0, width / 2],
                  [0, focal_length * height / 24, height / 2],
                  [0, 0, 1.]])
    
    # Set bin_size to reduce memory usage, but ensure it's large enough to fit all faces
    renderer = Renderer(width, height, device="cuda", faces=smplh_faces, K=K, bin_size=128)  # Use larger bin_size
    
    # Setup ground plane - Set at the calculated ground height, make the ground larger
    # Use the center of the human trajectory as the ground center
    trajectory_center = combined_verts.mean(dim=[0, 1])  # [3] Average position of all vertices
    trajectory_range = combined_verts.reshape(-1, 3)
    x_range = trajectory_range[:, 0].max() - trajectory_range[:, 0].min()
    z_range = trajectory_range[:, 2].max() - trajectory_range[:, 2].min()
    ground_scale = max(x_range.item(), z_range.item()) * 1.5  # Reduce ground range to save memory
    
    renderer.set_ground(
        ground_scale, 
        trajectory_center[0].item(),  # x center
        trajectory_center[2].item()   # z center
    )
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(save_path), fourcc, 30.0, (width, height))
    
    print(f"Rendering {num_frames} frames for global view of {video_name}")
    
    for i in tqdm(range(num_frames), desc="Rendering global frames"):
        # Clear GPU cache
        if i % 10 == 0:  # Every 10 frames
            torch.cuda.empty_cache()
            
        # Create camera for this frame - Ensure camera parameters are on GPU
        if isinstance(global_R[i], np.ndarray):
            cam_R = torch.from_numpy(global_R[i]).float().cuda()
        else:
            cam_R = global_R[i].cuda()
        
        if isinstance(global_T[i], np.ndarray):
            cam_T = torch.from_numpy(global_T[i]).float().cuda()
        else:
            cam_T = global_T[i].cuda()
            
        cameras = renderer.create_camera(cam_R, cam_T)
        
        # Render human and object
        hum_verts_frame = all_hum_verts[i]  # [V, 3]
        obj_verts_frame = all_obj_verts[i]  # [V_obj, 3]
        
        # Prepare vertices and colors for human and object
        frame_verts = torch.cat([hum_verts_frame.unsqueeze(0), obj_verts_frame.unsqueeze(0)], dim=1)
        hum_color = torch.tensor([0.2, 0.8, 0.3]).cuda()  # Green for human
        obj_color = torch.tensor([0.8, 0.2, 0.2]).cuda()  # Red for object
        hum_colors = hum_color.unsqueeze(0).unsqueeze(0).repeat(1, hum_verts_frame.shape[0], 1)
        obj_colors = obj_color.unsqueeze(0).unsqueeze(0).repeat(1, obj_verts_frame.shape[0], 1)
        frame_colors = torch.cat([hum_colors, obj_colors], dim=1)
        
        # Combine faces for human and object
        if isinstance(smplh_faces, np.ndarray):
            hum_faces = torch.from_numpy(smplh_faces.astype(np.int32)).cuda()
        else:
            hum_faces = smplh_faces.clone()
        
        obj_faces_offset = torch.tensor(obj_faces, dtype=torch.int32).cuda() + hum_verts_frame.shape[0]
        combined_faces = torch.cat([hum_faces.unsqueeze(0), obj_faces_offset.unsqueeze(0)], dim=1)
        
        try:
            # Render human, object, and ground with simple lighting
            img = renderer.render_with_ground(
                frame_verts, 
                frame_colors, 
                cameras, 
                renderer.lights,  # Use default simple lighting
                faces=combined_faces
            )
            writer.write(img)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU out of memory at frame {i}, skipping frame and clearing cache...")
                torch.cuda.empty_cache()
                # Create a black frame as a placeholder
                black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                writer.write(black_frame)
                continue
            else:
                raise e
    
    writer.release()
    print(f"Global video saved to {save_path}")


def render_all_videos_global(results_path, ref_path, metadata_path, save_dir="outputs/vis/render_global"):
    """
    Render all videos in global view
    """
    
    os.makedirs(save_dir, exist_ok=True)

    with open(results_path, 'rb') as f:
        pred = pickle.load(f)
    
    metadata = torch.load(metadata_path)

    with open(ref_path, 'rb') as f:
        obj_data = pickle.load(f)
        obj_templates = obj_data['templates']
    
    for video_name in pred.keys():
        print(f"Rendering global view for video: {video_name}")
        
        if video_name not in metadata:
            print(f"Warning: No metadata found for {video_name}, skipping")
            continue
        try:
            render_global(metadata, obj_templates, video_name, pred[video_name], f"{save_dir}/{video_name}_global.mp4")
        except Exception as e:
            print(f"Error rendering global view for {video_name}: {e}")
            continue


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render videos in global view.")
    parser.add_argument("--results-path", type=str, required=True, help="Path to the results file.")
    parser.add_argument("--ref_path", type=str, default="data/base_data/ref_hoi.pkl", help="Path to the object reference file.")
    parser.add_argument("--metadata-path", type=str, default="demo/val_data/metadata.pkl", required=True, help="Path to the metadata file.")
    parser.add_argument("--save_dir", type=str, default="outputs/vis/render_global", help="Directory to save rendered videos.")

    args = parser.parse_args()

    render_all_videos_global(
        results_path=args.results_path,
        ref_path=args.ref_path,
        metadata_path=args.metadata_path,
        save_dir=args.save_dir
    )