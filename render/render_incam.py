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
from renderer import Renderer


def render_incam(video_name, smplh_model_path, metadata_path, frames_dir, cam_intri, ref, results_path, save_dir="outputs/render_incam"):
    output_dir = Path(save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_name}_render.mp4"

    
    # Load predictions
    print(f"Loading predictions from {results_path}") 
    with open(results_path, 'rb') as f:
        pred = pickle.load(f)
    
    # Check if video_name exists in predictions
    if video_name not in pred:
        print(f"Video {video_name} not found in predictions")
        return
    
    video_pred = pred[video_name]

    metadata = torch.load(metadata_path)
    
    # Get gender and object name from metadata
    if video_name in metadata:
        gender = metadata[video_name].get('gender', 'neutral')
        obj_name = metadata[video_name].get('obj_name', None)
    else:
        gender = 'neutral'
        obj_name = None
        print(f"Warning: No metadata found for {video_name}, using defaults")
    
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
    
    # Load object templates from InterCap
    obj_templates_path = Path(ref)
    with open(obj_templates_path, 'rb') as f:
        obj_data = pickle.load(f)
        obj_templates = obj_data['templates']
    
    # Camera intrinsics
    K = np.array(cam_intri)
    
    # Get frame count
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    num_frames = len(frame_files)
    
    # Extract predictions for this video
    hum_pose_aa = video_pred['pose']  # [T, 66] - axis-angle format
    hum_betas = video_pred['betas']    # [T, 10]
    hum_trans = video_pred['trans'] # [T, 3]
    obj_rot_mat = video_pred['obj_rot']    # [T, 3, 3] - rotation matrix
    obj_trans = video_pred['obj_trans'] # [T, 3]
    
    # Use object name from metadata if available, otherwise try to get from predictions
    if obj_name is None:
        obj_id = video_pred['obj_id'][0] if 'obj_id' in video_pred else 0
        # Fallback to inverse transformation if needed
        from utils.data_utils import inverse_trans_obj_id
        obj_name = inverse_trans_obj_id(obj_id, dataset='intercap')
    
    # Get object template and faces
    if obj_name in obj_templates:
        obj_template = obj_templates[obj_name]['verts']
        obj_faces = obj_templates[obj_name]['faces']
    else:
        print(f"Warning: Object {obj_name} not found in templates")
        # Use first available object as fallback
        first_obj = list(obj_templates.keys())[0]
        obj_template = obj_templates[first_obj]['verts']
        obj_faces = obj_templates[first_obj]['faces']
        print(f"Using fallback object: {first_obj}")
    
    # Setup renderer
    # Get image dimensions from first frame
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    first_frame = cv2.imread(first_frame_path)
    height, width = first_frame.shape[:2]
    
    renderer = Renderer(width, height, device="cuda", faces=smplh_faces, K=K, bin_size=None)  # 禁用binning
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_video_path), fourcc, 30.0, (width, height))
    
    print(f"Rendering {num_frames} frames for video {video_name}")
    
    for i in tqdm(range(min(num_frames, len(hum_pose_aa))), desc="Rendering frames"):
        if i % 20 == 0: 
            torch.cuda.empty_cache()
            
        # Load frame
        frame_path = os.path.join(frames_dir, f"{str(i).zfill(4)}.jpg")
        if not os.path.exists(frame_path):
            continue
        img = cv2.imread(frame_path)
        
        # Convert poses to SMPL format
        pose_aa_frame = torch.from_numpy(hum_pose_aa[i:i+1]).float().cuda()  # [1, 66] - already in axis-angle
        betas_frame = torch.from_numpy(hum_betas[i:i+1]).float().cuda()    # [1, 10]
        trans_frame = torch.from_numpy(hum_trans[i:i+1]).float().cuda()    # [1, 3]
        
        # Create zero hand poses explicitly since model was created with create_hand_pose=False
        left_hand_pose = torch.zeros(1, 45).cuda()   # 15 joints * 3 = 45
        right_hand_pose = torch.zeros(1, 45).cuda()  # 15 joints * 3 = 45
        
        # SMPL-H forward pass - provide all required parameters
        smpl_output = smpl_layer(
            betas=betas_frame,
            body_pose=pose_aa_frame[:, 3:66],  # [1, 63] - body pose only
            global_orient=pose_aa_frame[:, :3],  # [1, 3] - global orientation
            left_hand_pose=left_hand_pose,   # [1, 45] - zero hand pose
            right_hand_pose=right_hand_pose, # [1, 45] - zero hand pose
        )
        
        # Get human vertices in camera space
        hum_verts = smpl_output.vertices[0] + trans_frame[0]  # [V, 3]
        
        # Get object vertices - rotation is already a matrix
        obj_rot_frame = torch.from_numpy(obj_rot_mat[i]).float().cuda()  # [3, 3] - rotation matrix
        obj_trans_frame = torch.from_numpy(obj_trans[i:i+1]).float().cuda()  # [1, 3]
        
        # Apply rotation and translation to object template
        obj_verts_template = torch.tensor(obj_template).float().cuda()  # [V, 3]
        
        # Apply rotation matrix directly (no conversion needed)
        obj_verts = torch.matmul(obj_verts_template, obj_rot_frame.T) + obj_trans_frame[0]  # [V, 3]
        
        try:
            img_rendered = renderer.render_mesh(hum_verts, img, colors=[0.2, 0.8, 0.3])  # Green for human
            
            obj_renderer = Renderer(width, height, device="cuda", faces=obj_faces, K=K, bin_size=None)
            img_rendered = obj_renderer.render_mesh(obj_verts, img_rendered, colors=[0.8, 0.2, 0.2])  # Red for object
            
            # Write frame
            writer.write(img_rendered)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"GPU out of memory at frame {i}, skipping frame and clearing cache...")
                torch.cuda.empty_cache()
                writer.write(img)
                continue
            else:
                raise e
    
    writer.release()
    print(f"Video saved to {output_video_path}")
