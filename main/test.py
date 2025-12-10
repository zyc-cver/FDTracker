import os
import sys
import argparse
import torch
import smplx
import numpy as np
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)
sys.path.append('lib')
from core.config import cfg, update_config
from models.model import get_model
from utils.validation import validate_model_test
from utils.train_utils_vid import get_dataloader, check_data_parallel
from loss.loss_total import get_loss_weights
from utils.data_utils import inverse_trans_obj_id

def init_smpl_layers():
    layer_args = {
        'create_global_orient': False, 
        'create_body_pose': False, 
        'create_left_hand_pose': True, 
        'create_right_hand_pose': True, 
        'create_betas': False, 
        'create_transl': False, 
        'use_pca': False
    }

    smpl_layers = {
        'NEUTRAL': smplx.create(cfg.EXP.smpl_path, 'smplh', gender='NEUTRAL', batch_size=1, **layer_args)
    }
    
    if torch.cuda.is_available():
        for gender in smpl_layers:
            smpl_layers[gender] = smpl_layers[gender].cuda()
    
    return smpl_layers

def get_smpl_layer(smpl_layers, gender, batch_size):
    current_layer = smpl_layers[gender]

    if current_layer.batch_size == batch_size:
        return current_layer

    layer_args = {
        'create_global_orient': False, 
        'create_body_pose': False, 
        'create_left_hand_pose': True, 
        'create_right_hand_pose': True, 
        'create_betas': False, 
        'create_transl': False, 
        'use_pca': False
    }
    
    new_layer = smplx.create(cfg.EXP.smpl_path, 'smplh', gender=gender, batch_size=batch_size, **layer_args)
    
    if torch.cuda.is_available():
        new_layer = new_layer.cuda()
    
    smpl_layers[gender] = new_layer
    
    return new_layer

def setup_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.EXP.gpu)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base_dir = os.path.join('results/output', f"{timestamp}")
    
    dirs = {
        'base': results_base_dir,
        'tensorboard': os.path.join(results_base_dir, "tensorboard"),
        'weights': os.path.join(results_base_dir, "checkpoints"),
        'logs': os.path.join(results_base_dir, "logs"),
        'viz': os.path.join(results_base_dir, "visualizations")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    print(f"Created: {results_base_dir}")
    return dirs

def parse_args_and_update_config():
    parser = argparse.ArgumentParser(description='Train FDTracker')
    parser.add_argument('--data-path', type=str, help='path to the dataset')
    parser.add_argument('--dataset-name', type=str, default='behave', choices=['behave', 'intercap'], help='dataset')
    parser.add_argument('--exp', type=str, default='', help='assign experiments directory')
    parser.add_argument('--checkpoint', type=str, help='model path for resuming')
    args = parser.parse_args()
    
    update_config(dataset_name=args.dataset_name.lower(), exp_dir=args.exp, ckpt_path=args.checkpoint)

    cfg.DATASET.data_path = args.data_path
    
    return args

def main():
    args = parse_args_and_update_config()
    dirs = setup_environment()
    
    best_weights = get_loss_weights()
      
    # Initialize model
    model = get_model()
    model = model.cuda() if torch.cuda.is_available() else model
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location='cuda')
    model_state_dict = None
    
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        loss_weights = checkpoint.get('loss_weights', None)
        epoch = checkpoint.get('epoch', 0)
    else:
        model_state_dict = checkpoint
        loss_weights = None
        epoch = 0
    
    model.load_weights(check_data_parallel(model_state_dict))
    print(f"Loaded model weights from {args.checkpoint}, epoch {epoch}")
    
    # Get validation dataloader
    val_loader, val_dataset = get_dataloader(is_train=False)
    if val_loader is None:
        print(f"Error: Could not create validation dataloader from {args.data_path}")
        return
    
    print(f"Validation dataset size: {len(val_dataset)}, batches: {len(val_loader)}")
    
    # Initialize SMPL layers
    smpl_layers = init_smpl_layers()

    if torch.cuda.is_available():
        for gender in smpl_layers:
            smpl_layers[gender] = smpl_layers[gender].cuda()

    validate_model_test(model, val_loader, val_dataset, best_weights, smpl_layers, dirs, get_smpl_layer)

if __name__ == "__main__":
    main()