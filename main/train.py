import os
import sys
import glob
import argparse
import __init_path
import torch
import smplx
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
# Import core functionality

sys.path.append('lib')
from core.config import cfg, update_config
from models.model import get_model

from utils.train_utils_vid import get_dataloader, train_setup, save_checkpoint, check_data_parallel
from utils.training import train_epoch
from utils.validation import validate_model
from loss.loss_total import get_loss_weights

def parse_args_and_update_config():
    parser = argparse.ArgumentParser(description='Train FDTracker')
    parser.add_argument('--resume_training', action='store_true', help='resume training')
    parser.add_argument('--data-path', type=str, default='data/datasets/BEHAVE', help='path to the dataset')
    parser.add_argument('--dataset_name', type=str, default='behave', choices=['behave', 'intercap'], help='dataset')
    parser.add_argument('--exp', type=str, default='', help='assign experiments directory')
    parser.add_argument('--checkpoint', type=str, default='', help='model path for resuming')
    parser.add_argument('--auto_tune', action='store_true', default=False, help='enable automatic hyperparameter tuning with Optuna')
    parser.add_argument('--n_trials', type=int, default=5, help='number of trials for hyperparameter tuning')
    args = parser.parse_args()
    
    update_config(dataset_name=args.dataset_name.lower(), exp_dir=args.exp, ckpt_path=args.checkpoint)
    
    cfg.DATASET.data_path = args.data_path
    cfg.EXP.auto_tune = args.auto_tune
    cfg.EXP.n_trials = args.n_trials
    cfg.EXP.resume_training = args.resume_training
    
    return args

def setup_environment():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.EXP.gpu)
    print(f"GPU: {os.environ['CUDA_VISIBLE_DEVICES']}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_base_dir = os.path.join(cfg.EXP.results_dir, f"{cfg.EXP.exp_name}_{timestamp}")
    
    dirs = {
        'base': results_base_dir,
        'tensorboard': os.path.join(results_base_dir, "tensorboard"),
        'weights': os.path.join(results_base_dir, "checkpoints"),
        'logs': os.path.join(results_base_dir, "logs"),
        'viz': os.path.join(results_base_dir, "visualizations")
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    print(f"Experiment directory created: {results_base_dir}")
    return dirs

def init_model_and_data(checkpoint=None):
    model = get_model()
    model = model.cuda() if torch.cuda.is_available() else model
    if checkpoint is not None:
        model.load_weights(checkpoint['model_state_dict'])
        print(f"Loading checkpoint: {cfg.MODEL.weight_path}")
    
    optimizer, lr_scheduler = train_setup(model, checkpoint)

    train_loader, train_dataset = get_dataloader(is_train=True)
    val_loader, val_dataset = get_dataloader(is_train=False)
    
    if train_loader is None:
        return None, None, None, None, None, None, None
    
    if val_loader:
        print(f"Validation dataset: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    return model, optimizer, lr_scheduler, train_loader, train_dataset, val_loader, val_dataset

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

def cleanup_old_checkpoints(checkpoint_path, weights_dir):
    checkpoint_pattern = os.path.join(weights_dir, 'checkpoint_epoch_*.pth')
    checkpoint_files = glob.glob(checkpoint_pattern)
    metric_pattern = os.path.join(weights_dir, 'epoch_*_smpl_*_obj_*.pth')
    metric_files = glob.glob(metric_pattern)
    best_pattern = os.path.join(weights_dir, 'best_*.pth')
    best_files = glob.glob(best_pattern)
    is_current_best = 'best_' in os.path.basename(checkpoint_path)
    all_files = checkpoint_files + metric_files + best_files

    all_files.sort(key=os.path.getmtime, reverse=True)
    
    files_to_keep = [checkpoint_path] 
    
    if is_current_best:
        pass
    else:
        best_files.sort(key=os.path.getmtime, reverse=True)
        if best_files:
            files_to_keep.append(best_files[0])
    
    regular_checkpoints = [f for f in all_files if 'checkpoint_epoch_' in os.path.basename(f)]
    files_to_keep.extend(regular_checkpoints[:3])
    
    for old_ckpt in all_files:
        if old_ckpt not in files_to_keep:
            try:
                os.remove(old_ckpt)
            except Exception as e:
                print(f"error removing file {old_ckpt}: {e}")

def save_checkpoint(states, epoch, file_path=None, is_best=False, save_path=None, best_metrics=None):
    if save_path is not None:
        output_path = save_path
    elif file_path is not None:
        output_path = file_path
    else:
        file_name = f'checkpoint_epoch_{epoch}.pth'
        if states['epoch'] == cfg.TRAIN.end_epoch:
            file_name = 'final.pth'
        output_path = os.path.join('checkpoints', file_name)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    torch.save(states, output_path)
    print(f"Checkpoint saved to {output_path}")

    if is_best and best_metrics is not None:
        save_dir = os.path.dirname(output_path)
        
        if 'smpl_error' in best_metrics and 'obj_error' in best_metrics:
            smpl_int = int(best_metrics['smpl_error'])
            obj_int = int(best_metrics['obj_error'])
            human_acc_int = int(best_metrics.get('human_acc', 0) * 100)
            obj_acc_int = int(best_metrics.get('obj_acc', 0) * 100)
            best_file_name = f"best_epoch_{epoch}_smpl_{smpl_int}_obj_{obj_int}_hacc_{human_acc_int}_oacc_{obj_acc_int}.pth"
        else:
            metrics_str = '_'.join([f"{k}_{v:.4f}" for k, v in best_metrics.items()])
            best_file_name = f'best_{metrics_str}.pth'
        
        best_file_path = os.path.join(save_dir, best_file_name)
        torch.save(states, best_file_path)
        print(f"Best model saved to {best_file_path}")

def main():
    args = parse_args_and_update_config() 
    dirs = setup_environment() 
    
    writer = SummaryWriter(dirs['tensorboard'])
    
    from core.config import save_config_summary
    save_config_summary(dirs['base'])
    
    checkpoint = None
    if cfg.EXP.resume_training or cfg.EXP.checkpoint:
        from lib.utils.train_utils import load_checkpoint
        checkpoint = load_checkpoint(cfg.MODEL.weight_path)
    
    model, optimizer, lr_scheduler, train_loader, train_dataset, val_loader, val_dataset = init_model_and_data(checkpoint)
    if model is None:
        return
    
    smpl_layers = init_smpl_layers()
    
    best_weights = get_loss_weights()
    
    best_smpl_error = float('inf')
    best_obj_error = float('inf')
    best_epoch = 0
    best_metrics = {}
    
    print("===> Start Training...")
    for epoch in range(cfg.TRAIN.begin_epoch, cfg.TRAIN.end_epoch + 1):
        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, best_weights, smpl_layers, epoch, writer, get_smpl_layer
        )
        
        lr_scheduler.step()
        
        if epoch % 5 == 0 or epoch == cfg.TRAIN.end_epoch:
            avg_val_loss, avg_val_components, val_metrics = validate_model(
                model, val_loader, val_dataset, best_weights, smpl_layers, epoch, writer, dirs, get_smpl_layer
            )
            
            current_smpl_error = val_metrics.get('smpl_error', float('inf'))
            current_obj_error = val_metrics.get('obj_error', float('inf'))
            current_human_acc = val_metrics.get('human_acc', 0.0)
            current_obj_acc = val_metrics.get('obj_acc', 0.0)

            evaluation_score = current_smpl_error + current_obj_error

            is_best = False
            if hasattr(main, 'best_evaluation_score'):
                is_best = evaluation_score < main.best_evaluation_score
            else:
                main.best_evaluation_score = float('inf')
                is_best = True

            if is_best:
                main.best_evaluation_score = evaluation_score
                best_smpl_error = current_smpl_error
                best_obj_error = current_obj_error
                best_human_acc = current_human_acc
                best_obj_acc = current_obj_acc
                best_epoch = epoch
                best_metrics = {
                    'smpl_error': best_smpl_error,
                    'obj_error': best_obj_error,
                    'human_acc': best_human_acc,  
                    'obj_acc': best_obj_acc,      
                    **avg_val_components
                }
                
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': check_data_parallel(model.state_dict()),
                'optim_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'loss_weights': best_weights,
                'best_smpl_error': best_smpl_error,
                'best_obj_error': best_obj_error,
                'best_epoch': best_epoch
            }
            
            checkpoint_path = os.path.join(dirs['weights'], f'checkpoint_epoch_{epoch}.pth')
            
            if 'smpl_error' in val_metrics and 'obj_error' in val_metrics:
                formatted_name = f"epoch_{epoch}_smpl_{int(current_smpl_error)}_obj_{int(current_obj_error)}_hacc_{int(current_human_acc*100)}_oacc_{int(current_obj_acc*100)}"
                checkpoint_path = os.path.join(dirs['weights'], f'{formatted_name}.pth')
            
            save_checkpoint(
                checkpoint_data, 
                epoch, 
                save_path=checkpoint_path,
                is_best=is_best,
                best_metrics=best_metrics
            )
            print(f"Save checkpoint: {checkpoint_path}")
            
            cleanup_old_checkpoints(checkpoint_path, dirs['weights'])
        else:
            is_best = False
        
    
    writer.close()

if __name__ == "__main__":
    main()