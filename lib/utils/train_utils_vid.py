import torch
import numpy as np
import os.path as osp
import copy
import torch.optim as optim
from collections import Counter
from collections import OrderedDict
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('lib.utils')
sys.path.append('data')
from core.config import cfg

def get_dataloader(is_train=True):
    data_path = cfg.DATASET.data_path
    dataset_name = cfg.DATASET.name
    split_dir = cfg.DATASET.train_dir if is_train else cfg.DATASET.val_dir
    split_path = os.path.join(data_path, split_dir)
    
    print(f"==> Preparing {'TRAIN' if is_train else 'TEST'} Dataloader...")
    print(f"Data Path: {split_path}")

    try:

        if is_train:
            if dataset_name == 'BEHAVE':
                from BEHAVE.dataset import BEHAVE
                dataset = BEHAVE(split_path)
            elif dataset_name == 'InterCap':
                from InterCap.dataset import InterCap
                dataset = InterCap(split_path)
        else:

            if dataset_name == 'BEHAVE':
                from BEHAVE.dataset import BEHAVETest
                dataset = BEHAVETest(split_path)
            elif dataset_name == 'InterCap':
                from InterCap.dataset import InterCapTest
                dataset = InterCapTest(split_path)
            
        dataset_len = len(dataset)
        print(f"# of {'TRAIN' if is_train else 'TEST'} data: {dataset_len}")
        
        if (dataset_len == 0):
            print(f"Empty Dataset: {split_path}")
            return None, dataset
        
        batch_per_dataset = cfg.TRAIN.batch_size if is_train else cfg.TEST.batch_size
        
        if dataset_len <= batch_per_dataset:
            adjusted_batch_size = max(1, dataset_len // 2)  

            batch_per_dataset = adjusted_batch_size
            drop_last = False 
        else:
            drop_last = is_train  
        
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=batch_per_dataset, 
            shuffle=cfg.TRAIN.shuffle if is_train else cfg.TEST.shuffle,
            num_workers=cfg.DATASET.workers, 
            pin_memory=True, 
            drop_last=drop_last,
            worker_init_fn=worker_init_fn
        )
        
        
        return dataloader, dataset
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return None, None


def train_setup(model, checkpoint):    
    # Use Adam optimizer as default - best for multiple loss components
    optimizer = get_optimizer(model=model, lr=cfg.TRAIN.lr, name='adam')
    
    # Use MultiStepLR scheduler as default - simple and effective
    lr_scheduler = get_scheduler(optimizer=optimizer)
    
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optim_state_dict'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        curr_lr = 0.0

        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']

        lr_state = checkpoint['scheduler_state_dict']
        lr_state['milestones'], lr_state['gamma'] = Counter(cfg.TRAIN.lr_step), cfg.TRAIN.lr_factor
        lr_scheduler.load_state_dict(lr_state)

        # History tracking removed as it's handled by tensorboard
        cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
        print("===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}"
              .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return optimizer, lr_scheduler


def check_data_parallel(train_weight):
    new_state_dict = OrderedDict()
    for k, v in train_weight.items():
        name = k[7:]  if k.startswith('module') else k  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_optimizer(model, lr=1.0e-4, name='adam'):
    total_params = []
    
    if hasattr(model, 'trainable_modules'):
        for module in model.trainable_modules:
            total_params += list(module.parameters())
    else:
        total_params += list(model.parameters())

    optimizer = None
    if name == 'sgd':
        optimizer = optim.SGD(
            total_params,
            lr=lr,
            momentum=cfg.TRAIN.momentum,
            weight_decay=cfg.TRAIN.weight_decay
        )
    elif name == 'rmsprop':
        optimizer = optim.RMSprop(
            total_params,
            lr=lr
        )
    elif name == 'adam':
        optimizer = optim.Adam(
            total_params,
            lr=lr,
            betas=(cfg.TRAIN.beta1, cfg.TRAIN.beta2)
        )
    elif name == 'adamw':
        optimizer = optim.AdamW(
            total_params,
            lr=lr,
            weight_decay=cfg.TRAIN.weight_decay
        )
    return optimizer


def get_scheduler(optimizer):
    scheduler = None
    if cfg.TRAIN.scheduler == 'step':
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.lr_step, gamma=cfg.TRAIN.lr_factor)
    elif cfg.TRAIN.scheduler == 'platue':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.TRAIN.lr_factor, patience=10, min_lr=1e-5)


    return scheduler


def save_checkpoint(states, epoch, file_path=None, is_best=False, save_path=None, best_metrics=None):
    if save_path is not None:
        output_path = save_path
    elif file_path is not None:
        output_path = file_path
    else:
        file_name = f'checkpoint_epoch_{epoch}.pth'
        if states['epoch'] == cfg.TRAIN.end_epoch:
            file_name = 'final.pth'
        output_path = os.path.join('.', file_name)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.save(states, output_path)
    print(f"Saving the latest checkpoint to {output_path}")
    # Save the best model
    if is_best:
        # Get the save directory
        save_dir = os.path.dirname(output_path)
        
        if best_metrics is None:
            best_file_path = os.path.join(save_dir, 'best.pth')
        else:
            # Create a descriptive filename with metrics
            metrics_str = '_'.join([f"{k}_{v:.4f}" for k, v in best_metrics.items()])
            best_file_path = os.path.join(save_dir, f'best_{metrics_str}.pth')
        
        torch.save(states, best_file_path)
        print(f"Saved the best model to {best_file_path}")


def load_checkpoint(load_dir, epoch=0, pick_best=False):
    try:
        checkpoint = torch.load(load_dir, map_location='cuda')
        return checkpoint
    except Exception as e:
        raise ValueError("No checkpoint exists!\n", e)


def worker_init_fn(worder_id):
    np.random.seed(np.random.get_state()[1][0] + worder_id)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)

def visualize_trans(preds, gt, video_list, epoch, viz_dir):
    os.makedirs(viz_dir, exist_ok=True)
    for i, video_name in enumerate(video_list):
        pred_hum_traj = preds[video_name]['trans']
        pred_obj_traj = preds[video_name]['obj_trans']
        gt_hum_traj = gt[video_name]['smplh_trans'].detach().cpu().numpy()
        gt_obj_traj = gt[video_name]['obj_trans'].detach().cpu().numpy()
        plt.figure(figsize=(10, 8))
        
        plt.plot(pred_hum_traj[:, 0], pred_hum_traj[:, 2], 'r-', linewidth=2, label='Pred Human', alpha=0.8)
        plt.plot(gt_hum_traj[:, 0], gt_hum_traj[:, 2], 'b-', linewidth=2, label='GT Human', alpha=0.8)
        plt.plot(pred_obj_traj[:, 0], pred_obj_traj[:, 2], 'orange', linewidth=2, label='Pred Object', alpha=0.8)
        plt.plot(gt_obj_traj[:, 0], gt_obj_traj[:, 2], 'g-', linewidth=2, label='GT Object', alpha=0.8)
        
        plt.scatter(gt_hum_traj[0, 0], gt_hum_traj[0, 2], c='blue', s=100, marker='^', 
                   label='Start (Human)', zorder=5)
        plt.scatter(gt_obj_traj[0, 0], gt_obj_traj[0, 2], c='green', s=100, marker='^', 
                   label='Start (Object)', zorder=5)
        
        plt.scatter(gt_hum_traj[-1, 0], gt_hum_traj[-1, 2], c='blue', s=100, marker='o', 
                   label='End (Human)', zorder=5)
        plt.scatter(gt_obj_traj[-1, 0], gt_obj_traj[-1, 2], c='green', s=100, marker='o', 
                   label='End (Object)', zorder=5)
        
        plt.xlabel('X (meters)')
        plt.ylabel('Z (meters)')
        plt.title(f'Trajectory Comparison - Epoch {epoch} - {video_name}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()

        save_path = os.path.join(viz_dir, f"{epoch}_{video_name}_trans.jpg")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
def visualize_render(batch, preds, gender, batch_idx, epoch, viz_dir):
    print(1)


def objective(trial, model, train_loader, val_loader, smpl_layers, logger=None):
    from core.loss_utils import get_loss_weights, compute_loss
    
    weights = get_loss_weights(trial)
    print(f"Trial {trial.number} with weights: {weights}")
    model.train()
    train_loss = 0.0
    for batch_idx, (batch, gender) in enumerate(train_loader):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda() if torch.cuda.is_available() else v
        preds = model(batch)
        total_loss, _ = compute_loss(preds, batch, gender, weights, smpl_layers)
        model.zero_grad()
        total_loss.backward()
        for param in model.parameters():
            if param.grad is not None:
                param.grad.detach_()
        train_loss += total_loss.item()
        if batch_idx >= 10:
            break
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_idx, (batch, gender) in enumerate(val_loader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda() if torch.cuda.is_available() else v
            preds = model(batch)
            total_loss, _ = compute_loss(preds, batch, gender, weights, smpl_layers)
            val_loss += total_loss.item()
            if batch_idx >= 10:
                break
    avg_val_loss = val_loss / min(len(val_loader), 11)
    return avg_val_loss