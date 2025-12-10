import torch
import numpy as np
import os.path as osp
import torch.optim as optim
from collections import Counter
from collections import OrderedDict
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pytorch3d.transforms import (
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
)

from core.config import cfg


def get_dataloader(data_path, is_train):
    dataset_name = data_path.split('/')[-1]
    split_dir = 'train_data' if is_train else 'val_data'
    split_path = os.path.join(data_path, split_dir)
    try:
        exec(f'from {dataset_name}.dataset import {dataset_name}')
    except ImportError as e:
        return None, None

    dataset_split = 'TRAIN' if is_train else 'TEST'
    batch_per_dataset = cfg[dataset_split].batch_size

    print(f"==> Preparing {dataset_split} Dataloader...")

    try:
        dataset = eval(f'{dataset_name}')(split_path)
        dataset_len = len(dataset)
        print(f"# of {dataset_split} {dataset_name} data: {dataset_len}")
        
        if dataset_len == 0:
            return None, dataset
        
        # Adjust batch size and drop_last for small datasets
        if dataset_len <= batch_per_dataset:
            adjusted_batch_size = max(1, dataset_len // 2)  # Ensure at least one batch
            batch_per_dataset = adjusted_batch_size
            drop_last = False  # For small datasets, do not drop the last batch
        else:
            drop_last = is_train  # Default: drop_last=True for training, False for validation
        
        # Unified DataLoader creation
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=batch_per_dataset, 
            shuffle=cfg[dataset_split].shuffle,
            num_workers=cfg.DATASET.workers, 
            pin_memory=True, 
            drop_last=drop_last,
            worker_init_fn=worker_init_fn
        )
        
        print(f"DataLoader created successfully, number of batches: {len(dataloader)}, "
              f"batch_size: {batch_per_dataset}, drop_last: {drop_last}")
        
        return dataloader, dataset
    
    except Exception as e:
        print(f"Failed to create dataset or DataLoader: {e}")
        import traceback
        print(traceback.format_exc())
        return None, None


def train_setup(model, checkpoint):    
    loss_history, eval_history = None, None
    optimizer = get_optimizer(model=model, lr=cfg.TRAIN.lr, name=cfg.TRAIN.optimizer)
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

        loss_history = checkpoint['train_log']
        eval_history = checkpoint['eval_history']
        cfg.TRAIN.begin_epoch = checkpoint['epoch'] + 1
        print("===> resume from epoch {:d}, current lr: {:.0e}, milestones: {}, lr factor: {:.0e}"
              .format(cfg.TRAIN.begin_epoch, curr_lr, lr_state['milestones'], lr_state['gamma']))

    return optimizer, lr_scheduler, loss_history, eval_history
    

class AverageMeterDict(object):
    def __init__(self, names):
        for name in names:
            value = AverageMeter()
            setattr(self, name, value)

    def __getitem__(self,key):
        return getattr(self, key)

    def update(self, name, val, n=1):
        getattr(self, name).update(val, n)


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count


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


def save_checkpoint(states, epoch, file_path=None, is_best=None, save_path=None):
    if file_path is None:
        file_name = f'epoch_{epoch}.pth.tar'
        output_dir = cfg.checkpoint_dir
        if states['epoch'] == cfg.TRAIN.end_epoch:
            file_name = 'final.pth.tar'
        file_path = osp.join(output_dir, file_name)
        
    if save_path is not None:
        file_path = save_path
            
    torch.save(states, file_path)

    if is_best:
        torch.save(states, osp.join(output_dir, 'best.pth.tar'))



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


def define_loss_weights(trial=None):
    if trial is not None:
        weights = {
            'hum_pose': trial.suggest_float('hum_pose', 0.5, 2.0),
            'hum_shape': trial.suggest_float('hum_shape', 0.5, 2.0),
            'hum_trans': trial.suggest_float('hum_trans', 5.0, 15.0),
            'obj_rot': trial.suggest_float('obj_rot', 0.5, 2.0),
            'obj_trans': trial.suggest_float('obj_trans', 0.5, 2.0),
            'hum_verts': trial.suggest_float('hum_verts', 0.5, 3.0)
        }
    else:
        weights = {
            'hum_pose': 1.0,
            'hum_shape': 1.0,
            'hum_trans': 10.0,
            'obj_rot': 1.0,
            'obj_trans': 1.0,
            'hum_verts': 1.0
        }
    return weights

def compute_loss(preds, batch, gender, weights, smpl_layers):
    B, T, N, C = batch['gt_pose6d'].shape
    loss_hum_pose = F.mse_loss(preds['hum_pose6d'].reshape(B, T, N, C), batch['gt_pose6d'])
    loss_hum_shape = F.mse_loss(preds['hum_betas'], batch['gt_betas']) 
    loss_hum_trans = F.mse_loss(preds['hum_trans_cam'], batch['gt_trans_cam'])
    loss_obj_rot = F.mse_loss(preds['obj_rot_mat'], batch['gt_obj_rot_mat'])
    loss_obj_trans = F.mse_loss(preds['obj_trans_cam'], batch['gt_obj_trans'])
    hum_pose6d = preds['hum_pose6d'].reshape(B, T, N, C)
    hum_pose = matrix_to_axis_angle(rotation_6d_to_matrix(hum_pose6d))  # (B, T, N, 3)
    hum_shape = preds['hum_betas']  # (B, shape_dim)
    hum_gt_pose6d = batch['gt_pose6d']
    hum_pose_gt = matrix_to_axis_angle(rotation_6d_to_matrix(hum_gt_pose6d))

    hum_verts = []
    hum_verts_gt = []
    for i in range(B):
        g = gender[i].upper() if isinstance(gender[i], str) else gender[i]
        smpl_layer = smpl_layers[g]
        # 时序批量输入
        # global_orient: (T, 1, 3) -> (T, 3)
        global_orient = hum_pose[i, :, 0:1, :].reshape(T, 3)
        body_pose = hum_pose[i, :, 1:, :].reshape(T, 21*3)
        global_orient_gt = hum_pose_gt[i, :, 0:1, :].reshape(T, 3)
        body_pose_gt = hum_pose_gt[i, :, 1:, :].reshape(T, 21*3)
        betas = hum_shape[i]
        betas_gt = batch['gt_betas'][i]
        hand_pose = torch.zeros((T, 45), dtype=torch.float32, device=betas.device)
        hum_verts.append(
            smpl_layer(
                betas=betas,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=hand_pose,
                right_hand_pose=hand_pose).vertices  # (T, V, 3)
        )
        hum_verts_gt.append(
            smpl_layer(
                betas=betas_gt,
                global_orient=global_orient_gt,
                body_pose=body_pose_gt,
                left_hand_pose=hand_pose,
                right_hand_pose=hand_pose).vertices
        )
    hum_verts = torch.stack(hum_verts, dim=0)      # (B, T, V, 3)
    hum_verts_gt = torch.stack(hum_verts_gt, dim=0)

    loss_humverts = F.mse_loss(hum_verts, hum_verts_gt)
    loss_L1_hum = weights['hum_pose'] * loss_hum_pose + weights['hum_shape'] * loss_hum_shape + weights['hum_trans'] * loss_hum_trans
    loss_L1_obj = weights['obj_rot'] * loss_obj_rot + weights['obj_trans'] * loss_obj_trans
    total_loss = loss_L1_hum + loss_L1_obj + weights['hum_verts'] * loss_humverts
    loss_dict = {
        'total': total_loss.item(),
        'hum_pose': loss_hum_pose.item(),
        'hum_shape': loss_hum_shape.item(),
        'hum_trans': loss_hum_trans.item(),
        'obj_rot': loss_obj_rot.item(),
        'obj_trans': loss_obj_trans.item(),
        'hum_verts': loss_humverts.item(),
    }
    return total_loss, loss_dict

def visualize_sample(batch, preds, gender, batch_idx, epoch, smpl_layers, viz_dir, writer):
    with torch.no_grad():
        B, T, _ = preds['hum_pose6d'].shape
        
        # For visualization purposes, select a specific batch index and time step
        batch_idx_viz = 0
        time_idx = 0
        
        # Get gender for the selected batch
        g = gender[batch_idx_viz].upper() if isinstance(gender[batch_idx_viz], str) else gender[batch_idx_viz]
        smpl_layer = smpl_layers[g]
        
        # Process predicted pose parameters
        hum_pose6d = preds['hum_pose6d'].reshape(B, T, -1, 6)  # (B, T, N, 6)
        hum_pose = matrix_to_axis_angle(rotation_6d_to_matrix(hum_pose6d))  # (B, T, N, 3)
        
        # Fold B and T dimensions for SMPL processing
        BT = B * T
        hum_pose_fold = hum_pose.reshape(BT, -1, 3)  # (B*T, N, 3)
        
        # Extract global orientation and body pose
        global_orient = hum_pose_fold[:, 0, :]  # (B*T, 1, 3)
        body_pose = hum_pose_fold[:, 1:, :].reshape(BT, -1)  # (B*T, (N-1)*3)
        
        # Repeat betas for each time step to match folded dimensions
        hum_shape_fold = preds['hum_betas'].reshape(BT, -1)  # (B*T, shape_dim)
        
        # Generate vertices using SMPL
        smpl_output = smpl_layer(
            betas=hum_shape_fold,
            global_orient=global_orient,
            body_pose=body_pose, batch_size=BT
        )
        
        # Extract vertices for the selected batch and time step
        batch_t_idx = batch_idx_viz * T + time_idx
        hum_verts = smpl_output.vertices[batch_t_idx].cpu().numpy()
        
        # Do the same for ground truth
        gt_pose6d = batch['gt_pose6d'].reshape(B, T, -1, 6)  # (B, T, N, 6)
        gt_pose = matrix_to_axis_angle(rotation_6d_to_matrix(gt_pose6d))  # (B, T, N, 3)
        
        # Fold B and T dimensions for ground truth
        gt_pose_fold = gt_pose.reshape(BT, -1, 3)  # (B*T, N, 3)
        gt_global_orient = gt_pose_fold[:, 0, :]  # (B*T, 1, 3)
        gt_body_pose = gt_pose_fold[:, 1:, :].reshape(BT, -1)  # (B*T, (N-1)*3)
        
        # Repeat betas for each time step
        gt_shape_fold = batch['gt_betas'].reshape(BT, -1)  # (B*T, shape_dim)
        
        # Generate ground truth vertices
        gt_output = smpl_layer(
            betas=gt_shape_fold,
            global_orient=gt_global_orient,
            body_pose=gt_body_pose, batch_size=BT
        )
        
        # Extract ground truth vertices for visualization
        hum_verts_gt = gt_output.vertices[batch_t_idx].cpu().numpy()
        
        # Visualization code
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].scatter(hum_verts[:, 0], hum_verts[:, 1], s=1, c='blue')
        ax[0].set_title('Prediction')
        ax[0].set_aspect('equal')
        ax[1].scatter(hum_verts_gt[:, 0], hum_verts_gt[:, 1], s=1, c='red')
        ax[1].set_title('Ground Truth')
        ax[1].set_aspect('equal')
        plt.suptitle(f'Epoch {epoch} - Sample {batch_idx}')
        viz_path = os.path.join(viz_dir, f'epoch_{epoch}_sample_{batch_idx}.png')
        plt.savefig(viz_path)
        plt.close()
        
        # Handle PIL ANTIALIAS deprecation
        try:
            # Read the saved image
            img = plt.imread(viz_path)
            # Convert to format expected by add_image
            img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)))
            writer.add_image(f'Validation Sample {batch_idx}', img_tensor, epoch)
        except Exception as e:
            print(f"Warning: Could not add image to tensorboard: {e}")
            # Alternative approach using torchvision if needed
            # from torchvision.io import read_image
            # try:
            #     img_tensor = read_image(viz_path)
            #     writer.add_image(f'Validation Sample {batch_idx}', img_tensor, epoch)
            # except Exception as e2:
            #     print(f"Failed to add image with alternative method: {e2}")

def objective(trial, model, train_loader, val_loader, smpl_layers, logger=None):
    weights = define_loss_weights(trial)
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