import torch
from tqdm import tqdm
import sys
sys.path.append('lib')
from loss.loss_total import compute_loss, get_loss_component_tracker

def train_epoch(model, train_loader, optimizer, weights, smpl_layers, epoch, writer, get_smpl_layer=None):
    model.train()
    train_loss = 0.0
    loss_components = get_loss_component_tracker()
    
    if len(train_loader) == 0:
        return 0.0, loss_components
    
    for batch_idx, (batch, gender) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} Training")):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.cuda() if torch.cuda.is_available() else v

        
        B, T = batch['hum_features'].shape[:2]
        current_smpl_layer = get_smpl_layer(smpl_layers, 'NEUTRAL', B*T)
        current_smpl_layers = {'NEUTRAL': current_smpl_layer.cuda()}
        preds = model(batch, current_smpl_layers['NEUTRAL'])
        total_loss, loss_dict = compute_loss(preds, batch, 'NEUTRAL', weights, current_smpl_layers)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        train_loss += total_loss.item()
        
        for k in loss_components.keys():
            if k in loss_dict:
                loss_components[k] += loss_dict[k]
        
        if batch_idx % 10 == 0:
            loss_str = ', '.join([f"{k}: {loss_dict[k]:.4f}" for k in loss_components.keys() if k in loss_dict])
            print(f"Epoch {epoch} Batch {batch_idx}: total loss, {total_loss:.4f} {loss_str}")
            
            for k in loss_components.keys():
                if k in loss_dict and k in weights and weights[k] > 0:
                    writer.add_scalar(f'Train/Batch_Loss_{k}', loss_dict[k], epoch * len(train_loader) + batch_idx)
    
    avg_loss = train_loss / len(train_loader)
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components