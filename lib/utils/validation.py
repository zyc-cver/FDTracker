import torch
import os
from tqdm import tqdm
import sys
import pickle
import numpy as np
from datetime import datetime
from utils.evaluate_tracking import evaluate_track

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append('lib')

from core.config import cfg
from utils.eval_utils import convert_predictions_for_evaluation, convert_predictions_for_evaluation_with_init

from utils.train_utils_vid import visualize_trans, visualize_render
from utils.data_utils import inverse_trans_obj_id
from loss.loss_total import get_loss_weights, compute_loss, get_loss_component_tracker

def update_video_predictions(videos_pred, preds, batch):
    vid_name = batch['meta']['seq_name'][0]  
    frame_ids = batch['meta']['frame_ids']   
    is_last = batch['meta']['is_last_segment'][0]
    orig_len = batch['meta']['orig_len'][0] 
    
    if vid_name not in videos_pred:
        videos_pred[vid_name] = {}
        for k, v in preds.items():
            if isinstance(v, torch.Tensor) and len(v.shape) >= 3:  
                output_shape = (orig_len,) + v.shape[2:]
                videos_pred[vid_name][k] = torch.zeros(
                    output_shape,
                    device='cpu',
                    dtype=v.dtype
                )
        
        videos_pred[vid_name]['_valid_mask'] = torch.zeros(orig_len, dtype=torch.bool)

    for t_rel, t_abs in enumerate(frame_ids):
        if t_abs < 0:
            continue
            
        if is_last and videos_pred[vid_name]['_valid_mask'][t_abs]:
            continue
        
        for k, v in preds.items():
            if isinstance(v, torch.Tensor):
                if len(v.shape) >= 3:
                    if t_rel < v.shape[1] and t_abs < videos_pred[vid_name][k].shape[0]:
                        videos_pred[vid_name][k][t_abs] = v[0, t_rel].cpu()
                elif len(v.shape) == 2:
                    videos_pred[vid_name][k] = v[0].cpu()
        videos_pred[vid_name]['_valid_mask'][t_abs] = True
    
    return videos_pred

def validate_model(model, val_loader, val_dataset, weights, smpl_layers, epoch, writer, dirs, get_smpl_layer=None):

    model.eval()
    val_loss = 0.0
    val_loss_components = get_loss_component_tracker()
    

    if val_loader is None or len(val_loader) == 0:
        print(f"Empty Validation loader!")
        return float('inf'), val_loss_components, {}
    

    videos_pred = {}

    with torch.no_grad():
        for batch_idx, (batch, gender) in enumerate(tqdm(val_loader, desc=f"Examining...")):

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda() if torch.cuda.is_available() else v
            
            B, T = batch['hum_features'].shape[:2]
            current_smpl_layer = get_smpl_layer(smpl_layers, 'NEUTRAL', B*T)
            current_smpl_layers = {'NEUTRAL': current_smpl_layer}
            preds = model(batch, smpl_layers['NEUTRAL'])
            total_loss, loss_dict = compute_loss(preds, batch, 'NEUTRAL', weights, current_smpl_layers)
                
            val_loss += loss_dict['total']
            
            for k in val_loss_components.keys():
                if k in loss_dict:
                    val_loss_components[k] += loss_dict[k]

            update_video_predictions(videos_pred, preds, batch)
    
    for vid_name in videos_pred:
        if '_valid_mask' in videos_pred[vid_name]:
            del videos_pred[vid_name]['_valid_mask']
    
    avg_val_loss = val_loss / len(val_loader)
    avg_val_components = {k: v / len(val_loader) for k, v in val_loss_components.items()}
    
    writer.add_scalar('Val/Loss', avg_val_loss, epoch)
    for k, v in avg_val_components.items():
        if k in weights and weights[k] > 0:
            writer.add_scalar(f'Val/Loss_{k}', v, epoch)
    
    print(f"Epoch {epoch}: Val Loss {avg_val_loss:.4f}")
    

    val_metrics = {'smpl_error': float('inf'), 'obj_error': float('inf'), 'human_acc': 0.0, 'obj_acc': 0.0}
    
    metrics_file = os.path.join(dirs['logs'], 'metrics.txt')
    vids_file = os.path.join(dirs['logs'], 'videos.txt')

    if hasattr(val_dataset, 'has_gt') and val_dataset.has_gt and (epoch % 1 == 0 or epoch == cfg.TRAIN.end_epoch):
        try:
            pred_data = convert_predictions_for_evaluation(videos_pred)
            gt_data = torch.load(os.path.join(cfg.DATASET.data_path, cfg.DATASET.val_dir, 'gt.pt'))
            metadata = torch.load(os.path.join(cfg.DATASET.data_path, cfg.DATASET.val_dir, 'metadata.pt'))
            
            video_list = None

            val_metrics, videos_metrics = evaluate_track(pred_data, gt_data, metadata, video_list)

            if cfg.DATASET.name == 'InterCap':
                smpl_error_key = 'SMPL_icap'
                obj_error_key = 'obj_icap'
                human_acc_key = 'acc-h_icap'
                obj_acc_key = 'acc-o_icap'
            elif cfg.DATASET.name == 'BEHAVE':
                smpl_error_key = 'SMPL_behave'
                obj_error_key = 'obj_behave'
                human_acc_key = 'acc-h_behave'
                obj_acc_key = 'acc-o_behave'
            
            smpl_error = val_metrics.get(smpl_error_key, float('inf'))
            obj_error = val_metrics.get(obj_error_key, float('inf'))
            human_acc = val_metrics.get(human_acc_key, 0.0)
            obj_acc = val_metrics.get(obj_acc_key, 0.0)

            val_metrics = {
                'smpl_error': smpl_error,
                'obj_error': obj_error,
                'human_acc': human_acc,
                'obj_acc': obj_acc
            }
            
            print(f"Epoch {epoch}: Video-level metrics - Human SMPL Error: {smpl_error:.4f}, "
                 f"Object Error: {obj_error:.4f}, Human Acc: {human_acc:.4f}, Object Acc: {obj_acc:.4f}")

            formatted_name = f"epoch_{epoch}_smpl_{int(smpl_error)}_obj_{int(obj_error)}_hacc_{int(human_acc*100)}_oacc_{int(obj_acc*100)}"
            
            pred_dir = os.path.join(dirs['logs'], "predictions")
            os.makedirs(pred_dir, exist_ok=True)
            
            if cfg.EXP.save_predictions:
                pred_file = os.path.join(pred_dir, f"{formatted_name}.pkl")
                with open(pred_file, 'wb') as f:
                    pickle.dump(pred_data, f)
                print(f"Save prediction data to {pred_file}")
            if cfg.TRAIN.vis_trans and video_list != None: 
                visualize_trans(pred_data, gt_data, video_list, epoch, dirs['viz'])
            if cfg.TRAIN.render and video_list != None:
                visualize_render(pred_data, gt_data, video_list, epoch, dirs['viz'])

            for k, v in val_metrics.items():
                writer.add_scalar(f'Track/{k}', v, epoch)
                
        except Exception as e:
            print(f"Warning! Wrong - {e}")
            import traceback
            print(traceback.format_exc())
    
    print(f"Save metrics ot file: {metrics_file}, {vids_file}")

    file_exists = os.path.isfile(metrics_file)
    try:
        with open(metrics_file, 'a') as f:
            if not file_exists:
                f.write("Epoch,SMPL_Error,Obj_Error,Human_Acc,Obj_Acc,TimeStamp\n")
                f.flush()
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metrics_line = f"{epoch},{val_metrics['smpl_error']:.4f},{val_metrics['obj_error']:.4f},{val_metrics['human_acc']:.4f},{val_metrics['obj_acc']:.4f},{timestamp}\n"
            f.write(metrics_line)
            f.flush()
            
            print(f"Matrix: {metrics_line.strip()}")
        if video_list != None:
            with open(vids_file, 'a') as f:
                if not file_exists:
                    f.write("Epoch,Video_Name,SMPL_Error,Obj_Error,Human_Acc,Obj_Acc\n")
                    f.flush()
                for video_name in video_list:
                    if cfg.DATASET.name == 'BEHAVE':
                        video_line = f"{epoch},{video_name},{videos_metrics[video_name]['SMPL_behave']:.4f},{videos_metrics[video_name]['obj_behave']:.4f},{videos_metrics[video_name]['acc-h_behave']:.4f},{videos_metrics[video_name]['acc-o_behave']:.4f}\n"
                    elif cfg.DATASET.name == 'InterCap':
                        video_line = f"{epoch},{video_name},{videos_metrics[video_name]['SMPL_icap']:.4f},{videos_metrics[video_name]['obj_icap']:.4f},{videos_metrics[video_name]['acc-h_icap']:.4f},{videos_metrics[video_name]['acc-o_icap']:.4f}\n"
                    f.write(video_line)
                    f.flush()
    except Exception as e:
        print(f"Error - {e}")
    
    return avg_val_loss, avg_val_components, val_metrics

def validate_model_test(model, val_loader, val_dataset, weights, smpl_layers, dirs, get_smpl_layer=None):

    model.eval()
    val_loss = 0.0
    val_loss_components = get_loss_component_tracker()
    

    if val_loader is None or len(val_loader) == 0:
        print(f"Empty Validation loader!")
        return float('inf'), val_loss_components, {}
    videos_pred = {}


    with torch.no_grad():
        for batch_idx, (batch, gender) in enumerate(tqdm(val_loader, desc=f"Examining...")):

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.cuda() if torch.cuda.is_available() else v
            
            B, T = batch['hum_features'].shape[:2]
            current_smpl_layer = get_smpl_layer(smpl_layers, 'NEUTRAL', B*T)
            current_smpl_layers = {'NEUTRAL': current_smpl_layer}
            preds = model(batch, smpl_layers['NEUTRAL'])
            total_loss, loss_dict = compute_loss(preds, batch, 'NEUTRAL', weights, current_smpl_layers)
                
            val_loss += loss_dict['total']
            
            for k in val_loss_components.keys():
                if k in loss_dict:
                    val_loss_components[k] += loss_dict[k]

            update_video_predictions(videos_pred, preds, batch)
    
    for vid_name in videos_pred:
        if '_valid_mask' in videos_pred[vid_name]:
            del videos_pred[vid_name]['_valid_mask']
          
    pred_data = convert_predictions_for_evaluation_with_init(videos_pred)

    
    pred_dir = os.path.join(dirs['logs'], "predictions")
    os.makedirs(pred_dir, exist_ok=True)
    
    if cfg.EXP.save_predictions:
        pred_file = os.path.join(pred_dir, f"result.pkl")
        with open(pred_file, 'wb') as f:
            pickle.dump(pred_data, f)
        print(f"Save prediction data to {pred_file}")
                    