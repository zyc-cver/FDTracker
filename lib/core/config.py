import os
import shutil
import datetime
import numpy as np
import os.path as osp
from easydict import EasyDict as edict
import trimesh


def init_dirs(dir_list):
    for dir in dir_list:
        if osp.exists(dir) and osp.isdir(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

cfg = edict()


""" Directory """
cfg.cur_dir = osp.dirname(osp.abspath(__file__))
cfg.root_dir = osp.join(cfg.cur_dir, '../../')
cfg.data_dir = osp.join(cfg.root_dir, 'data')
KST = datetime.timezone(datetime.timedelta(hours=9)) # CHANGE TIMEZONE FROM HERE


""" Dataset """
cfg.DATASET = edict()
cfg.DATASET.name = ''
cfg.DATASET.workers = 4
cfg.DATASET.random_seed = 123
cfg.DATASET.bbox_expand_ratio = 1.2
cfg.DATASET.obj_set = 'behave'

cfg.DATASET.data_path = f'data/datasets/{cfg.DATASET.name}'
cfg.DATASET.train_dir = 'train_data'
cfg.DATASET.val_dir = 'val_data'

""" Object Templates """
cfg.OBJ = edict()
cfg.OBJ.template_path = 'data/base_data/ref_hoi.pkl'
cfg.OBJ.template_sparse_path = f'data/base_data/object_models/{cfg.DATASET.obj_set}'
cfg.OBJ.template_key = 'templates'  # Key to access templates in the pickle file

""" Experiment """
cfg.EXP = edict()
cfg.EXP.gpu = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.EXP.gpu)
import torch
cfg.EXP.smpl_path = 'data/base_data/human_models'
cfg.EXP.results_dir = 'experiment'
cfg.EXP.exp_name = 'exp'
cfg.EXP.auto_tune = False
cfg.EXP.n_trials = 5
cfg.EXP.checkpoint = ''
cfg.EXP.save_predictions = True
""" Model - HMR """
cfg.MODEL = edict()
cfg.MODEL.input_img_shape = (512, 512)
cfg.MODEL.input_body_shape = (256, 256)
cfg.MODEL.input_hand_shape = (256, 256)
cfg.MODEL.img_feat_shape = (1, 1024)
cfg.MODEL.obj_feat_dim = 2048
cfg.MODEL.weight_path = ''


""" Train Detail """
cfg.TRAIN = edict()
cfg.TRAIN.batch_size = 16
cfg.TRAIN.shuffle = True
cfg.TRAIN.vis_trans = True
cfg.TRAIN.render = False
cfg.TRAIN.begin_epoch = 1
cfg.TRAIN.end_epoch = 200
cfg.TRAIN.warmup_epoch = 3
cfg.TRAIN.scheduler = 'step'
cfg.TRAIN.lr = 1e-4
cfg.TRAIN.min_lr = 1e-6
cfg.TRAIN.lr_step = [100]
cfg.TRAIN.lr_factor = 0.1
cfg.TRAIN.optimizer = 'adam'
cfg.TRAIN.momentum = 0
cfg.TRAIN.weight_decay = 0
cfg.TRAIN.beta1 = 0.5
cfg.TRAIN.beta2 = 0.999
cfg.TRAIN.print_freq = 10

# Human losses
cfg.TRAIN.h_pose_loss_weight = 3.0
cfg.TRAIN.h_shape_loss_weight = 0.001
cfg.TRAIN.h_verts_loss_weight = 1.0
cfg.TRAIN.h_trans_loss_weight = 5.0
cfg.TRAIN.h_proj_loss_weight = 100.0

# Object losses
cfg.TRAIN.o_rot_loss_weight = 5.0
cfg.TRAIN.o_trans_loss_weight = 2.0
cfg.TRAIN.o_centroid_loss_weight = 5.0
cfg.TRAIN.o_z_loss_weight = 1.0
cfg.TRAIN.o_points_loss_weight = 3.0
cfg.TRAIN.interact_contact_loss_weight = 2.0
cfg.TRAIN.directional_contact_loss_weight = 2.0
cfg.TRAIN.o_proj_loss_weight = 1.0
cfg.TRAIN.acc_loss_weight = 10.0


# Interaction losses
cfg.TRAIN.interaction_loss_weight = 10.0
cfg.TRAIN.distance_loss_weight = 10.0

# Other losses
# cfg.TRAIN.occlusion_loss_weight = 0.5

# Contact prediction losses (currently not used but kept for future)
cfg.TRAIN.interact_loss_weight = 1.0

""" Augmentation """
cfg.AUG = edict()
cfg.AUG.scale_factor = 0.2
cfg.AUG.rot_factor = 30
cfg.AUG.shift_factor = 0
cfg.AUG.color_factor = 0.2
cfg.AUG.blur_factor = 0
cfg.AUG.flip = False


""" Test Detail """
cfg.TEST = edict()
cfg.TEST.batch_size = 1
cfg.TEST.shuffle = False
cfg.TEST.do_eval = True
cfg.TEST.print_freq = 10
cfg.TEST.contact_thres = 0.05

cfg.TEST.apply_smoothing = True
cfg.TEST.smoothing_sigma = 3.0
cfg.TEST.motion_frames = 64
cfg.TEST.vis_freq = 10
cfg.TEST.save_all_results = False
cfg.TEST.run_eval = True


""" CAMERA """
cfg.CAMERA = edict()
cfg.CAMERA.original_img_size = (2048, 1536)  # (width, height)

np.random.seed(cfg.DATASET.random_seed)
torch.manual_seed(cfg.DATASET.random_seed)
torch.cuda.manual_seed(cfg.DATASET.random_seed)
torch.backends.cudnn.benchmark = True

    
def update_config(dataset_name='', exp_dir='', ckpt_path=''):
    if dataset_name != '':
        dataset_name_dict = {'behave': 'BEHAVE', 'intercap': 'InterCap'}
        cfg.DATASET.name = dataset_name_dict[dataset_name.lower()]
        cfg.DATASET.obj_set = dataset_name
        if dataset_name == 'behave':
            cfg.CAMERA.original_img_size = (2048, 1536)
        elif dataset_name == 'intercap':
            cfg.CAMERA.original_img_size = (1920, 1080)
        cfg.DATASET.data_path = f'data/datasets/{cfg.DATASET.name}'
        cfg.OBJ.template_path = f'data/base_data/ref_hoi.pkl'
        cfg.OBJ.template_sparse_path = f'data/base_data/object_models/{cfg.DATASET.obj_set}'


    if ckpt_path != '':
        cfg.MODEL.weight_path = ckpt_path
        cfg.EXP.checkpoint = ckpt_path
    
    if exp_dir != '':
        cfg.exp_name = exp_dir
        cfg.EXP.exp_name = exp_dir
    
    cfg.default_root_dir = osp.join(cfg.root_dir, 'experiment')
    
    print(f"Dataset: {cfg.DATASET.name}, Object set: {cfg.DATASET.obj_set}")
    if ckpt_path:
        print(f"Checkpoint path: {cfg.MODEL.weight_path}")

def save_config_summary(output_dir):
    """Save a concise summary of configuration parameters instead of copying all code."""
    config_summary = {
        'DATASET': {k: v for k, v in cfg.DATASET.items()},
        'MODEL': {k: v for k, v in cfg.MODEL.items()},
        'TRAIN': {k: v for k, v in cfg.TRAIN.items()},
        'TEST': {k: v for k, v in cfg.TEST.items()},
        'AUG': {k: v for k, v in cfg.AUG.items()},
        'CAMERA': {k: v for k, v in cfg.CAMERA.items()},
    }
    
    import json
    with open(osp.join(output_dir, 'config_summary.json'), 'w') as f:
        json.dump(config_summary, f, indent=2)
    
    # Also save key git info if available
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        git_diff = subprocess.check_output(['git', 'diff']).decode('ascii')
        git_info = {'hash': git_hash, 'diff': git_diff if len(git_diff) < 10000 else 'diff too large'}
        with open(osp.join(output_dir, 'git_info.json'), 'w') as f:
            json.dump(git_info, f)
    except:
        pass