import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp
import numpy as np
from core.config import cfg
from models.RoPE import EncoderRoPEBlock
from models.oia_fuser import OIAFuser
from utils.data_utils import compute_transl_full_cam, compute_bbox_info
from utils.obj_utils import pose_from_predictions_train, ortho6d_to_mat_batch
from models.smooth import smooth_pose_6d, smooth_translation, smooth_betas
from models.obj_pnp_net import PnPNet
from loss.compute_verts import compute_all_vertices

class FDTracker(nn.Module):
    def __init__(self, 
                 human_feat_dim=1024,   # Updated to 1024-dimensional human features
                 obj_feat_dim=1024,     # Object feature dimension
                 hidden_dim=512,        # Hidden feature dimension
                 human_pose_dim=145,    # Human pose output dimension
                 obj_rot_dim=6,         # Object 6D rotation output dimension
                 obj_trans_dim=3,       # Object 3D translation output dimension
                 num_layers=6,          # Number of transformer layers
                 num_heads=8,           # Number of attention heads
                 mlp_ratio=4.0,         # MLP hidden layer ratio
                 dropout=0.1):         
        super(FDTracker, self).__init__()

        self.downsample_smpl_index = np.load('data/base_data/down_sample.npy')
        self.downsample_smpl_index = torch.from_numpy(self.downsample_smpl_index).long().to('cuda')
        self.downsample_smpl_index = self.downsample_smpl_index.unsqueeze(0).expand(8 * 64, -1)
        self.downsample_smpl_index = self.downsample_smpl_index.unsqueeze(-1).expand(-1, -1, 3)
        # Feature dimensions
        self.human_feat_dim = human_feat_dim  # Updated to 1024
        self.obj_feat_dim = obj_feat_dim
        self.hidden_dim = hidden_dim
        
        # Output dimensions
        self.human_pose_dim = human_pose_dim
        self.obj_rot_dim = obj_rot_dim
        self.obj_trans_dim = obj_trans_dim
        
        # Human camera parameters: 256*192, vitpose
        self.register_buffer("hum_cam_mean", torch.tensor([0.9747, 0.0009, 0.1736]))
        self.register_buffer("hum_cam_std", torch.tensor([0.1622, 0.0872, 0.0865]))

        # ===== Human network =====
        # Temporal feature embedding
        self.human_embed = nn.Sequential(
            nn.LayerNorm(human_feat_dim),
            nn.Linear(human_feat_dim, hidden_dim),
            nn.GELU()
        )
        
        # Human transformer encoder - ensure the use of the fixed EncoderRoPEBlock
        self.human_transformer = nn.ModuleList([
            EncoderRoPEBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Human pose regressor
        self.human_pose_regressor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, human_pose_dim)
        )
        
        # ===== Object network =====
        # Object feature embedding
        self.obj_feat_embed = nn.Sequential(
            nn.LayerNorm(obj_feat_dim),
            nn.Linear(obj_feat_dim, hidden_dim),
            nn.GELU()
        )

        # Object category embedding
        self.obj_id_embed = nn.Embedding(50, hidden_dim)  # Assume up to 50 object categories

        # Object PnP net
        self.obj_pnp_net = PnPNet(hidden_dim)
        
        # Added human cliff_cam embedding layer
        self.cliff_cam_hum_embed = nn.Sequential(
            nn.LayerNorm(3),  # Modified to 3 to match the actual shape of cliff_cam
            nn.Linear(3, hidden_dim),
            nn.GELU()
        )

        
        # Added vis_score embedding layer
        self.vis_score_embed = nn.Sequential(
            nn.LayerNorm(1),  # vis_score is a single value
            nn.Linear(1, hidden_dim),
            nn.GELU()
        )
        
        # Object transformer encoder - ensure the use of the fixed EncoderRoPEBlock
        self.obj_transformer = nn.ModuleList([
            EncoderRoPEBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.oiafuser = OIAFuser()

        # Object rotation predictor
        self.obj_rot_regressor = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=obj_rot_dim,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.0,
            use_conv=False
        )
        
        # Object translation predictor
        self.obj_trans_regressor = Mlp(
            in_features=hidden_dim,
            hidden_features=hidden_dim,
            out_features=obj_trans_dim,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.0,
            use_conv=False
        )

        # Object visibility predictor
        self.obj_vis_predictor = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        
        # Initialize weights
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
    
    def load_weights(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)
    
    def forward(self, inputs, smpl_layer):
        
        human_feats = inputs['hum_features']  # [B, T, 1024]
        obj_feats = inputs['obj_features']    # [B, T, 1024]

        B, T, _ = obj_feats.shape
        if self.downsample_smpl_index.shape[0] != B * T:
            self.downsample_smpl_index = np.load('data/base_data/down_sample.npy')
            self.downsample_smpl_index = torch.from_numpy(self.downsample_smpl_index).long().to('cuda')
            self.downsample_smpl_index = self.downsample_smpl_index.unsqueeze(0).expand(B * T, -1)
            self.downsample_smpl_index = self.downsample_smpl_index.unsqueeze(-1).expand(-1, -1, 3)


        cliff_cam_hum = compute_bbox_info(inputs["bbox_xys_hum"])

        
        if human_feats.dim() == 6:
            B, T, J, D1, D2, D3 = human_feats.shape
            human_feats = human_feats.reshape(B, T, J, D1 * D2 * D3)
            human_feats = human_feats.mean(dim=2)
        
        batch_size, seq_len = human_feats.shape[0], human_feats.shape[1]
        
        human_embed = self.human_embed(human_feats)  # [B, T, hidden_dim]
        cliff_cam_hum_embedded = self.cliff_cam_hum_embed(cliff_cam_hum)  # [B, T, hidden_dim]
        human_embed = human_embed + cliff_cam_hum_embedded  # [B, T, hidden_dim]
        human_features = human_embed
        for block in self.human_transformer:
            human_features = block(human_features)
        
        obj_feat_embedded = self.obj_feat_embed(obj_feats)  # [B, T, hidden_dim]

        for block in self.obj_transformer:
            obj_feat_embedded = block(obj_feat_embedded)

        init_hum_feat = human_features
        init_obj_feat = obj_feat_embedded

        init_human_pose = self.human_pose_regressor(init_hum_feat)  # [B, T, human_pose_dim]
        
       
        init_obj_rot_6d, init_obj_trans_bbox = self.obj_pnp_net(init_obj_feat)    # [B, T, 6]

        init_obj_rot_mat = ortho6d_to_mat_batch(init_obj_rot_6d)  # [B, T, 3, 3]
        
        init_obj_rot, init_obj_trans = pose_from_predictions_train(
            init_obj_rot_mat,
            pred_centroids=init_obj_trans_bbox[..., :2],
            pred_z_vals=init_obj_trans_bbox[..., 2:3],
            roi_cams=inputs["roi_cam"],
            roi_centers=inputs["roi_center"],
            resize_ratios=inputs["resize_ratios"],
            roi_whs=inputs["roi_whs"],
            eps=1e-4,
        )

        init_human_trans = init_human_pose[:, :, 142:145]
        init_human_trans = init_human_trans * self.hum_cam_std + self.hum_cam_mean
        init_human_trans = compute_transl_full_cam(init_human_trans, inputs['bbox_xys_hum'])
        
        init_human_verts, init_obj_verts, init_human_joints, _ = compute_all_vertices(
            pred_human_pose6d=init_human_pose[:,:,:132],
            pred_human_betas=init_human_pose[:,:,132:142],
            pred_human_trans=init_human_trans,
            pred_obj_rot=init_obj_rot,
            pred_obj_trans=init_obj_trans,
            obj_ids=inputs['obj_id'],
            smpl_layer=smpl_layer,
            dataset_name=cfg.DATASET.obj_set
        )

        init_human_verts_downsampled = torch.gather(init_human_verts, 1, self.downsample_smpl_index)
        init_human_verts_downsampled = init_human_verts_downsampled.view(B, T, 96, 3)
        init_obj_verts = init_obj_verts.view(B, T, 64, 3)
        oia_human_feat, oia_obj_feat = self.oiafuser(init_hum_feat, init_obj_feat, init_human_verts_downsampled, init_obj_verts, inputs['obj_overlap'], inputs['roi_cam'], inputs['bbox'])

        human_pose = self.human_pose_regressor(oia_human_feat)  # [B, T, human_pose_dim]
        
        obj_rot_6d, obj_trans_bbox = self.obj_pnp_net(oia_obj_feat)    # [B, T, 6]

        obj_rot_mat = ortho6d_to_mat_batch(obj_rot_6d)  # [B, T, 3, 3]
        obj_rot, obj_trans = pose_from_predictions_train(
            obj_rot_mat,
            pred_centroids=obj_trans_bbox[..., :2],
            pred_z_vals=obj_trans_bbox[..., 2:3],  # must be [B, 1]
            roi_cams=inputs["roi_cam"],
            roi_centers=inputs["roi_center"], 
            resize_ratios=inputs["resize_ratios"], # resize_ratio
            roi_whs=inputs["roi_whs"], # roi_wh
            eps=1e-4,
        )

        human_trans = human_pose[:, :, 142:145]
        human_trans = human_trans * self.hum_cam_std + self.hum_cam_mean 
        human_trans = compute_transl_full_cam(human_trans, inputs['bbox_xys_hum'])

        human_verts, obj_verts, human_joints, _ = compute_all_vertices(
            pred_human_pose6d=human_pose[:,:,:132],
            pred_human_betas=human_pose[:,:,132:142],
            pred_human_trans=human_trans,
            pred_obj_rot=obj_rot,
            pred_obj_trans=obj_trans,
            obj_ids=inputs['obj_id'],
            smpl_layer=smpl_layer,
            dataset_name=cfg.DATASET.obj_set
        )

        # During inference, optionally apply Gaussian smoothing (controlled by cfg parameter)
        if not self.training and hasattr(cfg.TEST, 'apply_smoothing') and cfg.TEST.apply_smoothing:
            # Smooth human translation
            human_trans = smooth_translation(human_trans, sigma=cfg.TEST.smoothing_sigma)
            
            # Smooth object translation
            obj_trans = smooth_translation(obj_trans, sigma=cfg.TEST.smoothing_sigma)
            
            # Smooth human pose (6D rotation) - smooth each joint separately
            human_pose_6d = human_pose[:,:,:132].reshape(batch_size, seq_len, -1, 6)  # [B, T, 22, 6]
            human_pose_6d_smooth = smooth_pose_6d(human_pose_6d, sigma=cfg.TEST.smoothing_sigma)
            human_pose_smooth = torch.cat([
                human_pose_6d_smooth.reshape(batch_size, seq_len, -1),  # Pose: [B, T, 132]
                smooth_betas(human_pose[:,:,132:142], sigma=cfg.TEST.smoothing_sigma),  # Betas: [B, T, 10]
            ], dim=-1)
            
            human_pose = human_pose_smooth
       
        return {
            'hum_pose6d': human_pose[:,:,:132],                 
            'hum_betas': human_pose[:,:,132:142],               
            'hum_trans_cam': human_trans,                       
            'obj_rot_mat': obj_rot,                             
            'obj_trans_cam': obj_trans,                         

            'human_verts': human_verts,                         
            'obj_verts': obj_verts,                             
            'human_joints': human_joints,                       

            'obj_bbox_centroid': obj_trans_bbox[..., :2],
            "obj_bbox_z": obj_trans_bbox[..., 2],

            'init_hum_pose6d': init_human_pose[:,:,:132],       
            'init_hum_betas': init_human_pose[:,:,132:142],     
            'init_hum_trans_cam': init_human_trans,             
            'init_obj_rot_mat': init_obj_rot,                   
            'init_obj_trans_cam': init_obj_trans,               

            'init_obj_bbox_centroid': init_obj_trans_bbox[..., :2],
            "init_obj_bbox_z": init_obj_trans_bbox[..., 2],

            'init_human_verts': init_human_verts.view(-1, 6890, 3),
            'init_obj_verts': init_obj_verts.view(-1, 64, 3),
            'downsample_smpl_index': self.downsample_smpl_index
        }

def get_model(model_type='fd_tracker'):
    return FDTracker()
