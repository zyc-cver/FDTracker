import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_points_to_96(points: torch.Tensor, bbox_sizes: torch.Tensor) -> torch.Tensor:
    assert points.dim() == 4 and points.size(-1) == 2, "points (B,T,N,2)"
    assert bbox_sizes.dim() == 2, "bbox_sizes (B,T)"
    B, T, N, _ = points.shape

    # (B, T, 1, 1) -> broadcast  (B,T,N,2)
    scale = (96.0 / bbox_sizes).unsqueeze(-1).unsqueeze(-1)  
    scaled_points = points * scale
    return scaled_points


def make_K(K: Optional[torch.Tensor], B: int, T: int, device, dtype):
    if K is None:
        return None
    if K.ndim == 2 and K.shape == (3,3):
        K = K[None, None, ...].expand(B, T, 3, 3).to(device=device, dtype=dtype)
    elif K.ndim == 4 and K.shape[-2:] == (3,3):
        assert K.shape[0] == B and K.shape[1] == T, "K should be (B,T,3,3) if 4D"
        K = K.to(device=device, dtype=dtype)
    else:
        raise ValueError("K must be None, (3,3), or (B,T,3,3)")
    return K


def project_points_cam(points_bt_nc3: torch.Tensor, K_bt_33: torch.Tensor, eps=1e-6):
    """
    points: (B,T,N,3), K: (B,T,3,3)
    return: (B,T,N,2) (u,v)
    """
    B,T,N,_ = points_bt_nc3.shape
    X = points_bt_nc3[..., 0]
    Y = points_bt_nc3[..., 1]
    Z = points_bt_nc3[..., 2].clamp(min=eps)

    fx = K_bt_33[..., 0, 0].unsqueeze(-1)
    fy = K_bt_33[..., 1, 1].unsqueeze(-1)
    cx = K_bt_33[..., 0, 2].unsqueeze(-1)
    cy = K_bt_33[..., 1, 2].unsqueeze(-1)

    u = fx * (X / Z) + cx
    v = fy * (Y / Z) + cy
    return torch.stack([u, v], dim=-1)  # (B,T,N,2)


def sample_overlap_at_uv(overlap_b_t_1_h_w: torch.Tensor,
                         uv_bt_n2: torch.Tensor,
                         img_wh: Tuple[int, int]):
    B,T,_,H,W = overlap_b_t_1_h_w.shape
    _,_,N,_ = uv_bt_n2.shape
    W_img, H_img = img_wh

    x = uv_bt_n2[..., 0] / (W_img - 1) * 2 - 1
    y = uv_bt_n2[..., 1] / (H_img - 1) * 2 - 1
    grid = torch.stack((x, y), dim=-1)                          # (B,T,N,2)
    grid = grid.view(B*T, N, 1, 2)                              # (B*T,N,1,2)
    feat = F.grid_sample(
        overlap_b_t_1_h_w.view(B*T, 1, H, W), grid,
        align_corners=True, mode="bilinear", padding_mode="zeros"
    )                                                           # (B*T,1,N,1)
    feat = feat[:, :, :, 0].view(B, T, N)                       # (B,T,N)
    return feat.clamp(0, 1)


class OverlapEncoder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True),  # 48x48
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(inplace=True), # 24x24
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True), # 12x12
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True),# 6x6
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(128, out_dim)
        self.gate = nn.Sequential(nn.Linear(out_dim, 64),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(64, 1),
                                  nn.Sigmoid())

    def forward(self, overlap_bt_1_hw):  # (B,T,1,96,96)
        B,T,_,H,W = overlap_bt_1_hw.shape
        x = overlap_bt_1_hw.view(B*T, 1, H, W)
        feat = self.cnn(x).view(B*T, 128)
        out = self.proj(feat).view(B, T, -1)        # (B,T,128)
        gate = self.gate(out).view(B, T, 1)         # (B,T,1) in [0,1]
        return out, gate


class InteractionEncoder(nn.Module):
    def __init__(self, out_dim=128, k=8):
        super().__init__()
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(10, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim)
        )

    @staticmethod
    def _safe_norm(x, dim=-1, eps=1e-6):
        return torch.sqrt(torch.clamp((x**2).sum(dim=dim), min=eps))

    def forward(self,
        human_bt_n3, object_bt_m3,          # (B,T,Nh,3), (B,T,No,3)
        s_h_bt_n, s_o_bt_m,                 # (B,T,Nh), (B,T,No)
        tau: float = 0.05):
        B,T,Nh,_ = human_bt_n3.shape
        _,_,No,_ = object_bt_m3.shape
        BT = B*T

        h = human_bt_n3.view(BT, Nh, 3)
        o = object_bt_m3.view(BT, No, 3)
        s_h = s_h_bt_n.view(BT, Nh)   # [0,1]
        s_o = s_o_bt_m.view(BT, No)

        D = torch.cdist(h, o)                         # (BT, Nh, No)
        dmin_h, idx_o = torch.min(D, dim=-1)          # (BT, Nh)
        dmin_o, idx_h = torch.min(D, dim=-2)          # (BT, No)

        o_nn = torch.gather(o, 1, idx_o.unsqueeze(-1).expand(BT, Nh, 3))
        vec_h2o = o_nn - h                            # (BT, Nh, 3)
        dir_h2o = vec_h2o / self._safe_norm(vec_h2o, dim=-1, eps=1e-6).unsqueeze(-1)

        h_nn = torch.gather(h, 1, idx_h.unsqueeze(-1).expand(BT, No, 3))
        vec_o2h = h_nn - o
        dir_o2h = vec_o2h / self._safe_norm(vec_o2h, dim=-1, eps=1e-6).unsqueeze(-1)

        w_h = torch.exp(-dmin_h / tau) * s_h          # (BT, Nh)
        w_o = torch.exp(-dmin_o / tau) * s_o          # (BT, No)

        # Top-k（按 human->object）
        k = min(self.k, No)
        _, topk_idx = torch.topk(-D, k, dim=-1)  # (BT, Nh, k)

        topk_o = torch.gather(
            o.unsqueeze(1).expand(BT, Nh, No, 3),               # (BT, Nh, No, 3)
            2,
            topk_idx.unsqueeze(-1).expand(BT, Nh, k, 3)         # (BT, Nh, k, 3)
        )  # (BT, Nh, k, 3)

        topk_d = torch.gather(D, 2, topk_idx)  # (BT, Nh, k)

        topk_s_o = torch.gather(
            s_o.unsqueeze(1).expand(BT, Nh, No),  # (BT, Nh, No)
            2,
            topk_idx                              # (BT, Nh, k)
        )  # (BT, Nh, k)

        w_pair = torch.exp(-topk_d / tau) * (s_h.unsqueeze(-1)) * topk_s_o  # (BT,Nh,k)
        w_pair_sum = w_pair.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        mean_rel = ((topk_o - h.unsqueeze(2)) * w_pair.unsqueeze(-1)).sum(dim=2) / w_pair_sum  # (BT,Nh,3)
        mean_dist = (topk_d * w_pair).sum(dim=-1) / w_pair_sum.squeeze(-1)                     # (BT,Nh)

        def q(x, q):
            kq = int(max(1, round(q * x.shape[1])))
            vals, _ = torch.topk(x, kq, dim=1, largest=False)
            return vals.mean(dim=1, keepdim=True)

        feats_h = torch.cat([
            dmin_h.mean(dim=1, keepdim=True),
            dmin_h.min(dim=1, keepdim=True).values,
            q(dmin_h, 0.2), q(dmin_h, 0.5), q(dmin_h, 0.8),
            w_h.mean(dim=1, keepdim=True),
            dir_h2o.mean(dim=1),                    # (BT,3)
        ], dim=1)  # (BT, 9)

        feats_o = torch.cat([
            dmin_o.mean(dim=1, keepdim=True),
            w_o.mean(dim=1, keepdim=True),
        ], dim=1)  # (BT,2)

        # mean_rel / mean_dist
        extra = torch.cat([
            mean_rel.mean(dim=1),                   # (BT,3)
            mean_dist.mean(dim=1, keepdim=True),    # (BT,1)
        ], dim=1)  # (BT,4)

        feats = torch.cat([feats_h, feats_o, extra], dim=1)  # (BT, 15)
        feats = feats[:, :10]
        out = self.mlp(feats).view(B, T, -1)  # (B,T,out_dim=128)
        return out


class TemporalBlock(nn.Module):
    def __init__(self, dim, nhead=4, ff=1024, dropout=0.1, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=nhead, dim_feedforward=ff, dropout=dropout,
            activation="gelu"
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # (B,64,dim)
        return self.enc(x)

class ObjectAccelGate(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, object_xyz: torch.Tensor):
        B, T, N, _ = object_xyz.shape

        v = object_xyz[:, 1:] - object_xyz[:, :-1]          # (B,T-1,N,3)
        v = torch.cat([v[:, :1], v], dim=1)                 # pad → (B,T,N,3)

        a = v[:, 1:] - v[:, :-1]                            # (B,T-1,N,3)
        a = torch.cat([a[:, :1], a], dim=1)                 # pad → (B,T,N,3)

        amag = a.norm(dim=-1).mean(dim=-1, keepdim=True)    # (B,T,1)

        amax = amag.max(dim=1, keepdim=True).values         # (B,1,1)
        amean = amag.mean(dim=1, keepdim=True)              # (B,1,1)

        is_shaky = (amax - amean) / (amean + self.eps) >= 0.4  # (B,1,1)
        g_bad = torch.full_like(amag, 0.1)
        threshold = 0.4 * amax
        high_mask = (amag >= threshold).float()
        g_bad = torch.where(
            (is_shaky * high_mask).bool(), 
            torch.tensor(0.8, device=g_bad.device, dtype=g_bad.dtype), 
            g_bad
        )

        return g_bad, amag

class OIAFuser(nn.Module):
    def __init__(self, d_model=512, inter_dim=256, k=8, img_hw=(96,96)):
        super().__init__()
        self.img_hw = img_hw

        self.overlap_enc = OverlapEncoder(out_dim=128)
        self.interaction_enc = InteractionEncoder(out_dim=128, k=k)
        self.inter_proj = nn.Linear(128+128, inter_dim)  # concat(overlap, interaction)

        self.temporal_inter = TemporalBlock(inter_dim, nhead=4, ff=1024, dropout=0.1, num_layers=1)

        self.film_h = nn.Sequential(nn.Linear(inter_dim, d_model*2))
        self.film_o = nn.Sequential(nn.Linear(inter_dim, d_model*2))
        self.cross_h_from_o = nn.Linear(d_model, d_model)
        self.cross_o_from_h = nn.Linear(d_model, d_model)
        self.cross_gate = nn.Sequential(nn.Linear(inter_dim, d_model), nn.Sigmoid())
        self.temporal_h = TemporalBlock(d_model, nhead=8, ff=2048, dropout=0.1, num_layers=1)
        self.temporal_o = TemporalBlock(d_model, nhead=8, ff=2048, dropout=0.1, num_layers=1)

        self.norm_h = nn.LayerNorm(d_model)
        self.norm_o = nn.LayerNorm(d_model)

        self.rot_gate = ObjectAccelGate()

    def forward(self,
                human_feat, object_feat,           # (B,64,512)
                human_xyz, object_xyz,             # (B,64,96,3), (B,64,64,3)
                overlap2d,                         # (B,64,96,96) or (B,64,1,96,96)
                K,
                bbox
                ):

        B,T,_ = human_feat.shape
        assert T == 64, "64"

        if overlap2d.ndim == 4:   # (B,T,H,W) -> (B,T,1,H,W)
            overlap2d = overlap2d.unsqueeze(2)
        _,_,_,H,W = overlap2d.shape

        overlap_emb, _ = self.overlap_enc(overlap2d)     # (B,T,128), (B,T,1)

        K_bt = make_K(K, B, T, human_xyz.device, human_xyz.dtype)

        uv_h = project_points_cam(human_xyz, K_bt)              # (B,T,Nh,2)
        uv_o = project_points_cam(object_xyz, K_bt)             # (B,T,No,2)

        xy = bbox[..., :2]
        wh = bbox[..., 2:4]
        center = xy + wh / 2.0
        side = torch.max(wh[...,0], wh[...,1])
        new_top_left = center - side.unsqueeze(-1) / 2.0
        uv_h = uv_h - new_top_left.unsqueeze(2)
        uv_o = uv_o - new_top_left.unsqueeze(2)

        uv_h = resize_points_to_96(uv_h, side)                 # (B,T,Nh,2)
        uv_o = resize_points_to_96(uv_o, side)                 # (B

        s_h = sample_overlap_at_uv(overlap2d, uv_h, (W, H))     # (B,T,Nh)
        s_o = sample_overlap_at_uv(overlap2d, uv_o, (W, H))     # (B,T,No)

        inter_geo = self.interaction_enc(human_xyz, object_xyz, s_h, s_o)  # (B,T,128)

        inter = torch.cat([overlap_emb, inter_geo], dim=-1)         # (B,T,256)
        inter = self.inter_proj(inter)                               # (B,T,inter_dim)
        inter = self.temporal_inter(inter)                           # (B,T,inter_dim)

        gamma_beta_h = self.film_h(inter)                            # (B,T,2*512)
        gamma_beta_o = self.film_o(inter)
        gamma_h, beta_h = torch.chunk(gamma_beta_h, 2, dim=-1)
        gamma_o, beta_o = torch.chunk(gamma_beta_o, 2, dim=-1)

        # g = overlap_gate                                             # (B,T,1)
        # human_mod = human_feat + g * (gamma_h * human_feat + beta_h)
        # object_mod = object_feat + g * (gamma_o * object_feat + beta_o)

        object_xyz_centered = object_xyz - object_xyz.mean(dim=2, keepdim=True)

        g_bad, _ = self.rot_gate(object_xyz_centered.detach()) 
        human_mod = human_feat + (gamma_h * human_feat + beta_h)

        # object_corr = object_feat + (gamma_o * object_feat + beta_o)
        # object_mod  = (1.0 - g_bad) * object_feat + g_bad * object_corr

        object_mod  = (1.0 - g_bad) * object_feat + g_bad * (gamma_o * object_feat + beta_o)

        gate = self.cross_gate(inter)                                # (B,T,512) in (0,1)
        h_cross = human_mod + gate * self.cross_h_from_o(object_mod)
        o_cross = object_mod + gate * self.cross_o_from_h(human_mod)

        h_out = self.norm_h(self.temporal_h(h_cross))
        o_out = self.norm_o(self.temporal_o(o_cross))
        return h_out, o_out
