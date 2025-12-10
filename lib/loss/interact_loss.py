import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class InterContactLoss(nn.Module):
    """
    Interaction (contact) loss (class version; no chunk splitting):
      - soft-BCE: p = sigmoid((thr - d)/alpha), aligned with (real/fake) contact masks
      - Distance: penalizes Huber loss of (d - contact_target)_+ only at "should contact" locations
    Suitable inputs:
      human_verts:  [BT, Vh, 3]  Predicted human vertices (camera/unified coordinates)
      object_verts: [BT, Vo, 3]  Predicted object vertices
      gt_human_verts / gt_object_verts: Optional; if not provided, pseudo-labels are used
    """

    def __init__(self,
                 distance_threshold: float = 0.2,  # Threshold to determine contact (unit consistent with vertices)
                 contact_target: float = 0.1,      # Target contact distance (≤ this value is acceptable)
                 alpha: float = 0.01,             # Soft probability temperature
                 huber_delta: float = 0.01,       # Huber smoothing region
                 w_bce: float = 0.5,              # Weight for classification term
                 w_dist: float = 1.0,             # Weight for distance term
                 class_balance: bool = True,      # Whether to reweight BCE based on positive/negative ratio
                 return_details: bool = False,    # Whether to return a details dictionary
                 eps: float = 1e-6):
        super().__init__()
        self.distance_threshold = distance_threshold
        self.contact_target = contact_target
        self.alpha = alpha
        self.huber_delta = huber_delta
        self.w_bce = w_bce
        self.w_dist = w_dist
        self.class_balance = class_balance
        self.return_details = return_details
        self.eps = eps

    @staticmethod
    def _min_cdist(A: torch.Tensor, B: torch.Tensor):
        """A:[BT, NA, 3], B:[BT, NB, 3] -> dmin:[BT, NA]"""
        D = torch.cdist(A, B)                 # [BT, NA, NB]
        return D.min(dim=-1).values           # [BT, NA]

    def _masked_huber(self, over: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        sel = mask > 0.5
        if sel.sum() == 0:
            return torch.zeros([], device=over.device)
        x = over[sel]
        ax = x.abs()
        delta = self.huber_delta
        hub = torch.where(ax < delta, 0.5 * (ax ** 2) / delta, ax - 0.5 * delta)
        return hub.mean()

    def _bce_balanced(self, p: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if not self.class_balance:
            return F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), target)
        pos = target.sum()
        neg = target.numel() - pos
        pos_w = (neg + self.eps) / (pos + self.eps)
        w = torch.where(target > 0.5,
                        torch.full_like(target, pos_w),
                        torch.ones_like(target))
        return F.binary_cross_entropy(p.clamp(1e-6, 1-1e-6), target, weight=w)

    def forward(self,
                human_verts: torch.Tensor,      # [BT, Vh, 3]
                object_verts: torch.Tensor,     # [BT, Vo, 3]
                gt_human_verts: torch.Tensor = None,   # [BT, Vh, 3]
                gt_object_verts: torch.Tensor = None   # [BT, Vo, 3]
                ):
        device = human_verts.device
        assert human_verts.dim() == 3 and object_verts.dim() == 3, "verts should be [BT, V, 3]"
        assert human_verts.shape[0] == object_verts.shape[0], "BT mismatch"

        dmin_h = self._min_cdist(human_verts, object_verts)  # [BT, Vh]
        dmin_o = self._min_cdist(object_verts, human_verts)  # [BT, Vo]

        if gt_human_verts is not None and gt_object_verts is not None:
            dmin_h_gt = self._min_cdist(gt_human_verts, gt_object_verts)
            dmin_o_gt = self._min_cdist(gt_object_verts, gt_human_verts)
            gt_mask_h = (dmin_h_gt < self.distance_threshold).float()
            gt_mask_o = (dmin_o_gt < self.distance_threshold).float()
        else:
            gt_mask_h = (dmin_h.detach() < self.distance_threshold).float()
            gt_mask_o = (dmin_o.detach() < self.distance_threshold).float()

        thr = self.distance_threshold
        p_h = torch.sigmoid((thr - dmin_h) / max(self.alpha, 1e-8))
        p_o = torch.sigmoid((thr - dmin_o) / max(self.alpha, 1e-8))
        bce_h = self._bce_balanced(p_h, gt_mask_h)
        bce_o = self._bce_balanced(p_o, gt_mask_o)
        L_bce = 0.5 * (bce_h + bce_o)

        over_h = (dmin_h - self.contact_target).clamp_min(0.0)
        over_o = (dmin_o - self.contact_target).clamp_min(0.0)
        L_dist = self._masked_huber(over_h, gt_mask_h) + self._masked_huber(over_o, gt_mask_o)

        total = self.w_bce * L_bce + self.w_dist * L_dist

        if not self.return_details:
            return total

        details = {
            "L_total": total.detach(),
            "L_bce": L_bce.detach(),
            "L_dist": L_dist.detach(),
            "pos_rate_h": gt_mask_h.mean().detach(),
            "pos_rate_o": gt_mask_o.mean().detach(),
            "dmin_h_mean": dmin_h.mean().detach(),
            "dmin_o_mean": dmin_o.mean().detach(),
        }
        return total, details


class BilateralContactDirectionalLoss(nn.Module):
    """
    Compute the contact directional weighted L1 loss for both human-to-object and object-to-human in one pass.
    - Use GT vertices for nearest neighbor matching (teacher-forcing) to construct relative displacement vectors;
    - Weights are based on weight_basis = nearest distance (default using GT, can also use predictions);
    - Assign higher weights to the closest topk_ratio points within positive samples using ((t-d)/t)^gamma;
    - Regress direction/displacement only on positive samples (distance < thr), distant points do not affect gradients.
    """

    def __init__(self,
                 topk_ratio: float = 0.2,
                 gamma: float = 4.0,
                 distance_threshold: float = 0.2,
                 basis_mode: str = "gt",  # "gt" | "pred"
                 eps: float = 1e-8):
        super().__init__()
        assert 0.0 < topk_ratio < 1.0
        assert basis_mode in ("gt", "pred")
        self.topk_ratio = topk_ratio
        self.gamma = gamma
        self.distance_threshold = distance_threshold
        self.basis_mode = basis_mode
        self.eps = eps

    @staticmethod
    def _min_dist_and_index(A: torch.Tensor, B: torch.Tensor):
        # A: [BT, NA, 3], B: [BT, NB, 3]
        D = torch.cdist(A, B)              # [BT, NA, NB]
        dmin, idx = D.min(dim=-1)          # [BT, NA], [BT, NA]
        return dmin, idx

    def _weighted_l1_batch(self, pred_vec, gt_vec, basis, mask):
        """
        Perform top-k weighted L1 loss computation within each batch sample for [BT, V, 3] / [BT, V].
        Only positions where mask == True are included; all other weights are set to 0.
        Returns a scalar.
        """
        BT, V, _ = pred_vec.shape
        num = pred_vec.new_zeros(())
        den = pred_vec.new_zeros(())
        for b in range(BT):
            sel = mask[b]                   # [V]
            if sel.sum() == 0:
                continue
            d = basis[b, sel]               # [V_sel]
            diff = (pred_vec[b, sel] - gt_vec[b, sel]).abs().sum(dim=-1)  # [V_sel]

            k = max(1, int(round(self.topk_ratio * d.shape[0])))
            t = torch.kthvalue(d, k).values
            w = ((t - d) / (t + self.eps)).clamp(min=0.0).pow(self.gamma)  # [V_sel]

            num = num + (w * diff).sum()
            den = den + w.sum()

        return num / (den + self.eps)

    def forward(self,
                pred_h_verts: torch.Tensor,  # [BT, Vh, 3]
                pred_o_verts: torch.Tensor,  # [BT, Vo, 3]
                gt_h_verts:   torch.Tensor = None,  # [BT, Vh, 3]
                gt_o_verts:   torch.Tensor = None   # [BT, Vo, 3]
                ) -> torch.Tensor:

        device = pred_h_verts.device
        assert pred_h_verts.dim() == 3 and pred_o_verts.dim() == 3
        BT, Vh, _ = pred_h_verts.shape
        BT2, Vo, _ = pred_o_verts.shape
        assert BT == BT2, "BT mismatch between pred_h_verts and pred_o_verts"

        if self.basis_mode == "gt":
            dmin_h_gt, idx_o_gt = self._min_dist_and_index(gt_h_verts, gt_o_verts)  # [BT,Vh], [BT,Vh]
            dmin_o_gt, idx_h_gt = self._min_dist_and_index(gt_o_verts, gt_h_verts)  # [BT,Vo], [BT,Vo]

            o_nn_gt = torch.gather(gt_o_verts, 1, idx_o_gt.unsqueeze(-1).expand(-1, -1, 3))  # [BT,Vh,3]
            h_nn_gt = torch.gather(gt_h_verts, 1, idx_h_gt.unsqueeze(-1).expand(-1, -1, 3))  # [BT,Vo,3]
            rel_h2o_gt = o_nn_gt - gt_h_verts  # [BT,Vh,3]
            rel_o2h_gt = h_nn_gt - gt_o_verts  # [BT,Vo,3]

            o_nn_pred = torch.gather(pred_o_verts, 1, idx_o_gt.unsqueeze(-1).expand(-1, -1, 3))  # [BT,Vh,3]
            h_nn_pred = torch.gather(pred_h_verts, 1, idx_h_gt.unsqueeze(-1).expand(-1, -1, 3))  # [BT,Vo,3]
            rel_h2o_pred = o_nn_pred - pred_h_verts
            rel_o2h_pred = h_nn_pred - pred_o_verts

            basis_h = dmin_h_gt            # [BT,Vh]
            basis_o = dmin_o_gt            # [BT,Vo]

            mask_h = (dmin_h_gt < self.distance_threshold)
            mask_o = (dmin_o_gt < self.distance_threshold)

        else:
            dmin_h_pred, idx_o_pred = self._min_dist_and_index(pred_h_verts, pred_o_verts)  # [BT,Vh], [BT,Vh]
            dmin_o_pred, idx_h_pred = self._min_dist_and_index(pred_o_verts, pred_h_verts)  # [BT,Vo], [BT,Vo]

            if gt_h_verts is not None and gt_o_verts is not None:
                o_nn_gt = torch.gather(gt_o_verts, 1, idx_o_pred.unsqueeze(-1).expand(-1, -1, 3))
                h_nn_gt = torch.gather(gt_h_verts, 1, idx_h_pred.unsqueeze(-1).expand(-1, -1, 3))
                rel_h2o_gt = o_nn_gt - gt_h_verts
                rel_o2h_gt = h_nn_gt - gt_o_verts
            else:
                return torch.zeros((), device=device)

            o_nn_pred = torch.gather(pred_o_verts, 1, idx_o_pred.unsqueeze(-1).expand(-1, -1, 3))
            h_nn_pred = torch.gather(pred_h_verts, 1, idx_h_pred.unsqueeze(-1).expand(-1, -1, 3))
            rel_h2o_pred = o_nn_pred - pred_h_verts
            rel_o2h_pred = h_nn_pred - pred_o_verts
            basis_h = dmin_h_pred.detach()
            basis_o = dmin_o_pred.detach()

            mask_h = (dmin_h_pred < self.distance_threshold)
            mask_o = (dmin_o_pred < self.distance_threshold)

        L_h = self._weighted_l1_batch(rel_h2o_pred, rel_h2o_gt, basis_h, mask_h)
        L_o = self._weighted_l1_batch(rel_o2h_pred, rel_o2h_gt, basis_o, mask_o)
        return L_h + L_o
