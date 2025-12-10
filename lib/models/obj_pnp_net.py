import torch
from torch import nn


class PnPNet(nn.Module):
    def __init__(
        self,
        hidden_dim=512,
        rot_dim=6,
    ):
        """
        Args:
            nIn: input feature channel
            spatial_pooltype: max | soft
            spatial_topk: 1
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rot_dim = rot_dim
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.fc_r = nn.Linear(hidden_dim // 4, rot_dim)  # quat or rot6d
        # TODO: predict centroid and z separately
        self.fc_t = nn.Linear(hidden_dim // 4, 3)
        self.act = nn.LeakyReLU(0.1, inplace=True)


    def forward(self, obj_feat, region=None, extents=None, mask_attention=None):
        B, T, C = obj_feat.shape
        x = obj_feat
        x = x.view(-1, self.hidden_dim)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        rot = self.fc_r(x)
        t = self.fc_t(x)
        rot = rot.view(B, T, self.rot_dim)
        t = t.view(B, T, 3)
        return rot, t