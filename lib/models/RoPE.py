import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from einops import rearrange, repeat, einsum
from timm.models.vision_transformer import Mlp
from torch.cuda.amp import autocast

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


@autocast(enabled=False)
def apply_rotary_emb(freqs, t, start_index=0, scale=1.0, seq_dim=-2):
    if t.ndim == 3:
        seq_len = t.shape[seq_dim]
        freqs = freqs[-seq_len:].to(t)

    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim

    assert (
        rot_dim <= t.shape[-1]
    ), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"

    t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
    t = (t * freqs.cos() * scale) + (rotate_half(t) * freqs.sin() * scale)
    return torch.cat((t_left, t, t_right), dim=-1)


def get_encoding(d_model, max_seq_len=4096):
    """Return: (L, D)"""
    t = torch.arange(max_seq_len).float()
    freqs = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
    freqs = torch.einsum("i, j -> i j", t, freqs)
    freqs = repeat(freqs, "i j -> i (j r)", r=2)
    return freqs

class EncoderRoPEBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, dropout=0.1, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        self.attn = RoPEAttention(hidden_size, num_heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=True, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=dropout)

        self.gate_msa = nn.Parameter(torch.zeros(1, 1, hidden_size))
        self.gate_mlp = nn.Parameter(torch.zeros(1, 1, hidden_size))

        # Zero-out adaLN modulation layers
        nn.init.constant_(self.gate_msa, 0)
        nn.init.constant_(self.gate_mlp, 0)

    def forward(self, x, attn_mask=None, tgt_key_padding_mask=None):
        x = x + self.gate_msa * self._sa_block(
            self.norm1(x), attn_mask=attn_mask, key_padding_mask=tgt_key_padding_mask
        )
        x = x + self.gate_mlp * self.mlp(self.norm2(x))
        return x

    def _sa_block(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        x = self.attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        return x


class ROPE(nn.Module):
    """Minimal impl of a lang-style positional encoding."""

    def __init__(self, d_model, max_seq_len=4096):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Pre-cache a freqs tensor
        encoding = get_encoding(d_model, max_seq_len)
        self.register_buffer("encoding", encoding, False)

    def rotate_queries_or_keys(self, x):
        """
        Args:
            x : (B, H, L, D)
        Returns:
            rotated_x: (B, H, L, D)
        """

        seq_len, d_model = x.shape[-2:]
        assert d_model == self.d_model

        # encoding: (L, D)s
        if seq_len > self.max_seq_len:
            encoding = get_encoding(d_model, seq_len).to(x)
        else:
            encoding = self.encoding[:seq_len]

        # encoding: (L, D)
        # x: (B, H, L, D)
        rotated_x = apply_rotary_emb(encoding, x, seq_dim=-2)

        return rotated_x


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.rope = ROPE(self.head_dim, max_seq_len=4096)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x: (B, L, C)
        # attn_mask: (L, L)
        # key_padding_mask: (B, L)
        B, L, _ = x.shape
        xq, xk, xv = self.query(x), self.key(x), self.value(x)

        xq = xq.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xk = xk.reshape(B, L, self.num_heads, -1).transpose(1, 2)
        xv = xv.reshape(B, L, self.num_heads, -1).transpose(1, 2)

        xq = self.rope.rotate_queries_or_keys(xq)  # B, N, L, C
        xk = self.rope.rotate_queries_or_keys(xk)  # B, N, L, C

        attn_score = einsum(xq, xk, "b n i c, b n j c -> b n i j") / math.sqrt(self.head_dim)
        if attn_mask is not None:
            attn_mask = attn_mask.reshape(1, 1, L, L).expand(B, self.num_heads, -1, -1)
            attn_score = attn_score.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.reshape(B, 1, 1, L).expand(-1, self.num_heads, L, -1)
            attn_score = attn_score.masked_fill(key_padding_mask, float("-inf"))

        attn_score = torch.softmax(attn_score, dim=-1)
        attn_score = self.dropout(attn_score)
        output = einsum(attn_score, xv, "b n i j, b n j c -> b n i c")  # B, N, L, C
        output = output.transpose(1, 2).reshape(B, L, -1)  # B, L, C
        output = self.proj(output)  # B, L, C
        return output

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, context):
        B, L, _ = x.shape
        _, S, _ = context.shape
        
        q = self.query(x).reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # [B, N, L, C//N]
        k = self.key(context).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, N, S, C//N]
        v = self.value(context).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)  # [B, N, S, C//N]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, N, L, S]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, N, L, C//N]
        out = out.transpose(1, 2).reshape(B, L, self.embed_dim)  # [B, L, C]
        out = self.proj(out)
        
        return out


class HumanObjectInteraction(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.human_norm = nn.LayerNorm(hidden_dim)
        self.obj_norm = nn.LayerNorm(hidden_dim)

        self.human_to_obj = CrossAttention(hidden_dim, num_heads, dropout)
        
        self.obj_to_human = CrossAttention(hidden_dim, num_heads, dropout)

        self.human_gate = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.obj_gate = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        
    def forward(self, human_features, obj_features):
        """
        human_features: [B, T, C]
        obj_features: [B, T, C]
        """
        norm_human = self.human_norm(human_features)
        norm_obj = self.obj_norm(obj_features)
        
        human_to_obj_features = self.human_to_obj(norm_obj, norm_human)
        obj_to_human_features = self.obj_to_human(norm_human, norm_obj)
        
        updated_human = human_features + self.human_gate * obj_to_human_features
        updated_obj = obj_features + self.obj_gate * human_to_obj_features
        
        return updated_human, updated_obj


class TemporalTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        

        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU()
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            EncoderRoPEBlock(hidden_dim, num_heads, mlp_ratio=mlp_ratio, dropout=dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):

        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x, tgt_key_padding_mask=mask)
            
        return x

class OcclusionAwareTemporalModule(nn.Module):

    def __init__(self, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.rope_attn = RoPEAttention(hidden_dim, num_heads, dropout)
        
        self.occlusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for occlusion value
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, features, occlusion):


        temp_features = self.norm(features)
        temp_features = self.rope_attn(temp_features)

        occlusion_features = torch.cat([temp_features, occlusion], dim=-1)
        occlusion_aware_features = self.occlusion_mlp(occlusion_features)

        output = features + occlusion_aware_features
        
        return output
