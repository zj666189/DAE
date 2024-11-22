import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from einops import rearrange
from timm.models.layers import to_2tuple, trunc_normal_, DropPath


class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False

# 使用3x3卷积核
class ShiftConv2d0(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d0, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.n_div = 5
        g = inp_channels // self.n_div

        conv3x3 = nn.Conv2d(inp_channels, out_channels, 3, 1, 1)
        mask = nn.Parameter(torch.zeros((self.out_channels, self.inp_channels, 3, 3)), requires_grad=False)
        mask[:, 0 * g:1 * g, 1, 2] = 1.0
        mask[:, 1 * g:2 * g, 1, 0] = 1.0
        mask[:, 2 * g:3 * g, 2, 1] = 1.0
        mask[:, 3 * g:4 * g, 0, 1] = 1.0
        mask[:, 4 * g:, 1, 1] = 1.0
        self.w = conv3x3.weight
        self.b = conv3x3.bias
        self.m = mask

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.w * self.m, bias=self.b, stride=1, padding=1)
        return y

# 使用1x1卷积核
class ShiftConv2d1(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(ShiftConv2d1, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.zeros(inp_channels, 1, 3, 3), requires_grad=False)
        self.n_div = 5
        g = inp_channels // self.n_div
        self.weight[0 * g:1 * g, 0, 1, 2] = 1.0  ## left
        self.weight[1 * g:2 * g, 0, 1, 0] = 1.0  ## right
        self.weight[2 * g:3 * g, 0, 2, 1] = 1.0  ## up
        self.weight[3 * g:4 * g, 0, 0, 1] = 1.0  ## down
        self.weight[4 * g:, 0, 1, 1] = 1.0  ## identity

        self.conv1x1 = nn.Conv2d(inp_channels, out_channels, 1)

    def forward(self, x):
        y = F.conv2d(input=x, weight=self.weight, bias=None, stride=1, padding=1, groups=self.inp_channels)
        y = self.conv1x1(y)
        return y

# 根据指定的类型选择ShiftConv2d0、ShiftConv2d1或普通卷积操作
class ShiftConv2d(nn.Module):
    def __init__(self, inp_channels, out_channels, conv_type='fast-training-speed'):
        super(ShiftConv2d, self).__init__()
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.conv_type = conv_type
        if conv_type == 'low-training-memory':
            self.shift_conv = ShiftConv2d0(inp_channels, out_channels)
        elif conv_type == 'fast-training-speed':
            self.shift_conv = ShiftConv2d1(inp_channels, out_channels)
        elif conv_type == 'common':
            self.shift_conv = nn.Conv2d(inp_channels, out_channels, kernel_size=1)
        else:
            raise ValueError('invalid type of shift-conv2d')

    def forward(self, x):
        y = self.shift_conv(x)
        return y

# 轻量级的前馈神经网络模块
class LFE(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=4, act_type='gelu'):
        super(LFE, self).__init__()
        self.exp_ratio = exp_ratio
        self.act_type = act_type

        self.conv0 = ShiftConv2d(inp_channels, out_channels * exp_ratio)
        self.dwconv = nn.Conv2d(out_channels * exp_ratio, out_channels * exp_ratio, 3, 1, 1, bias=True,
                                groups=out_channels * exp_ratio)
        self.conv1 = ShiftConv2d(out_channels * exp_ratio, out_channels)

        if self.act_type == 'linear':
            self.act = None
        elif self.act_type == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif self.act_type == 'gelu':
            self.act = nn.GELU()
        else:
            raise ValueError('unsupport type of activation')

    def forward(self, x):
        y = self.conv0(x)
        y = self.act(y)
        y = self.act(y + self.dwconv(y))
        y = self.conv1(y)
        return y
    

class GFE(nn.Module):

    def __init__(self, channels, attn_drop, proj_drop, offset_range_factor, stride, window_size=5, shifts=0, split_size=1, attention_type='dot attention'):
        super(GFE, self).__init__()
        self.nc = channels
        self.scale = channels ** -0.5
        self.stride = stride
        self.shifts = shifts
        self.split_size = split_size
        self.window_size = window_size
        self.attention_type = attention_type
        self.split_chns = [channels * 2 // 3 for _ in range(3)]
        self.project_inp = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=1),
                                         nn.BatchNorm2d(channels * 2))
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)
        self.attn_drop = nn.Dropout(0.0, inplace=True)
        self.offset_range_factor = offset_range_factor
        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.nc, self.nc, 3, 1, 1, groups=self.nc),
            LayerNormProxy(self.nc),
            nn.GELU(),
            nn.Conv2d(self.nc, 2, 1, 1, 0, bias=False)
        )
        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )
        self.rpe_table = nn.Parameter(
            torch.zeros(1, 96 * 2 - 1, 96 * 2 - 1)
        )
        trunc_normal_(self.rpe_table, std=0.01)
        self.proj_drop = nn.Dropout(proj_drop, inplace=True)
        self.attn_drop = nn.Dropout(attn_drop, inplace=True)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1) # B * g H W 2

        return ref
    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):

        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B, -1, -1, -1) # B * g H W 2

        return ref
    

    def forward(self, x):
        B, C, h, w = x.shape
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        q_off = einops.rearrange(q, 'b (g c) h w -> (b g) c h w', g=1, c=self.nc)
        offset = self.conv_offset(q_off).contiguous()  # B * g 2 Hg Wg
        Hk, Wk = offset.size(2), offset.size(3)
        n_sample = Hk * Wk

        offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=device).reshape(1, 2, 1, 1)
        offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)
        pos = offset + reference
        x_sampled = F.grid_sample(
                input=x.reshape(B, self.nc, h, w), 
                grid=pos[..., (1, 0)], # y, x -> x, y
                mode='bilinear', align_corners=True) # B * g, Cg, Hg, Wg
        x_sampled = x_sampled.reshape(B, C, 1, n_sample)

        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        wsize = self.window_size
        fuse_out = []

        q = q.reshape(B, self.nc, h * w)
        k = self.proj_k(x_sampled).reshape(B, self.nc, n_sample)
        v = self.proj_v(x_sampled).reshape(B, self.nc, n_sample)
        
        attn = torch.einsum('b c m, b c n -> b m n', q, k) # B * h, HW, Ns
        attn = attn.mul(self.scale)
        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        q_grid = self._get_q_grid(h, w, B, dtype, device)
        displacement = (q_grid.reshape(B, h * w, 2).unsqueeze(2) - pos.reshape(B, n_sample, 2).unsqueeze(1)).mul(0.5)
        attn_bias = F.grid_sample(
            input=einops.rearrange(rpe_bias, 'b (g c) h w -> (b g) c h w', c=1, g=1),
            grid=displacement[..., (1, 0)],
            mode='bilinear', align_corners=True) # B * g, h_g, HW, Ns

        attn_bias = attn_bias.reshape(B, h * w, n_sample)
        attn = attn + attn_bias
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        
        out_dat = torch.einsum('b m n, b c n -> b c m', attn, v)
        out_dat = out_dat.reshape(B, C, h, w)
        # return out_dat

        # shifted window attention
        if self.shifts > 0:
            shifted_x = torch.roll(xs[0], shifts=(-self.shifts, -self.shifts), dims=(2, 3))
        else:
            shifted_x = xs[0]
        q, v = rearrange(
            shifted_x, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
            qv=2, dh=wsize, dw=wsize
        ) 
        atn = (q @ q.transpose(-2, -1)) if self.attention_type == 'dot attention' else (F.normalize(q, dim=-1) @ F.normalize(q, dim=-1).transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        out_win = (atn @ v)
        
        out_win = rearrange(
            out_win, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
            h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
        )
        if self.shifts > 0:
            out_win = torch.roll(out_win, shifts=(self.shifts, self.shifts), dims=(2, 3))
        
        fuse_out.append(out_win)

        # axis attentin
        h_col, w_col = h, self.split_size
        w_row, h_row = w, self.split_size
        # col-axis attention
        q, v = rearrange(
            xs[1], 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
            qv=2, dh=h_col, dw=w_col
        )
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        out_col = (atn @ v)
        out_col = rearrange(
            out_col, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
            h=h // h_col, w=w // w_col, dh=h_col, dw=w_col
        )
        fuse_out.append(out_col)
        # row-axis attention
        q, v = rearrange(
            xs[2], 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
            qv=2, dh=h_row, dw=w_row
        )
        atn = (q @ q.transpose(-2, -1))
        atn = atn.softmax(dim=-1)
        out_row = (atn @ v)
        out_row = rearrange(
            out_row, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
            h=h // h_row, w=w // w_row, dh=h_row, dw=w_row
        )
        fuse_out.append(out_row)

        out = torch.cat(fuse_out, dim=1)
        out = self.project_out(out)
        res = out * 0.5 + out_dat * 0.5

        return res
        
        

class LayerNormProxy(nn.Module):
    
    def __init__(self, dim):
        
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):

        x = einops.rearrange(x, 'b c h w -> b h w c')
        x = self.norm(x)
        return einops.rearrange(x, 'b h w c -> b c h w')
    

    
class MPAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, window_size=5, act_type='gelu', shifts=0, split_size=1, attention_type='dot attention'):
        super(MPAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shifts = shifts
        
        self.LFE = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, act_type=act_type)
        self.DAT = GFE(channels=inp_channels, attn_drop=0.0, proj_drop=0.0, offset_range_factor = 4,stride = 1, window_size=window_size, shifts=shifts, split_size=split_size, attention_type='dot attention')

    def forward(self, x):
        x = self.DAT(x) + x
        x = self.LFE(x) + x
        return x
    