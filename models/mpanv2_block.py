import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.nn import Softmax


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

class HiLo(nn.Module):

    def __init__(self, dim, num_heads=6, qkv_bias=False, attn_drop=0., proj_drop=0., window_size=2, alpha=0.5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        head_dim = int(dim/num_heads)
        self.dim = dim

        # self-attention heads in Lo-Fi
        self.l_heads = int(num_heads * alpha)
        # token dimension in Lo-Fi
        self.l_dim = self.l_heads * head_dim

        # self-attention heads in Hi-Fi
        self.h_heads = num_heads - self.l_heads
        # token dimension in Hi-Fi
        self.h_dim = self.h_heads * head_dim

        # local window size. The `s` in our paper.
        self.ws = window_size

        self.scale = head_dim ** -0.5

         # 添加分块大小参数
        self.chunk_size = 8192*2  # 可以根据显存大小调整

        # Low frequence attention (Lo-Fi)
        if self.l_heads > 0:
            if self.ws != 1:
                self.sr = nn.AvgPool2d(kernel_size=window_size, stride=window_size)
            self.l_q = nn.Linear(self.dim, self.l_dim, bias=qkv_bias)
            self.l_kv = nn.Linear(self.dim, self.l_dim * 2, bias=qkv_bias)
            self.l_proj = nn.Linear(self.l_dim, self.l_dim)

        # High frequence attention (Hi-Fi)
        if self.h_heads > 0:
            self.h_qkv = nn.Linear(self.dim, self.h_dim * 3, bias=qkv_bias)
            self.h_proj = nn.Linear(self.h_dim, self.h_dim)
        
        # self.project_inp = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=1),
        #                                  nn.BatchNorm2d(channels * 2))
        # self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def hifi(self, x):
        B, H, W, C = x.shape
        # x = self.project_inp(x)

        h_group, w_group = H // self.ws, W // self.ws

        total_groups = h_group * w_group

        x = x.reshape(B, h_group, self.ws, w_group, self.ws, C).transpose(2, 3)

        qkv = self.h_qkv(x).reshape(B, total_groups, -1, 3, self.h_heads, self.h_dim // self.h_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, hw, n_head, ws*ws, head_dim

        attn = (q @ k.transpose(-2, -1)) * self.scale  # B, hw, n_head, ws*ws, ws*ws
        attn = attn.softmax(dim=-1)
        attn = (attn @ v).transpose(2, 3).reshape(B, h_group, w_group, self.ws, self.ws, self.h_dim)
        x = attn.transpose(2, 3).reshape(B, h_group * self.ws, w_group * self.ws, self.h_dim)

        x = self.h_proj(x)
        return x

    def lofi(self, x):
        B, H, W, C = x.shape
        N = H * W
        # x = self.project_inp(x)
        q = self.l_q(x).reshape(B, H * W, self.l_heads, self.l_dim // self.l_heads).permute(0, 2, 1, 3)

        if self.ws > 1:
            x_ = x.permute(0, 3, 1, 2)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            kv = self.l_kv(x_).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.l_kv(x).reshape(B, -1, 2, self.l_heads, self.l_dim // self.l_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, self.l_dim)


        # 开始分块计算
        # chunk_size = min(self.chunk_size, N)
        # pool_outputs = []
        # for i in range(0, N, chunk_size):
        #     end_idx = min(i + chunk_size, N)
            
        #     # 获取当前块的 q
        #     q_chunk = q[:, :, i:end_idx, :]
            
        #     # 计算当前块的注意力
        #     attn_pool_chunk = (q_chunk @ k.transpose(-2, -1)) * self.scale
        #     attn_pool_chunk = attn_pool_chunk.softmax(dim=-1)
            
        #     # 计算当前块的 v
        #     x_pool_chunk = attn_pool_chunk @ v
            
        #     # 将结果存入列表
        #     pool_outputs.append(x_pool_chunk)
            
        #     # 释放缓存
        #     del attn_pool_chunk
        #     torch.cuda.empty_cache()

        # # 将所有块拼接起来
        # x = torch.cat(pool_outputs, dim=2)
        # del pool_outputs
        # torch.cuda.empty_cache()

        # # 恢复到原始的维度
        # x = x.transpose(1, 2).reshape(B, H, W, self.l_dim)

        x = self.l_proj(x)
        return x
        

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)

        hifi_out = self.hifi(x)
        lofi_out = self.lofi(x)

        x = torch.cat((hifi_out, lofi_out), dim=-1)
        x = x.reshape(B, H*W, C)
        x = x.permute(0,2,1).view(B,C,H,W)
        return x

class GFE(nn.Module):
    def __init__(self, channels, window_size=5, shifts=0, split_size=1, attention_type='dot attention'):
        super(GFE, self).__init__()
        self.shifts = shifts
        self.split_size = split_size
        self.window_size = window_size
        self.attention_type = attention_type
        self.split_chns = [channels * 2 // 3 for _ in range(3)]
        self.project_inp = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=1),
                                         nn.BatchNorm2d(channels * 2))
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)


    def forward(self, x):
        b, c, h, w = x.shape
        x = self.project_inp(x)
        xs = torch.split(x, self.split_chns, dim=1)
        wsize = self.window_size
        fuse_out = []
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

        return out




class MPAB(nn.Module):
    def __init__(self, inp_channels, out_channels, exp_ratio=2, window_size=5, act_type='gelu', shifts=0, split_size=1, attention_type='dot attention'):
        super(MPAB, self).__init__()
        self.exp_ratio = exp_ratio
        self.inp_channels = inp_channels
        self.out_channels = out_channels
        self.shifts = shifts
        self.LFE = LFE(inp_channels=inp_channels, out_channels=out_channels, exp_ratio=exp_ratio, act_type=act_type)
        self.GFE = GFE(channels=inp_channels, window_size=window_size, shifts=shifts, split_size=split_size, attention_type='dot attention')
        # self.HiLo = HiLo(dim=inp_channels)


    def forward(self, x):
        # x = self.GFE(x)
        # x = self.HiLo(x) + x
        x = self.GFE(x) + x
        x = self.LFE(x) + x
        return x
