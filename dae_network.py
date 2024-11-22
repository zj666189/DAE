import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.dae_block import MPAB, MeanShift

def create_model(args):
    return MPAN(args)


class MPAN(nn.Module):
    def __init__(self, args):
        super(MPAN, self).__init__()

        self.scale = args.scale
        self.colors = args.colors
        self.split_size = args.split_size
        self.window_sizes = args.window_sizes
        self.n_mpab = args.n_mpab
        self.c_mpan = args.c_mpan
        self.r_expand = args.r_expand
        self.act_type = args.act_type

        self.sub_mean = MeanShift(args.rgb_range)
        self.add_mean = MeanShift(args.rgb_range, sign=1)

        # define head module
        m_head = [nn.Conv2d(self.colors, self.c_mpan, kernel_size=3, stride=1, padding=1)]
        # define body module
        self.body = nn.ModuleList()
        number = self.n_mpab // len(self.window_sizes)
        for i in range(number):
            if i % 2 == 0:
                for window_size, split_size, idx in zip(self.window_sizes, self.split_size, [0, 1, 0]):
                    self.body.append(MPAB(self.c_mpan, self.c_mpan, self.r_expand,
                                          window_size, act_type=self.act_type,
                                          shifts=0 if (idx == 0) else window_size//2,
                                          split_size=split_size))
            else:
                for window_size, split_size, idx in zip(self.window_sizes, self.split_size, [1, 0, 1]):
                    self.body.append(MPAB(self.c_mpan, self.c_mpan, self.r_expand,
                                          window_size, act_type=self.act_type,
                                          shifts=0 if (idx == 0) else window_size//2,
                                          split_size=split_size))

        self.conv_after_body = nn.Conv2d(self.c_mpan, self.c_mpan, 3, 1, 1)
        # define tail module
        m_tail = [
            nn.Conv2d(self.c_mpan, self.colors * self.scale * self.scale, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(self.scale)
        ]

        self.head = nn.Sequential(*m_head)
        # self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        H, W = x.shape[2:]
        x = self.check_image_size(x)

        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for idx, stage in enumerate(self.body):
            res = stage(res)
        res = self.conv_after_body(res)
        res = res + x
        x = self.tail(res)
        x = self.add_mean(x)
        return x[:, :, 0:H * self.scale, 0:W * self.scale]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def load(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            name = name[name.index('.') + 1:]
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                    # own_state[name].requires_grad = False
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


import argparse, yaml
from thop import profile, clever_format
from nni.compression.utils.counter import count_flops_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MSTN')
    ## yaml configuration files
    parser.add_argument('--config', type=str, default='./configs/mpan_x2.yml')
    args = parser.parse_args()
    opt = vars(args)
    yaml_args = yaml.load(open(args.config), Loader=yaml.FullLoader)
    opt.update(yaml_args)
    model = create_model(args).to('cuda:1')

    x = torch.rand(1, 3, 96, 96).to('cuda:1')
    x = model(x).to('cuda:1')
    print(x.shape)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("params: %d" % num_params)

    macs, params = profile(model, inputs=(x,))
    flops, params = clever_format([macs, params], "%.3f")
    print(flops, params)

    flops, params, results = count_flops_params(model, x)
    print(flops, params)

