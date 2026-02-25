import torch
import torch.nn as nn
import torch.nn.functional as F

class Adapter(nn.Module):
    def __init__(self, norm_nc, label_nc, kernel_size=3, norm_type='instance'):
        super().__init__()

        if norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE' % norm_type)

        nhidden = 64
        pw = kernel_size // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=kernel_size, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=kernel_size, padding=pw)

    def forward(self, x):
        normalized = self.param_free_norm(x)

        actv = self.mlp_shared(x)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
       

        return normalized * (1 + gamma) + beta


class Adapter_MultiField(nn.Module):
    def __init__(self, fields, norm_nc, label_nc, kernel_size, norm_type='instance'):
        super().__init__()
        self.adapters = nn.ModuleDict({
            field: Adapter(norm_nc, label_nc, kernel_size, norm_type)
            for field in fields
        })


    def forward(self, x, field):
        if field not in self.adapters:
            raise ValueError('%s is not a recognized modality in SPADE_Multimodal' % field)
        return self.adapters[field](x)


class AdapterResnet(nn.Module):
    def __init__(self, contrasts, fin, fout):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        self.norm_0 = Adapter_MultiField(contrasts, fin, fin, kernel_size=3)
        self.norm_1 = Adapter_MultiField(contrasts, fmiddle, fmiddle, kernel_size=3)

    def forward(self, x, fields):
        x_s = self.shortcut(x)
        dx = self.conv_0(self.actvn(self.norm_0(x, fields)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, fields)))
        return x_s + dx

    def shortcut(self, x):
        return self.conv_s(x) if self.learned_shortcut else x

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)



class AdaptiveFieldGen(nn.Module):
    def __init__(self,fields, z_dim=1, nf=128):
        super().__init__()
        self.block = nn.ModuleList([
            AdapterResnet(fields, z_dim, nf),
            AdapterResnet(fields, nf, z_dim)
        ])

    def forward(self, x, y):
        for block in self.block:
            x = block(x, y)
        return x

