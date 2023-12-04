import torch
import torch.nn as nn
import torch.fft as fft
import math
'''
# --------------------------------------------
# Useful blocks 
# --------------------------------
# ConvBlock: Conv2d | ReLU | BN
# ResBlock: residual block
# SampleBlock: 
# --------------------------------------------
'''

# --------------------------------------------
# ConvBlock: Conv2d | ReLU | BN
# --------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels=64, out_channels=64, kernel_size=3, bias=True, layers='CR'):
        super(ConvBlock, self).__init__()
        B = []
        for layer in layers:
            if layer == 'C':
                B.append(nn.Conv2d( in_channels  = in_channels,
                                    out_channels = out_channels,
                                    kernel_size  = kernel_size,
                                    padding      = kernel_size//2,
                                    bias         = bias))
            elif layer == 'R':
                B.append(nn.ReLU(inplace=True))
            elif layer == 'r':
                B.append(nn.ReLU(inplace=False))
            elif layer == 'B':
                B.append(nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-04, affine=True))
            else:
                raise NotImplementedError('Undefined layer type: '.format(layer))
        self.B = nn.Sequential(*B)

    def forward(self, x):
        return self.B(x)
    

# --------------------------------------------
# ResBlock: x + Conv2d(Relu(Conv2d(x)))
# --------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, n_features=64, kernel_size=3, bias=True):
        super(ResBlock, self).__init__()
        self.res = ConvBlock( in_channels  = n_features,
                              out_channels = n_features,
                              kernel_size  = kernel_size,
                              bias         = bias,
                              layers       = 'CRC' )
        
    def forward(self, x):
        res = self.res(x)
        return x + res

# --------------------------------------------
# SampleBlock: sample and concatenate raw images with specified positions 
# --------------------------------------------
class SampleBlock(nn.Module):
    def __init__(self, positions):
        super(SampleBlock, self).__init__()
        self.positions = positions
        
    def forward(self, x_in):
        h_step, w_step = self.positions.shape
        N, _, H, W = x_in.shape            # the shape of raw image: N-1-H-W
        x_out = torch.zeros((N, h_step*w_step, H, W), dtype = x_in.dtype, device = x_in.device)

        for h in range(h_step):
            for w in range(w_step):
                x_out[:, h*w_step+w, h:H:h_step, w:W:w_step] = x_in[:, 0, h:H:h_step, w:W:w_step].clone()

        return x_out
