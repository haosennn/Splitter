import torch
import torch.nn as nn
import random

from networks.net_splitter import Splitter 
from networks.net_reconstruction import Reconstruction 

class Imaging_System(nn.Module):
    def __init__(self, sigmas = [0, 30], rgb_range = 255.0, S_type='A', S_requires_grad='False', n_resblocks=10, n_features=64):
        super(Imaging_System, self).__init__()
        # set the normalization factor
        self.factor = 255.0/rgb_range
        # set the range of the noise level
        self.sigmas = [float(x) for x in sigmas] 
        # set the imaging network
        self.rgb2raw = Splitter(S_type = S_type, S_requires_grad = S_requires_grad)
        # set the reconstruction network
        s_positions = self.rgb2raw.positions
        self.raw2rgb = Reconstruction(s_positions = s_positions, n_resblocks = n_resblocks, n_features = n_features)

    def forward(self, img):
        N, _, H, W = img.shape

        # Capture a raw image from a RGB image through splitter-based imaging process
        raw = self.rgb2raw(img)

        # Add noise to the captured raw image
        sigma = random.uniform(self.sigmas[0], self.sigmas[1])
        noise = torch.randn((N, 1, H, W), dtype = img.dtype, device = img.device) * sigma
        raw  += noise/self.factor

        # Reconstruct the RGB information from the noisy raw image
        img = self.raw2rgb(raw)
    
        # return img, raw
        return img, raw, noise
    
    def get_splitter(self):
        return {'pattern':  self.rgb2raw.get_patterns(), 
                'position': self.rgb2raw.positions}