import torch
import torch.nn as nn
import torch.nn.functional as F

class Splitter(nn.Module):
    def __init__(self, S_type = 'A', S_requires_grad = 'False'):
        super(Splitter, self).__init__()
        
        if S_type == 'A':
            '2: blur kernel for splitter-based bayer pattern' 'RG/GB' 'ref-0'
            patterns = torch.zeros((4, 3, 1, 3, 3), dtype = torch.float32)
            patterns[0] = torch.tensor([[[[0.00, 0.00, 0.00], [0.00, 1.00, 0.00], [0.00, 0.00, 0.00]]],    # top-left: R
                                        [[[0.00, 0.25, 0.00], [0.25, 0.00, 0.25], [0.00, 0.25, 0.00]]],
                                        [[[0.25, 0.00, 0.25], [0.00, 0.00, 0.00], [0.25, 0.00, 0.25]]]])
            patterns[1] = torch.tensor([[[[0.00, 0.00, 0.00], [0.50, 0.00, 0.50], [0.00, 0.00, 0.00]]],    # top-right: G1
                                        [[[0.00, 0.00, 0.00], [0.00, 1.00, 0.00], [0.00, 0.00, 0.00]]],
                                        [[[0.00, 0.50, 0.00], [0.00, 0.00, 0.00], [0.00, 0.50, 0.00]]]])
            patterns[2] = torch.tensor([[[[0.00, 0.50, 0.00], [0.00, 0.00, 0.00], [0.00, 0.50, 0.00]]],    # bottom-left: G2
                                        [[[0.00, 0.00, 0.00], [0.00, 1.00, 0.00], [0.00, 0.00, 0.00]]],
                                        [[[0.00, 0.00, 0.00], [0.50, 0.00, 0.50], [0.00, 0.00, 0.00]]]])
            patterns[3] = torch.tensor([[[[0.25, 0.00, 0.25], [0.00, 0.00, 0.00], [0.25, 0.00, 0.25]]],    # bottom-right B
                                        [[[0.00, 0.25, 0.00], [0.25, 0.00, 0.25], [0.00, 0.25, 0.00]]],
                                        [[[0.00, 0.00, 0.00], [0.00, 1.00, 0.00], [0.00, 0.00, 0.00]]]])
            self.positions = torch.tensor([[0, 1], [2, 3]])
        elif S_type == 'D':
            '3: blur kernel for splitter-based bayer pattern' '(W+R)(W-R)/(W-B)(W+B)' 'ref-1'  
            patterns = torch.zeros((4, 3, 1, 1, 3), dtype = torch.float32)
            patterns[0] = torch.tensor([[[[0.0, 1.0, 0.0]]], [[[0.0, 1.0, 0.0]]], [[[0.0, 1.0, 0.0]]]])   # top-left:     W+R
            patterns[1] = torch.tensor([[[[0.5, 0.0, 0.5]]], [[[0.0, 1.0, 0.0]]], [[[0.0, 1.0, 0.0]]]])   # top-right:    W-R
            patterns[2] = torch.tensor([[[[0.0, 1.0, 0.0]]], [[[0.0, 1.0, 0.0]]], [[[0.5, 0.0, 0.5]]]])   # bottom-left:  W-B
            patterns[3] = torch.tensor([[[[0.0, 1.0, 0.0]]], [[[0.0, 1.0, 0.0]]], [[[0.0, 1.0, 0.0]]]])   # bottom-right: W+B
            self.positions = torch.tensor([[0, 1], [2, 3]])
        else:
            NotImplementedError

        if not S_requires_grad:
            self.patterns = nn.Parameter(patterns, requires_grad = False)
        else:
            patterns = torch.where(patterns == 0, 0, 1).float()
            self.patterns = nn.Parameter(patterns.mul(5), requires_grad = True)

        self.soft_max = S_requires_grad
            

    def forward(self, img):
        S_N, S_C, _, S_H, S_W = self.patterns.shape
        if self.soft_max:
            patterns = self.patterns.reshape((S_N*S_C, S_H*S_W)).softmax(dim = 1).reshape((S_N, S_C, 1, S_H, S_W))
        else:
            patterns = self.patterns

        P_H, P_W = self.positions.shape
        N, C, H, W = img.shape
        raw = torch.zeros((N, 1, H, W), dtype = img.dtype, device = img.device)
        for ii in range(P_H):
            for jj in range(P_W):
                # sample pixels for each splitter
                sample = torch.zeros((N, C, H, W), dtype = img.dtype, device = img.device)
                sample[:, :, ii:H:P_H, jj:W:P_W] = img[:, :, ii:H:P_H, jj:W:P_W].clone()

                # imaging process for each splitter
                pattern = patterns[self.positions[ii][jj]]
                raw[:,0,:,:] += F.conv_transpose2d(sample, pattern, groups = 3)[..., S_H//2:S_H//2+H, S_W//2:S_W//2+W].sum(dim=1)
    
        return raw
    
    def get_patterns(self):
        S_N, S_C, _, S_H, S_W = self.patterns.shape
        if self.soft_max:
            patterns = self.patterns.reshape((S_N*S_C, S_H*S_W)).softmax(dim = 1).reshape((S_N, S_C, 1, S_H, S_W))
        else:
            patterns = self.patterns
        return patterns