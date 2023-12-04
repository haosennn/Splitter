import torch.nn as nn
import networks.basicblock as B

class Reconstruction(nn.Module):
    def __init__(self, s_positions, n_resblocks = 10, n_features = 64, kernel_size = 3, bias = True):
        super(Reconstruction, self).__init__()
        # define sampling module
        net_sampling = [B.SampleBlock(s_positions)]

        # define head module
        net_head = [B.ConvBlock(s_positions.numel(), n_features, kernel_size, bias, 'C')]
        
        # define body module
        net_body = [B.ResBlock(n_features, kernel_size, bias) for _ in range(n_resblocks)]
        net_body.append(B.ConvBlock(n_features, n_features, kernel_size, bias, 'C'))

        # define tail module
        net_tail = [B.ConvBlock(n_features, 3, kernel_size, bias, 'C')]
        
        self.sampling = nn.Sequential(*net_sampling)
        self.head = nn.Sequential(*net_head)
        self.body = nn.Sequential(*net_body)
        self.tail = nn.Sequential(*net_tail)

    def forward(self, x):

        x = self.sampling(x)

        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x