
def create_network(opt):

    from networks.net_joint import Imaging_System as N
    net = N(sigmas          = opt['sigmas'], 
            rgb_range       = opt['rgb_range'], 
            S_type          = opt['S_type'], 
            S_requires_grad = opt['S_learn'], 
            n_features      = opt['n_features'], 
            n_resblocks     = opt['n_resblocks'])

    return net
