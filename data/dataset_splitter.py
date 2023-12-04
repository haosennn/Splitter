import os
import random
from natsort import natsorted

from torch.utils.data import Dataset

from utils.util_image import *

import time

class DatasetSplitter(Dataset):
    def __init__(self, root_target, patch_size, factor, phase):
        super(DatasetSplitter, self).__init__()
        self.phase = phase
        self.ps = patch_size
        self.factor = factor

        self.root_target = os.path.join(root_target, 'GT')

        self.fnames = natsorted(os.listdir(self.root_target))

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # read the target image 
        path_target = os.path.join(self.root_target, self.fnames[index])
        img_target  = imread(path_target)

        # crop and augment only for training phase
        if self.phase == 'train':
            _, H, W = imsize(img_target)

            ps    = self.ps
            rnd_h = random.randint(0, max(0, H - ps))
            rnd_w = random.randint(0, max(0, W - ps))
            img_target = img_target[:, rnd_h:rnd_h + ps, rnd_w:rnd_w + ps]
            
            mode = random.randint(0, 7)                     
            img_target = augment(img_target, mode)

        # do normalization (convert dtype and rescale value range)
        img_target  = normalize(img_target, factor=self.factor)

        # from numpy to tensor
        img_target = ndarray2tensor(img_target)

        return {'target':      img_target, 
                'path_target': path_target}
        