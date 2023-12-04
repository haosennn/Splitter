import argparse
import os
import json
import random
import logging
import pandas as pd

import numpy as np
import torch
from torch.utils.data import DataLoader

import data
import networks
from option import dict_to_nonedict
from utils import util_logger
from utils import util_path
from utils import util_image


def main():
    '''
    # ----------------------------------------
    # Step 1: set experimental setting
    # ----------------------------------------
    '''
    # 1-1) get option for this testing experiment
    parser = argparse.ArgumentParser(description='Splitter')
    # Environment
    parser.add_argument('--gpu_ids', type = str, nargs = '+', default = ['0'], help = 'gpus to use')
    parser.add_argument('--manual_seed', type = int, default = 1, help = 'manual seed for random generators')
    parser.add_argument('--random_seed', action = 'store_true', help = 'turn on flag to use random seed')
    # Trained Model
    parser.add_argument('--root_exps', type = str,  default ='./exps', help = 'root path of experimental records')
    parser.add_argument('--exp', type = int, default = 1, help = 'the index of model training experiment')
    parser.add_argument('--step', type = int, default = int(1e6), help = 'the number of training step')
    # Test setting (data & noise level)
    parser.add_argument('--root_data', type = str,  default ='./dataset', help = 'root path of datasets')
    parser.add_argument('--data_test', type = str, default = 'Kodak24', help='dataset used for test')   # Kodak24 | McMaster
    parser.add_argument('--rgb_range', type = float, default = 1.0, help = 'maximal value of gray/rgb channels')
    parser.add_argument('--sigmas', type = int, nargs = '+', default = [15, 15], help = 'noise level for the denoising task' )
    # Saving
    parser.add_argument('--save', action = 'store_true', help = 'turn on flag to save images')
    opt = parser.parse_args()
    # Post-processing
    if len(opt.sigmas) == 1:
        opt.sigmas = [opt.sigmas[0], opt.sigmas[0]]

    # 1-2) get paths
    opt.paths = util_path.get_paths_for_testing(opt)
    opt = vars(opt)
    opt = dict_to_nonedict(opt)
    
    # 1-3) configure GPU setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(opt['gpu_ids'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 1-4) configure seeds for random number generators
    opt['seed'] = opt['manual_seed'] if not opt['random_seed'] else random.randint(1, 10000)
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])

    # 1-5）configure logger
    logger_name = 'test'
    util_logger.set_logger(logger_name, opt['paths']['log'])
    logger  = logging.getLogger(logger_name)

    # 1-6) print option information
    logger.info('The option used for this test is as follows:' + util_logger.dict2str(opt))

    '''
    # ----------------------------------------
    # Step 2: set dataloader
    # ----------------------------------------
    '''
    test_ds = data.create_dataset(opt, 'test')
    test_dl = DataLoader(dataset = test_ds,
                          batch_size = 1,
                          num_workers = 0,
                          shuffle = False,
                          drop_last = False,
                          pin_memory = True)    

    '''
    # ----------------------------------------
    # Step 3: create network and load parameters
    # ----------------------------------------
    '''
    # 3-1) load options used during training
    with open(opt['paths']['opt_train'], 'r') as json_file:
        opt_train = json.load(json_file)

    # 3-2) create network
    opt_train['sigmas'] = opt['sigmas']
    net = networks.create_network(opt_train).to(device)
    
    # 3-3) Load the network parameters
    print(opt['paths']['model'])
    net.load_state_dict(torch.load(opt['paths']['model']))
    net.to(device)

    # 3-4) print the splitter patterns
    logger.info('The splitter partterns involved in this evaluation are as follows:')
    logger.info(net.get_splitter()['pattern'])

    '''
    # ----------------------------------------
    # Step 4: start testing
    # ----------------------------------------
    '''
    with torch.no_grad():
        net.eval()
        # ssim_avg, psnr_avg = 0.0, 0.0
        names, psnrs, ssims = [], [], []
        # start evaluation
        for _, imgs in enumerate(test_dl):
            img_name = str.split(str.split(''.join(imgs['path_target']), '/')[-1], '.')[0]
            names.append(img_name)
            # from cpu to speficied device (e.g., gpu)
            imgs_target = imgs['target'].to(device)
            # forward process
            imgs_output, imgs_raw, noise = net(imgs_target)
            # from specified device (e.g., gpu) to cpu
            imgs_target = imgs_target.to(torch.device("cpu"))
            imgs_output = imgs_output.to(torch.device("cpu"))
            # squeeze, quantize, and from tensor to numpy 
            factor = 255.0/opt['rgb_range']
            imgs_target = imgs_target.squeeze(dim=0).mul(factor).clamp(0, 255).round().numpy()#.astype('float64')
            imgs_output = imgs_output.squeeze(dim=0).mul(factor).clamp(0, 255).round().numpy()#.astype('float64')
            # calculate psnr and ssim
            psnr = util_image.cal_psnr_for_numpy(imgs_output, imgs_target, 255.0)
            ssim = util_image.cal_ssim_for_numpy(imgs_output, imgs_target, 0, False)
            psnrs.append(psnr)
            ssims.append(ssim)
            logger.info('[{}] {} PSNR: {:.4f}, SSIM:{:.4f}'.format(opt['data_test'], ('['+img_name+']').ljust(12), psnr, ssim))
            # save target image
            if opt['save']:
                util_image.imwrite(imgs_target.astype('uint8'), os.path.join(opt['paths']['image'], img_name+'.png'))

        # record and print validation result
        psnr_avg, ssim_avg  = np.mean(psnrs), np.mean(ssims)
        logger.info('[{}] Average PSNR: {:.4f}, Average SSIM: {:.4f}'.format(opt['data_test'], psnr_avg, ssim_avg))

        # save metrics to '.xlsx' file
        psnrs.append(psnr_avg)
        ssims.append(ssim_avg)
        names.append('Average')
        psnrs = np.array(psnrs).round(decimals=4)
        ssims = np.array(ssims).round(decimals=4)
        df = pd.DataFrame(zip(psnrs, ssims), index = names, columns=['psnr', 'ssim'])
        df.to_excel(os.path.join(opt['paths']['log'], opt['data_test']+'.xlsx'), index=True)
    
if __name__ == '__main__':
    main()