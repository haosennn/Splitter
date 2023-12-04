import option
import data
import networks
from utils import util_logger
from utils import util_path
from utils import util_image

import os
import time
import random
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

def main():
    '''
    # ----------------------------------------
    # Step 1: set experimental setting
    # ----------------------------------------
    '''
    # 1-1) set the experimental option
    opt = option.get_opt()

    # 1-2) configure logger
    logger_name = 'train'
    util_logger.set_logger(logger_name, opt['paths']['logs'])
    logger  = logging.getLogger(logger_name)
    logger.info('The logger configuration is completed.')
    
    # 1-3) configure GPU setting
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(opt['gpu_ids'])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logger.info('The GPU configuration is completed.')

    # 1-4) configure seeds for random number generators
    random.seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.manual_seed(opt['seed'])
    torch.cuda.manual_seed_all(opt['seed'])
    logger.info('The seed configuration is completed.')

    # 1-5) save option and source files
    path_opt = os.path.join(opt['paths']['src'] ,'opt.json')
    option.opt_save(opt, path_opt)
    logger.info('The option is saved.' + util_logger.dict2str(opt))
    
    if opt['debug']:
        option.opt_save(opt, './opt.json')

    '''
    # ----------------------------------------
    # Step 2: set dataloader
    # ----------------------------------------
    '''
    # 2-1) dataloder for training set
    train_ds = data.create_dataset(opt, 'train')
    train_dl = DataLoader(dataset = train_ds,
                          batch_size = opt['batch_size'],
                          num_workers = opt['n_workers'],
                          shuffle = True,
                          drop_last = True,
                          pin_memory = True)    

    # 2-2) dataloder for validation set
    valid_ds = data.create_dataset(opt, 'valid')
    valid_dl = DataLoader(dataset = valid_ds,
                          batch_size = 1,
                          num_workers = 0,
                          shuffle = False,
                          drop_last = False,
                          pin_memory = True)
    
    logger.info('Dataloaders for training and validation are configured.')
    
    '''
    # ----------------------------------------
    # Step 3: create network, loss function, and optimizer
    # ----------------------------------------
    '''
    # 3-1) create the network
    net = networks.create_network(opt).to(device)
    logger.info('The network is created, its info is provided below:')
    logger.info(net) # logger.info(summary(net, (3, 64, 64), device=device, verbose=0)) 

    # 3-2) define the loss function
    criterion = torch.nn.MSELoss().to(device)
    logger.info('The loss function is set as L2 (MSE) norm.')

    # 3-3) define the optimizer and the scheduler
    optimizer = torch.optim.Adam(net.parameters(), lr=opt['lr'], betas=opt['betas'])
    logger.info('Optimizer is configured.')

    '''
    # ----------------------------------------
    # Step 4: (main training)
    # ----------------------------------------
    '''
    writer = SummaryWriter(opt['paths']['logs'])
    step_current, time_start = 0, time.time()
    time_data, time_model, time_tic = 0.0, 0.0, time.time()         # time record
    for _ in range(int(1e8)):   # keep running...
        if step_current > opt['n_max_step']:
            break

        for _, imgs in enumerate(train_dl):
            time_toc = time.time()                                  # time record
            time_data += time_toc - time_tic                        # time record
            time_tic = time.time()                                  # time record

            # 4-1) update current step
            step_current += 1
            if step_current > opt['n_max_step']:
                break

            # 4-2) transfer data to specified device
            imgs_target = imgs['target'].to(device)

            # 4-3) zero gradients
            net.train()
            optimizer.zero_grad()

            # 4-4) forward process
            # imgs_output, imgs_raw, noise = net(imgs_target)
            imgs_output, _, _ = net(imgs_target)

            # 4-5) calculate loss
            loss = criterion(imgs_output, imgs_target)

            # 4-6) backward process
            loss.backward()

            # 4-7) update network parameters
            optimizer.step()

            # ----------------------------------------
            # 4-8) check point 1：print training information 
            # ----------------------------------------
            if step_current % opt['freq_print_train'] == 0:
                with torch.no_grad():
                    # Calculate PSNR on training set
                    psnrs = util_image.cal_PSNRs(imgs_output, imgs_target, opt['rgb_range'])
                    # record and print validation result
                    writer.add_scalar('loss on training data', loss.item(), step_current)
                    writer.add_scalar('PSNR on training data', torch.mean(psnrs).item(), step_current)
                    logger.info("[train] [steps %07d] time elapsed: %.1f, loss: %.6f, psnr_Train: %.4f"
                      % (step_current, time.time()-time_start, loss.item(),  torch.mean(psnrs).item()))
            
            # ----------------------------------------
            # 4-9) check point 2：evaluate trained network 
            # ----------------------------------------
            if step_current % opt['freq_print_valid'] == 0:
                with torch.no_grad():
                    net.eval()
                    loss_avg, psnr_avg = 0.0, 0.0
                    # start evaluation
                    for _, imgs in enumerate(valid_dl):
                        # transfer validation data to specified device
                        imgs_target = imgs['target'].to(device)
                        # forward process
                        imgs_output, _, _ = net(imgs_target)
                        # calculate loss
                        loss = criterion(imgs_output, imgs_target)
                        loss_avg += loss.item()
                        # calculate psnr
                        psnr = util_image.cal_PSNRs(imgs_output, imgs_target, opt['rgb_range'])
                        psnr_avg += psnr.mean().item()
                    # record and print validation result
                    loss_avg, psnr_avg = loss_avg/len(valid_dl), psnr_avg/len(valid_dl)
                    writer.add_scalar('loss on validation set', loss_avg, step_current)
                    writer.add_scalar('PSNR on validation set', psnr_avg, step_current)
                    logger.info('')
                    logger.info("[valid] [steps %07d] time elapsed: %.1f, loss: %.6f, psnr_Val: %.4f"
                                % (step_current, time.time()-time_start, loss_avg, psnr_avg))
                    logger.info('')
            
            # ----------------------------------------
            # 4-10) check point 3：save trained network 
            # ----------------------------------------
            if step_current % opt['freq_save_check'] == 0:
                save_path = os.path.join(opt['paths']['models'], 'S_{:07d}.pth'.format(step_current))
                state_dict = net.state_dict()
                for key, param in state_dict.items():
                    state_dict[key] = param.cpu()
                torch.save(state_dict, save_path)

            time_toc = time.time()                                  # time record
            time_model += time_toc - time_tic                       # time record
            time_tic = time.time()                                  # time record
            if step_current % 100 == 0:
                print("time_data:{}, time_model:{}".format(time_data, time_model))
                time_data, time_model = 0.0, 0.0                    # time record

    logger.info('Training completed')


if __name__ == '__main__':
    main()