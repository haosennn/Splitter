import argparse
import json
import random
from utils.util_path import get_paths_for_training as get_paths

'''
# --------------------------------------------
# get option from argparse
# --------------------------------------------
'''
def get_opt():
    parser = argparse.ArgumentParser(description='Splitter')

    # Experimental setting
    parser.add_argument('--debug', action = 'store_true', help = 'turn on flag to debug')
    parser.add_argument('--gpu_ids', type = str, nargs = '+', default = ['0'], help = 'gpus to use')
    parser.add_argument('--manual_seed', type = int, default = 1, help = 'manual seed for random generators')
    parser.add_argument('--random_seed', action = 'store_true', help = 'turn on flag to use random seed')

    # Root paths
    parser.add_argument('--root_exps', type = str,  default ='./exps', help = 'root path of experimental records')
    parser.add_argument('--root_data', type = str,  default ='./dataset', help = 'root path of datasets')

    # Dataloader
    parser.add_argument('--data_train', type = str, default = 'DIV2K_train_480x480_pt', help='dataset used for training') #DIV2K_train DIV2K_train_480x480_pt
    parser.add_argument('--data_valid', type = str, default = 'Kodak24', help='dataset used for validation')
    parser.add_argument('--rgb_range', type = float, default  = 1.0, help = 'maximal value of gray/rgb channels')
    parser.add_argument('--patch_size', type = int, default = 64, help = 'patch size of input image')
    parser.add_argument('--batch_size', type = int, default = 80, help = 'number of images in each batch')
    parser.add_argument('--n_workers', type = int, default = 0, help = 'number of threads to use in data loader')

    # Setting of imaging process (i.e., splitter network and noise levels)
    parser.add_argument('--S_type', type = str, default = 'A', choices = ['A', 'D'], help = 'splitter type')
    parser.add_argument('--S_learn', action = 'store_true', help = 'turn on flag to optimize splitter')
    parser.add_argument('--sigmas', type = int, nargs = '+', default = [0, 30], help = 'noise level' )

    # Setting of reconstruction network
    parser.add_argument('--n_resblocks', type = int, default = 10, help = 'number of residule blocks')
    parser.add_argument('--n_features', type = int, default = 64, help = 'number of features')

    # Optimizer 
    parser.add_argument('--lr', type = float, default = 1e-4, help = 'learning rate')
    parser.add_argument('--betas', type = tuple, default = (0.9, 0.999), help = 'ADAM betas')

    # Training setting
    parser.add_argument('--n_max_step', type = int, default = int(1e6), help = 'maximal number of iteration steps')
    parser.add_argument('--freq_print_train', type = int, default = 1e1, help = 'frequency of printing training info')
    parser.add_argument('--freq_print_valid', type = int, default = 1e3, help = 'frequency of printing validation info')
    parser.add_argument('--freq_save_check', type = int, default = 1e5, help = 'frequency of saving checkpoints')

    opt = parser.parse_args()

    if len(opt.sigmas) == 1:
        opt.sigmas = [opt.sigmas[0], opt.sigmas[0]]

    # Experimental paths
    opt.paths = get_paths(opt)

    # Seed for random number generators
    opt.seed = opt.manual_seed if not opt.random_seed else random.randint(1, 10000)

    # From 'namespace' to dict
    opt = vars(opt)
    # opt = dict_to_nonedict(opt)

    return opt


'''
# --------------------------------------------
# save option
# --------------------------------------------
'''
def opt_save(opt, path):
    # opt_dict = vars(opt)  # Convert data type from 'namespace' to 'dict'
    with open(path, 'w') as json_file:
        json.dump(opt, json_file, indent=4)

        
'''
# --------------------------------------------
# convert OrderedDict to NoneDict,
# return None for missing key
# --------------------------------------------
'''
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt


class NoneDict(dict):
    def __missing__(self, key):
        return None
