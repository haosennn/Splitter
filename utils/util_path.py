import os
import shutil
from natsort import natsorted

'''
# --------------------------------------------
# get a brief description from the specified option
# --------------------------------------------
'''
def get_opt_desc(opt):
    desc =  str()
    
    if opt.debug:
        desc = 'debug'
    else:
        desc = 'Pattern' + '_' + opt.S_type
        if opt.S_learn:
            desc += '_Optimized'

    return desc

'''
# --------------------------------------------
# get paths from specified option (training phase)
# --------------------------------------------
'''
def get_paths_for_training(opt):
    # create an empty dictionary
    paths = dict()

    # set paths for training/validation data
    paths['data_train'] = os.path.join(opt.root_data, opt.data_train)                   
    paths['data_valid'] = os.path.join(opt.root_data, opt.data_valid)             
    
    # set path for this experiment
    exp_desc = get_opt_desc(opt)
    if opt.debug:
        exp_idx = 0
    elif not os.path.exists(opt.root_exps) or not os.listdir(opt.root_exps):
        exp_idx = 1
    else:
        paths_history = sorted(os.listdir(opt.root_exps), key=lambda x: int(x.split('_')[0]))
        exp_idx = int(paths_history[-1].split('_')[0])+1
    path_this_exp = os.path.join(opt.root_exps, '{}_{}'.format(exp_idx, exp_desc))

    # set and create paths to store source files, log files, and trained models
    path_names = ['src', 'logs', 'models']
    for path_name in path_names:
        paths[path_name] = os.path.join(path_this_exp, path_name)
        if not os.path.exists(paths[path_name]):
            os.makedirs(paths[path_name])

    # return paths
    return paths

'''
# --------------------------------------------
# get paths from specified option (testing phase)
# --------------------------------------------
'''
def get_paths_for_testing(opt):
    # create an empty dictionary
    paths = dict()

    # set path for test data
    paths['data_test'] = os.path.join(opt.root_data, opt.data_test) 

    # set path for the specified training experiment
    path_exp = os.path.join(opt.root_exps, natsorted(os.listdir(opt.root_exps))[opt.exp])

    # set path for the training option
    paths['opt_train'] = os.path.join(path_exp, 'src', 'opt.json')

    # set path for the trained model
    for name in os.listdir(os.path.join(path_exp, 'models')):
        step = int(name.split('_')[-1][:-4])
        if step == opt.step:
            paths['model'] = os.path.join(path_exp, 'models', name)

    # set path for saving log files and processed images
    paths['log'] = os.path.join(path_exp, 'tests_S_{:07d}_N_{:02d}-{:02d}'.format(opt.step, opt.sigmas[0], opt.sigmas[1]))
    paths['image'] = os.path.join(paths['log'], opt.data_test)
    mkdir(paths['image'])

    return paths

'''
# --------------------------------------------
# makedir
# --------------------------------------------
'''
def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


'''
# --------------------------------------------
# copydir
# --------------------------------------------
'''
def copydir(dir_src, dir_dst):
    files = os.listdir(dir_src)
    for file in files:
        src = os.path.join(dir_src, file)
        dst = os.path.join(dir_dst, file)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)