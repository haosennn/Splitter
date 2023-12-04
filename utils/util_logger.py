import logging
import os

'''
# --------------------------------------------
# Configure logger
# --------------------------------------------
'''
def set_logger(name, dir):

    # initialize logger
    logger = logging.getLogger(name)

    # set level
    level = logging.INFO
    logger.setLevel(level)

    # set formatter
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%y-%m-%d %H:%M:%S')

    # set and add FileHandler
    path = os.path.join(dir, (name + '.log'))
    fh = logging.FileHandler(path, mode='a')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # set and add StreamHandler
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

'''
# --------------------------------------------
# dict to string for logger
# --------------------------------------------
'''
def dict2str(opt, indent_l=2):
    msg = '\n'
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg
