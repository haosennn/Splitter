
def create_dataset(opt, phase):

    from data.dataset_splitter import DatasetSplitter as D
    dataset = D(root_target = opt['paths']['data_'+phase], 
                patch_size  = opt['patch_size'], 
                factor      = 255.0/opt['rgb_range'], 
                phase       = phase)
        
    return dataset
