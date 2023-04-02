import torch, os
import torch.nn as nn
from model.base import BaseModel
from model.hier import HierModel




def init_xavier(model):
    for p in model.named_parameters():
        if 'weight' in p[0] and 'norm' not in p[0]:
            nn.init.xavier_uniform_(p[1])            



def count_params(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params
    


def check_size(model):
    param_size, buffer_size = 0, 0

    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb



def load_model(config):
    if config.task == 'sum':
        model = HierModel(config)
    else:
        model = BaseModel(config)
    
    model.apply(init_xavier)
    print(f"Initialized model for {config.task} task has loaded")

    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Model states has loaded from {config.ckpt}")       
    
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    return model.to(config.device)