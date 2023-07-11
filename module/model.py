import torch, os
import torch.nn as nn
from model.base_model import B_Transformer
from model.torch_model import T_Transformer
from model.hybrid_model import H_Transformer




def init_weights(model):
    for name, param in model.named_parameters():
        if 'weight' in name and 'norm' not in name:
            nn.init.xavier_uniform_(param)            



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
    if config.model_type == 'base':
        model = B_Transformer(config)
    elif config.model_type == 'torch':
        model = T_Transformer(config)
    elif config.model_type == 'hybrid':
        model = H_Transformer(config)        
    
    init_weights(model)
    print(f"Initialized {config.model_type} model has loaded")

    if config.mode != 'train':
        assert os.path.exists(config.ckpt)
        model_state = torch.load(config.ckpt, map_location=config.device)['model_state_dict']
        model.load_state_dict(model_state)
        print(f"Model states has loaded from {config.ckpt}")       
    
    print(f"--- Model Params: {count_params(model):,}")
    print(f"--- Model  Size : {check_size(model):.3f} MB\n")
    return model.to(config.device)