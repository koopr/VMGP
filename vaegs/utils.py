import torch
import numpy as np
from torch.utils.data import DataLoader

def try_gpu(i=0):
    """Return gpu(i) if exists, otherwise return cpu().

    Defined in :numref:`sec_use_gpu`"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def summary(data, name):
    data = data.numpy()
    print('summary of', name)
    print('size:',len(data))
    print('mean:',np.mean(data, axis=0))
    print('std:',np.std(data, axis=0))
    print('max:',np.max(data, axis=0))
    print('min:',np.min(data, axis=0))

def summarylable(sub_dataset, name='data'):
    sdl = DataLoader(sub_dataset, batch_size=len(sub_dataset))
    for batch_idx, (data, labels) in enumerate(sdl):
        summary(labels,name)

def appendresult(result,filename):
    fn = 'output_data/' + filename
    with open(fn, 'a+') as f:
        for line in result:
            f.write('\n'+line) 

def get_mask(length, masked):
    indicies = np.random.choice(range(1, length), size=masked, replace=False)
    mask = np.zeros(length)
    mask[indicies] = 1
    return mask

def save_model(net, f):
    torch.save(net, f)

def load_model(f, device):
    return torch.load(f).to(device)