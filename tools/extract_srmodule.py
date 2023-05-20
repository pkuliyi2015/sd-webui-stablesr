'''
    This script extracts the spade and structcond module from the official stablesr_000117.ckpt
'''

import torch

stablesr_path = 'models/stablesr_000117.ckpt'


with open(stablesr_path, 'rb') as f:
    stablesr_ckpt = torch.load(f, map_location='cpu')

srmodule = {}
for k, v in stablesr_ckpt['state_dict'].items():
    if 'spade' in k or 'structcond' in k:
        srmodule[k] = v
        # print(k) 
# save

torch.save(srmodule, 'models/stablesr_sd21.ckpt')