import torch

vae_path = 'models/vqgan_cfw_00011.ckpt'

with open(vae_path, 'rb') as f:
    vae_ckpt = torch.load(f, map_location='cpu')

prune_keys = []
for k, v in vae_ckpt['state_dict'].items():
    if 'decoder.fusion_layer' in k:
        prune_keys.append(k)
        print(k)

vae_cfw = {}
for k in prune_keys:
    vae_cfw[k] = vae_ckpt['state_dict'][k]
    del vae_ckpt['state_dict'][k]

torch.save(vae_ckpt, 'models/vqgan_cfw_00011_vae_only.ckpt')
torch.save(vae_cfw, 'models/vqgan_cfw_00011_cfw_only.ckpt')
