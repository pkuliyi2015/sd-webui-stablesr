"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import re
import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.util import normalization, checkpoint
from ldm.modules.diffusionmodules.openaimodel import ResBlock, UNetModel


class SPADE(nn.Module):
    def __init__(self, norm_nc, label_nc=256, config_text='spadeinstance3x3'):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        ks = int(parsed.group(2))
        self.param_free_norm = normalization(norm_nc)

        # The dimension of the intermediate embedding space. Yes, hardcoded.
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x_dic, segmap_dic):
        return checkpoint(
            self._forward, (x_dic, segmap_dic), self.parameters(), True
        )

    def _forward(self, x_dic, segmap_dic):
        segmap = segmap_dic[str(x_dic.size(-1))]
        x = x_dic

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        # segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)

        repeat_factor = normalized.shape[0]//segmap.shape[0]
        if repeat_factor > 1:
            out = normalized
            out *= (1 + self.mlp_gamma(actv).repeat_interleave(repeat_factor, dim=0))
            out += self.mlp_beta(actv).repeat_interleave(repeat_factor, dim=0)
        else:
            out = normalized
            out *= (1 + self.mlp_gamma(actv))
            out += self.mlp_beta(actv)
        return out
    
def dual_resblock_forward(self: ResBlock, x, emb, spade: SPADE, get_struct_cond):
    if self.updown:
        in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
        h = in_rest(x)
        h = self.h_upd(h)
        x = self.x_upd(x)
        h = in_conv(h)
    else:
        h = self.in_layers(x)
    emb_out = self.emb_layers(emb).type(h.dtype)
    while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]
    if self.use_scale_shift_norm:
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
    else:
        h = h + emb_out
        h = self.out_layers(h)
    h = spade(h, get_struct_cond())
    return self.skip_connection(x) + h

    
class SPADELayers(nn.Module):
    def __init__(self):
        '''
        A container class for fast SPADE layer loading.
        params inferred from the official checkpoint
        '''
        super().__init__()
        self.input_blocks = nn.ModuleList([
            nn.Identity(),
            SPADE(320),
            SPADE(320),
            nn.Identity(),
            SPADE(640),
            SPADE(640),
            nn.Identity(),
            SPADE(1280),
            SPADE(1280),
            nn.Identity(),
            SPADE(1280),
            SPADE(1280),
        ])
        self.middle_block = nn.ModuleList([
            SPADE(1280),
            nn.Identity(),
            SPADE(1280),
        ])
        self.output_blocks = nn.ModuleList([
            SPADE(1280),
            SPADE(1280),
            SPADE(1280),
            SPADE(1280),
            SPADE(1280),
            SPADE(1280),
            SPADE(640),
            SPADE(640),
            SPADE(640),
            SPADE(320),
            SPADE(320),
            SPADE(320),
        ])
        self.input_ids = [1,2,4,5,7,8,10,11]
        self.output_ids = list(range(12))
        self.mid_ids = [0,2]
        self.forward_cache_name = 'org_forward_stablesr'
        self.unet = None


    def hook(self, unet: UNetModel, get_struct_cond):
        # hook all resblocks
        self.unet = unet
        resblock: ResBlock = None
        for i in self.input_ids:
            resblock = unet.input_blocks[i][0]
            # debug
            # assert isinstance(resblock, ResBlock)
            if not hasattr(resblock, self.forward_cache_name):
                setattr(resblock, self.forward_cache_name, resblock._forward)
            resblock._forward = lambda x, timesteps, resblock=resblock, spade=self.input_blocks[i]: dual_resblock_forward(resblock, x, timesteps, spade, get_struct_cond)

        for i in self.output_ids:
            resblock = unet.output_blocks[i][0]
            # debug
            # assert isinstance(resblock, ResBlock)
            if not hasattr(resblock, self.forward_cache_name):
                setattr(resblock, self.forward_cache_name, resblock._forward)
            resblock._forward = lambda x, timesteps, resblock=resblock, spade=self.output_blocks[i]: dual_resblock_forward(resblock, x, timesteps, spade, get_struct_cond)

        for i in self.mid_ids:
            resblock = unet.middle_block[i]
            # debug
            # assert isinstance(resblock, ResBlock)
            if not hasattr(resblock, self.forward_cache_name):
                setattr(resblock, self.forward_cache_name, resblock._forward)
            resblock._forward = lambda x, timesteps, resblock=resblock, spade=self.middle_block[i]: dual_resblock_forward(resblock, x, timesteps, spade, get_struct_cond)

    def unhook(self):
        unet = self.unet
        if unet is None: return
        resblock: ResBlock = None
        for i in self.input_ids:
            resblock = unet.input_blocks[i][0]
            if hasattr(resblock, self.forward_cache_name):
                resblock._forward = getattr(resblock, self.forward_cache_name)
                delattr(resblock, self.forward_cache_name)

        for i in self.output_ids:
            resblock = unet.output_blocks[i][0]
            if hasattr(resblock, self.forward_cache_name):
                resblock._forward = getattr(resblock, self.forward_cache_name)
                delattr(resblock, self.forward_cache_name)

        for i in self.mid_ids:
            resblock = unet.middle_block[i]
            if hasattr(resblock, self.forward_cache_name):
                resblock._forward = getattr(resblock, self.forward_cache_name)
                delattr(resblock, self.forward_cache_name)
        self.unet = None


    def load_from_dict(self, state_dict):
        """
        Load model weights from a dictionary.
        :param state_dict: a dict of parameters.
        """
        filtered_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model.diffusion_model."):
                key = k[len("model.diffusion_model.") :]
                # remove the '.0.spade' within the key
                if 'middle_block' not in key:
                    key = key.replace('.0.spade', '')
                else:
                    key = key.replace('.spade', '')
                filtered_dict[key] = v
        self.load_state_dict(filtered_dict)


if __name__ == '__main__':
    path = '../models/stablesr_sd21.ckpt'
    state_dict = torch.load(path)
    model = SPADELayers()
    model.load_from_dict(state_dict)
    print(model)