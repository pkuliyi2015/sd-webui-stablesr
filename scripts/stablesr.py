'''
# --------------------------------------------------------------------------------
#
#   StableSR for Automatic1111 WebUI
#
#   Introducing state-of-the super-resolution method: StableSR!
#   Techniques is originally proposed by my schoolmate Jianyi Wang et, al.
#
#   Project Page: https://iceclear.github.io/projects/stablesr/
#   Official Repo: https://github.com/IceClear/StableSR
#   Paper: https://arxiv.org/abs/2305.07015
#   
#   @original author: Jianyi Wang et, al.
#   @migration: LI YI 
#   @organization: Nanyang Technological University - Singapore
#   @date: 2023-05-20
#   @license: 
#       S-Lab License 1.0 (see LICENSE file)
#       CC BY-NC-SA 4.0 (required by NVIDIA SPADE module)
# 
#   @disclaimer: 
#       All code in this extension is for research purpose only. 
#       The commercial use of the code & checkpoint is strictly prohibited.
#
# --------------------------------------------------------------------------------
#
#   IMPORTANT NOTICE FOR OUTCOME IMAGES:
#       - Please be aware that the CC BY-NC-SA 4.0 license in SPADE module
#         also prohibits the commercial use of outcome images.
#       - Jianyi Wang may change the SPADE module to a commercial-friendly one.
#         If you want to use the outcome images for commercial purposes, please
#         contact Jianyi Wang for more information.
#
#   Please give me a star (and also Jianyi's repo) if you like this project!
#
# --------------------------------------------------------------------------------
'''

import os
import torch
import gradio as gr
import numpy as np
import PIL.Image as Image

from pathlib import Path
from torch import Tensor
from tqdm import tqdm

from modules import scripts, processing, sd_samplers, devices, images, shared
from modules.processing import StableDiffusionProcessingImg2Img, Processed
from modules.shared import opts
from ldm.modules.diffusionmodules.openaimodel import UNetModel

from srmodule.spade import SPADELayers
from srmodule.struct_cond import EncoderUNetModelWT, build_unetwt
from srmodule.colorfix import adain_color_fix, wavelet_color_fix

SD_WEBUI_PATH = Path.cwd()
ME_PATH = SD_WEBUI_PATH / 'extensions' / 'sd-webui-stablesr'
MODEL_PATH = ME_PATH / 'models'
FORWARD_CACHE_NAME = 'org_forward_stablesr'

class StableSR:
    def __init__(self, path, dtype, device):
        state_dict = torch.load(path, map_location='cpu')
        self.struct_cond_model: EncoderUNetModelWT = build_unetwt()
        self.spade_layers: SPADELayers = SPADELayers()
        self.struct_cond_model.load_from_dict(state_dict)
        self.spade_layers.load_from_dict(state_dict)
        del state_dict
        self.struct_cond_model.apply(lambda x: x.to(dtype=dtype, device=device))
        self.spade_layers.apply(lambda x: x.to(dtype=dtype, device=device))

        self.latent_image: Tensor = None
        self.set_image_hooks = {}
        self.struct_cond: Tensor = None

    def set_latent_image(self, latent_image):
        self.latent_image = latent_image
        for hook in self.set_image_hooks.values():
            hook(latent_image)

    def hook(self, unet: UNetModel):
        # hook unet to set the struct_cond
        if not hasattr(unet, FORWARD_CACHE_NAME):
            setattr(unet, FORWARD_CACHE_NAME, unet.forward)

        def unet_forward(x, timesteps=None, context=None, y=None,**kwargs):
            self.latent_image = self.latent_image.to(x.device)
            # Ensure the device of all modules layers is the same as the unet
            # This will fix the issue when user use --medvram or --lowvram
            self.spade_layers.to(x.device)
            self.struct_cond_model.to(x.device)
            timesteps = timesteps.to(x.device)
            self.struct_cond = None # mitigate vram peak
            self.struct_cond = self.struct_cond_model(self.latent_image, timesteps[:self.latent_image.shape[0]])
            return getattr(unet, FORWARD_CACHE_NAME)(x, timesteps, context, y, **kwargs)
        
        unet.forward = unet_forward

        self.spade_layers.hook(unet, lambda: self.struct_cond)


    def unhook(self, unet: UNetModel):
        # clean up cache
        self.latent_image = None
        self.struct_cond = None
        self.set_image_hooks = {}
        # unhook unet forward
        if hasattr(unet, FORWARD_CACHE_NAME):
            unet.forward = getattr(unet, FORWARD_CACHE_NAME)
            delattr(unet, FORWARD_CACHE_NAME)

        # unhook spade layers
        self.spade_layers.unhook()


class Script(scripts.Script):
    def __init__(self) -> None:
        self.model_list = {}
        self.load_model_list()
        self.last_path = None
        self.stablesr_model: StableSR = None

    def load_model_list(self):
        # traverse the CFG_PATH and add all files to the model list
        self.model_list = {}
        if not MODEL_PATH.exists():
            MODEL_PATH.mkdir()
        for file in MODEL_PATH.iterdir():
            if file.is_file():
                # save tha absolute path
                self.model_list[file.name] = str(file.absolute())
        self.model_list['None'] = None

    def title(self):
        return "StableSR"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        with gr.Row():
            model = gr.Dropdown(list(self.model_list.keys()), label="SR Model")
            refresh = gr.Button(value='↻', variant='tool')
            def refresh_fn(selected):
                self.load_model_list()
                if selected not in self.model_list:
                    selected = 'None'
                return gr.Dropdown.update(value=selected, choices=list(self.model_list.keys()))
            refresh.click(fn=refresh_fn,inputs=model, outputs=model)
        with gr.Row():
            scale_factor = gr.Slider(minimum=1, maximum=16, step=0.1, value=2, label='Scale Factor', elem_id=f'StableSR-scale')
        with gr.Row():
            color_fix = gr.Dropdown(['None', 'Wavelet', 'AdaIN'], label="Color Fix", value='Wavelet', elem_id=f'StableSR-color-fix')
            save_original = gr.Checkbox(label='Save Original', value=False, elem_id=f'StableSR-save-original', visible=color_fix.value != 'None')
            color_fix.change(fn=lambda selected: gr.Checkbox.update(visible=selected != 'None'), inputs=color_fix, outputs=save_original, show_progress=False)
            pure_noise = gr.Checkbox(label='Pure Noise', value=True, elem_id=f'StableSR-pure-noise')
            unload_model= gr.Button(value='Unload Model', variant='tool')
            def unload_model_fn():
                if self.stablesr_model is not None:
                    self.stablesr_model = None
                    devices.torch_gc()
                    print('[StableSR] Model unloaded!')
                else:
                    print('[StableSR] No model loaded.')
            unload_model.click(fn=unload_model_fn)
        return [model, scale_factor, pure_noise, color_fix, save_original]

    def run(self, p: StableDiffusionProcessingImg2Img, model: str, scale_factor:float, pure_noise: bool, color_fix:str, save_original:bool) -> Processed:

        if model == 'None':
            # do clean up
            self.stablesr_model = None
            self.last_model_path = None
            return
        
        if model not in self.model_list:
            raise gr.Error(f"Model {model} is not in the list! Please refresh your browser!")
        
        if not os.path.exists(self.model_list[model]):
            raise gr.Error(f"Model {model} is not on your disk! Please refresh the model list!")

        if color_fix not in ['None', 'Wavelet', 'AdaIN']:
            print(f'[StableSR] Invalid color fix method: {color_fix}')
            color_fix = 'None'

        # upscale the image, set the ouput size 
        init_img: Image = p.init_images[0]
        target_width = int(init_img.width * scale_factor)
        target_height = int(init_img.height * scale_factor)
        # if the target width is not dividable by 8, then round it up
        if target_width % 8 != 0:
            target_width = target_width + 8 - target_width % 8
        # if the target height is not dividable by 8, then round it up
        if target_height % 8 != 0:
            target_height = target_height + 8 - target_height % 8
        init_img = init_img.resize((target_width, target_height), Image.LANCZOS)
        p.init_images[0] = init_img
        p.width = init_img.width
        p.height = init_img.height

        print('[StableSR] Target image size: {}x{}'.format(init_img.width, init_img.height))

        first_param = shared.sd_model.parameters().__next__()
        if self.last_path != self.model_list[model]:
            # load the model
            self.stablesr_model = None
        
        if self.stablesr_model is None:
            self.stablesr_model = StableSR(self.model_list[model], dtype=first_param.dtype, device=first_param.device)
            self.last_path = self.model_list[model]

        def sample_custom(conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength, prompts):
            try:
                unet: UNetModel = shared.sd_model.model.diffusion_model
                self.stablesr_model.hook(unet)
                self.stablesr_model.set_latent_image(p.init_latent)
                x = processing.create_random_tensors(p.init_latent.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength, seed_resize_from_h=p.seed_resize_from_h, seed_resize_from_w=p.seed_resize_from_w, p=p)
                sampler = sd_samplers.create_sampler(p.sampler_name, p.sd_model)
                if pure_noise:
                    # NOTE: use txt2img instead of img2img sampling
                    samples = sampler.sample(p, x, conditioning, unconditional_conditioning, image_conditioning=p.image_conditioning)
                else:
                    if p.initial_noise_multiplier != 1.0:
                        p.extra_generation_params["Noise multiplier"] =p.initial_noise_multiplier
                        x *= p.initial_noise_multiplier
                    samples = sampler.sample_img2img(p, p.init_latent, x, conditioning, unconditional_conditioning, image_conditioning=p.image_conditioning)
                
                if p.mask is not None:
                    samples = samples * p.nmask + p.init_latent * p.mask
                del x
                devices.torch_gc()
                return samples
            finally:
                self.stablesr_model.unhook(unet)
                # in --medvram and --lowvram mode, we send the model back to the initial device
                self.stablesr_model.struct_cond_model.to(device=first_param.device)
                self.stablesr_model.spade_layers.to(device=first_param.device)

                
        # replace the sample function
        p.sample = sample_custom
        
        if color_fix != 'None':
            p.do_not_save_samples = True

        result: Processed = processing.process_images(p)

        if color_fix != 'None':

            fixed_images = []
            # fix the color
            color_fix_func = wavelet_color_fix if color_fix == 'Wavelet' else adain_color_fix
            for i in range(len(result.images)):
                try:
                    fixed_images.append(color_fix_func(result.images[i], init_img))
                except Exception as e:
                    print(f'[StableSR] Error fixing color with default method: {e}')

            # save the fixed color images
            for i in range(len(fixed_images)):
                try:
                    images.save_image(fixed_images[i], p.outpath_samples, "", p.all_seeds[i], p.all_prompts[i], opts.samples_format, info=result.infotexts[i], p=p)
                except Exception as e:
                    print(f'[StableSR] Error saving color fixed image: {e}')

            if save_original:
                for i in range(len(result.images)):
                    try:
                        images.save_image(result.images[i], p.outpath_samples, "", p.all_seeds[i], p.all_prompts[i], opts.samples_format, info=result.infotexts[i], p=p, suffix="-before-color-fix")
                    except Exception as e:
                        print(f'[StableSR] Error saving original image: {e}')
            result.images = result.images + fixed_images

        return result
