import math
import torch
import torch.nn as nn

from ldm.modules.diffusionmodules.openaimodel import (
    TimestepEmbedSequential, 
    ResBlock, 
    Downsample, 
)

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    timestep_embedding,
    checkpoint,
    normalization,
    zero_module,
)

from srmodule.attn import get_attn_func

attn_func = None


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.
        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        # Legacy Attention
        # scale = 1 / math.sqrt(math.sqrt(ch))
        # weight = torch.einsum(
        #     "bct,bcs->bts", q * scale, k * scale
        # )  # More stable with f16 than dividing afterwards
        # weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        # a = torch.einsum("bts,bcs->bct", weight, v)
        # a = a.reshape(bs, -1, length)
        q, k, v = map(
            lambda t:t.permute(0,2,1)
            .contiguous(),
            (q, k, v),
        )
        global attn_func
        a = attn_func(q, k, v)
        a = (
            a.permute(0,2,1)
            .reshape(bs, -1, length)
        )
        return a

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.
    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)   # TODO: check checkpoint usage, is True # TODO: fix the .half call!!!

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


class EncoderUNetModelWT(nn.Module):
    """
    The half UNet model with attention and timestep embedding.
    For usage, see UNet.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = []
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        input_block_chans.append(ch)
        self._feature_size += ch
        self.input_block_chans = input_block_chans

        self.fea_tran = nn.ModuleList([])

        for i in range(len(input_block_chans)):
            self.fea_tran.append(
                ResBlock(
                    input_block_chans[i],
                    time_embed_dim,
                    dropout,
                    out_channels=out_channels,
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm,
                )
            )

    @torch.no_grad()
    def forward(self, x, timesteps):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :return: an [N x K] Tensor of outputs.
        """
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))

        result_list = []
        results = {}
        h = x.type(self.dtype)
        for module in self.input_blocks:
            last_h = h
            h = module(h, emb)
            if h.size(-1) != last_h.size(-1):
                result_list.append(last_h)
        h = self.middle_block(h, emb)
        result_list.append(h)

        assert len(result_list) == len(self.fea_tran)

        for i in range(len(result_list)):
            results[str(result_list[i].size(-1))] = self.fea_tran[i](result_list[i], emb)

        return results
    
    def load_from_dict(self, state_dict):
        """
        Load model weights from a dictionary.
        :param state_dict: a dict of parameters.
        """
        filtered_dict = {}
        for k, v in state_dict.items():
            if k.startswith("structcond_stage_model."):
                filtered_dict[k[len("structcond_stage_model.") :]] = v
        self.load_state_dict(filtered_dict)


def build_unetwt() -> EncoderUNetModelWT:
    """
    Build a model from a state dict.
    :param state_dict: a dict of parameters.
    :return: a nn.Module.
    """
    # The settings is from official setting yaml file.
    # https://github.com/IceClear/StableSR/blob/main/configs/stableSRNew/v2-finetune_text_T_512.yaml

    model = EncoderUNetModelWT(
        in_channels=4,
        model_channels=256,
        out_channels=256,
        num_res_blocks=2,
        attention_resolutions=[ 4, 2, 1 ],
        dropout=0.0,
        channel_mult=[1, 1, 2, 2],
        conv_resample=True,
        dims=2,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=4,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
    )
    global attn_func
    attn_func = get_attn_func()
    return model


if __name__ == "__main__":
    '''
    Test the lr encoder model.
    '''
    path = '../models/stablesr_sd21.ckpt'
    state_dict = torch.load(path)
    for key in state_dict.keys():
        print(key)
    model = build_unetwt()
    model.load_from_dict(state_dict)
    model = model.cuda()
    test_latent = torch.randn(1, 4, 64, 64).half().cuda()
    test_timesteps = torch.tensor([0]).half().cuda()
    with torch.no_grad():
        test_result = model(test_latent, test_timesteps)
    print(test_result.keys())