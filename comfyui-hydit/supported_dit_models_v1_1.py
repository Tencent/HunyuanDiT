import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import torch
from collections import namedtuple
from .hydit_v1_1.modules.models_comfyui import HunYuanDiT as HYDiT
import pdb

def batch_embeddings(embeds, batch_size):
	bs_embed, seq_len, _ = embeds.shape
	embeds = embeds.repeat(1, batch_size, 1)
	embeds = embeds.view(bs_embed * batch_size, seq_len, -1)
	return embeds
    
class HunYuan_DiT(comfy.supported_models_base.BASE):
    Conf = namedtuple('DiT', ['learn_sigma', 'text_states_dim', 'text_states_dim_t5', 'text_len', 'text_len_t5', 'norm', 'infer_mode', 'use_fp16'])
    conf = {
        'learn_sigma': True,
        'text_states_dim': 1024,
        'text_states_dim_t5': 2048,
        'text_len': 77,
        'text_len_t5': 256,
        'norm': 'layer',
        'infer_mode': 'torch',
        'use_fp16': True
    }

    unet_config = {}
    unet_extra_config = {
        "num_heads": 16
    }
    latent_format = comfy.latent_formats.SDXL

    dit_conf = Conf(**conf)

    def __init__(self, model_conf):
        self.unet_config = model_conf.get("unet_config", {})
        #print(model_conf)
        print(self.unet_config)
        self.sampling_settings = model_conf.get("sampling_settings", {})
        self.latent_format = self.latent_format()
        self.unet_config["disable_unet_model_creation"] = True
        #self.unet_config["disable_unet_model_creation"] = self.unet_config.get("disable_unet_model_creation", True)

    def model_type(self, state_dict, prefix=""):
        return comfy.model_base.ModelType.V_PREDICTION
    
class HYDiT_Model(comfy.model_base.BaseModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def extra_conds(self, **kwargs):
        out = super().extra_conds(**kwargs)
        addit_embeds = kwargs['cross_attn'].addit_embeds
        for name in addit_embeds:
            out[name] = comfy.conds.CONDRegular(addit_embeds[name])

        return out
    
class ModifiedHunYuanDiT(HYDiT):
    def forward_core(self, *args, **kwargs):
        return super().forward(*args, **kwargs)

    def forward(self, x, timesteps, context, t5_embeds=None, attention_mask=None, t5_attention_mask=None, image_meta_size=None, **kwargs):
        batch_size, _, height, width = x.shape
        #assert(0)
        #pdb.set_trace()
     
        
        style = torch.as_tensor([0, 0] * (batch_size//2), device=x.device)
        src_size_cond = (height//2*16, width//2*16)
        size_cond = list(src_size_cond) + [width*8, height*8, 0, 0]
        image_meta_size = torch.as_tensor([size_cond] * batch_size, device=x.device)
        rope = self.calc_rope(*src_size_cond)


        noise_pred = self.forward_core(
            x = x.to(self.dtype),
            t = timesteps.to(self.dtype),
            encoder_hidden_states = context.to(self.dtype),
            text_embedding_mask   = attention_mask.to(self.dtype),
            encoder_hidden_states_t5 = t5_embeds.to(self.dtype),
            text_embedding_mask_t5   = t5_attention_mask.to(self.dtype),
            image_meta_size = image_meta_size.to(self.dtype),
            style = style,
            cos_cis_img = rope[0],
            sin_cis_img = rope[1],
            return_dict=False
        )
        noise_pred = noise_pred.to(torch.float)
        eps, _ = noise_pred[:, :self.in_channels], noise_pred[:, self.in_channels:]
        return eps

