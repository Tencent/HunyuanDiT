import comfy.supported_models_base
import comfy.latent_formats
import comfy.model_patcher
import comfy.model_base
import comfy.utils
import torch
from collections import namedtuple

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
        self.sampling_settings = model_conf.get("sampling_settings", {})
        self.latent_format = self.latent_format()
        self.unet_config["disable_unet_model_creation"] = True

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

