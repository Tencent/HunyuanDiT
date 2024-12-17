# import comfy.utils
import logging
import torch
import numpy as np


def load_lora(lora, to_load, weight):
    model_dict = to_load
    patch_dict = {}
    loaded_keys = set()
    for x in to_load:
        alpha_name = "{}.alpha".format(x)
        alpha = None
        if alpha_name in lora.keys():
            alpha = lora[alpha_name].item()
            loaded_keys.add(alpha_name)
        dora_scale_name = "{}.dora_scale".format(x)
        dora_scale = None
        if dora_scale_name in lora.keys():
            dora_scale = lora[dora_scale_name]
            loaded_keys.add(dora_scale_name)
        hunyuan_lora = "unet.{}.lora.up.weight".format(
            x.replace(".weight", "").replace("_", ".")
        )
        A_name = None

        if hunyuan_lora in lora.keys():
            A_name = hunyuan_lora
            B_name = "unet.{}.lora.down.weight".format(
                x.replace(".weight", "").replace("_", ".")
            )
            mid_name = None
            bias_name = "{}.bias".format(x.replace(".weight", ""))

        if A_name is not None:
            mid = None
            if mid_name is not None and mid_name in lora.keys():
                mid = lora[mid_name]
                loaded_keys.add(mid_name)
            patch_dict[to_load[x]] = (
                "lora",
                (lora[A_name], lora[B_name], alpha, mid, dora_scale),
            )
            lora_update = torch.matmul(lora[A_name].to("cuda"), lora[B_name].to("cuda"))
            if alpha:
                lora_update *= alpha / lora[A_name].shape[1]
            else:
                lora_update /= np.sqrt(lora[A_name].shape[1])
            lora_update *= weight
            model_dict[x] += lora_update
            loaded_keys.add(A_name)
            loaded_keys.add(B_name)

    for x in lora.keys():
        if x not in loaded_keys:
            logging.warning("lora key not loaded: {}".format(x))
    return model_dict
