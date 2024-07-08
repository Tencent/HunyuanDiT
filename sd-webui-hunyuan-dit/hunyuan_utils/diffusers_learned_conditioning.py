from modules import prompt_parser, shared

def get_learned_conditioning_hunyuan(batch: prompt_parser.SdConditioning | list[str]):
    clip_l_conds, clip_l_attention = shared.clip_l_model(batch)
    t5_conds, t5_attention = shared.mt5_model(batch)
    return {"crossattn":clip_l_conds, "mask":clip_l_attention, "crossattn_2":t5_conds, "mask_2":t5_attention}