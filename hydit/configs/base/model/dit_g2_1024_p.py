# =========================
#   HunYuan_DiT Config
# =========================

model = 'DiT-g/2'           # choices=list(HUNYUAN_DIT_CONFIG.keys())
image_size = (1024, 1024)   # Image size (h, w)
qk_norm = True              # Query Key normalization. http://arxiv.org/abs/2302.05442 for details.
norm = 'layer'              # Normalization layer type, choices=["rms", "laryer"]
text_states_dim = 1024      # Hidden size of CLIP text encoder
text_len = 77               # Token length of CLIP text encoder output
text_states_dim_t5 = 2048   # Hidden size of CLIP text encoder
text_len_t5 = 256           # Token length of T5 text encoder output
