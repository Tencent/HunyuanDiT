# =========================
#   LoRA Config
# =========================

rank = 64                   # Rank of LoRA
lora_ckpt = None            # LoRA checkpoint
target_modules = ['Wqkv', 'q_proj', 'kv_proj', 'out_proj']  # Target modules for LoRA fine tune
output_merge_path = None    # Output path for merged model