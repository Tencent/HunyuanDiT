task_flag = ''
training_parts = 'all'  # Training parts, choices=['all', 'lora']

# General Setting
seed = 42           # A seed for all the prompts
batch_size = 1      # Per GPU batch size
use_fp16 = True     # Use FP16 precision
extra_fp16 = False  # Use extra fp16 for vae and text_encoder
lr = 1e-4
epochs = 100
max_training_steps = 10_000_000
gc_interval = 40    # To address the memory bottleneck encountered during the preprocessing of the dataset, memory fragments are reclaimed here by invoking the gc.collect() function.
log_every = 100
ckpt_every = 100_000        # Create a ckpt every a few steps.
ckpt_latest_every = 10_000  # Create a ckpt named `latest.pt` every a few steps.
ckpt_every_n_epoch = 0      # Create a ckpt every a few epochs. If 0, do not create ckpt based on epoch. Default is 0.
num_workers = 4
global_seed = 1234
warmup_min_lr = 1e-6
warmup_num_steps = 0
weight_decay = 0    # Weight_decay in optimizer
rope_img = None     # Extend or interpolate the positional embedding of the image, choices=['extend', 'base512', 'base1024'] 
rope_real = False   # Use real part and imaginary part separately for RoPE.

# Classifier_free
uncond_p = 0.2      # The probability of dropping training text used for CLIP feature extraction
uncond_p_t5 = 0.2   # The probability of dropping training text used for mT5 feature extraction

# Directory
results_dir = 'results'
resume = False
resume_module_root = None   # Resume model states.
resume_ema_root = None      # Resume ema states.
strict = True               # Strict loading of checkpoint

# Additional condition
random_shrink_size_cond = False # Randomly shrink the original size condition.
merge_src_cond = False          # Merge the source condition into a single value.

# EMA Model
use_ema = False         # Use EMA model
ema_dtype = 'none'      # EMA data type. If none, use the same data type as the model, choices=['fp16', 'fp32', 'none']
ema_decay = None        # EMA decay rate. If None, use the default value of the model.
ema_warmup = False      # EMA warmup. If True, perform ema_decay warmup from 0 to ema_decay.
ema_warmup_power = None # EMA power. If None, use the default value of the model.
ema_reset_decay = False # Reset EMA decay to 0 and restart increasing the EMA decay. Only works when ema_warmup is enabled.

# Acceleration
use_flash_attn = False          # During training, flash attention is used to accelerate training.
use_zero_stage = 1              # Use AngelPTM zero stage. Support 2 and 3
grad_accu_steps = 1             # Gradient accumulation steps.
gradient_checkpointing = False  # Use gradient checkpointing.
cpu_offloading = False          # Use cpu offloading for parameters and optimizer states.
save_optimizer_state = False    # Save optimizer state in the checkpoint.

# DeepSpeed
local_rank = None
deepspeed_optimizer = False # Switching to the optimizers in DeepSpeed
remote_device = 'none'      # Remote device for ZeRO_3 initialized parameters, choices=['none', 'cpu', 'nvme']
zero_stage = 1