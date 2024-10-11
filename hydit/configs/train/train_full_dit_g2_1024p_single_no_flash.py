_base_ = [
    '../base/dataset/single_porcelain.py',
    '../base/model/dit_g2_1024_p.py',
    '../base/model/lora_r64.py',
    '../base/model/controlnet_canny.py',
    '../base/model/diffusion_v_pred.py',
    '../base/schedule/train_full.py',
    '../base/schedule/inference.py'
]

task_flag = 'dit_g2_full_1024p'

batch_size = 1                  # training batch size
use_fp16 = True 
extra_fp16 = True

lr = 0.0001     # learning rate
epochs = 8      # total training epochs

log_every = 10 

ckpt_every = 9999999        # create a ckpt every a few steps.
ckpt_latest_every = 9999999 # create a ckpt named `latest.pt` every a few steps.
ckpt_every_n_epoch=2        # create a ckpt every a few epochs.

global_seed = 999 

warmup_num_steps=0      # warm_up steps

rope_img = 'base512'
rope_real = True

uncond_p = 0 
uncond_p_t5 = 0 

results_dir = './log_EXP'   # save root for results
resume = True
# checkpoint root for model resume
resume_module_root = './ckpts/t2i/model/pytorch_model_module.pt'
# checkpoint root for ema resume
resume_ema_root = './ckpts/t2i/model/pytorch_model_ema.pt'          

use_zero_stage = 2 
grad_accu_steps=1               # gradient accumulation
cpu_offloading = True
gradient_checkpointing = True

# model

predict_type = 'v_prediction' 
noise_schedule = 'scaled_linear'
beta_start = 0.00085
beta_end = 0.018

# dataset

random_flip = True


