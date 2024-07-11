train_config = {
    'task_flag': 'lora_porcelain_ema_rank64',           # the task flag is used to identify folders
    'model_config': {
        'model': 'DiT-g/2',                             # choices = ["DiT-g/2", "DiT-XL/2"]
        'image_size': [1024, 1024],                     # training image resolution
        'qk_norm': True,                                # Query Key normalization
        'norm': 'layer',                                # normalization layer type, choices=["rms", "layer"]
        'text_states_dim': 1024,                        # hidden size of CLIP text encoder
        'text_len': 77,                                 # token length of CLIP text encoder output
        'text_states_dim_t5': 2048,                     # hidden size of CLIP text encoder
        'text_len_t5': 256,                             # token length of T5 text encoder output
        'learn_sigma': False,                           # learn extra channels for sigma
        'predict_type': 'v_prediction',                 # choices = ["epsilon", "sample", "v_prediction"]
        'noise_schedule': 'scaled_linear',              # choices = ["linear", "scaled_linear", "squaredcos_cap_v2"]
        'beta_start': 0.00085,                          # beta start value
        'beta_end': 0.03,                               # beta end value
        'sigma_small': False,                           # if True, use a smaller fixed sigma otherwise a larger one
        'mse_loss_weight_type': 'constant',             # Min-SNR-gamma, choices = ['constant', 'min_snr_<gamma>'(gamma is a integer)]
        'model_var_type': None,                         # specify the model variable type
        'noise_offset': 0.0                             # add extra noise to the input image
    },
    'dataset_config': {
        'batch_size': 1,                                # per-GPU batch size
        'seed': 42,                                     # a seed for all the prompts
        'index_file':
            'dataset/porcelain/jsons/porcelain.json',   # index file for dataloader
        'random_flip': True,                            # random flip image
        'reset_loader': False,                          # reset the data loader
        'multireso': False,                             # use multi-resolution training
        'reso_step': None,                              # step size for multi-resolution training
        'random_shrink_size_cond': False,               # randomly shrink the original size condition
        'merge_src_cond': False                         # merge the source condition into a single value
    },
    'training_config': {
        'lr': 0.0001,                                   # learning rate
        'epochs': 1400,                                 # training epochs
        'max_training_steps': 2000,                 # max training steps
        'gc_interval': 40,                              # frequency (in steps) to invoke gc.collect()
        'log_every': 10,                                # frequency (in steps) to log training progress
        'ckpt_every': 100,                            # frequency (in steps) to create a ckpt
        'ckpt_latest_every': 2000,                      # frequency (in steps) to create a ckpt named `latest.pt`
        'num_workers': 4,                               # number of workers for data loading
        'global_seed': 999,                             # global random seed
        'warmup_min_lr': 0.000001,                      # minimum learning rate during warmup
        'warmup_num_steps': 0,                          # number of steps to warm up the learning rate
        'weight_decay': 0,                              # weight-decay in optimizer
        'rope_img': 'base512',                          # extend or interpolate the positional embedding of the image, choices = ['extend', 'base512', 'base1024']
        'rope_real': True,                              # use real part and imaginary part separately for RoPE
        'uncond_p': 0.44,                               # the probability of dropping training text used for CLIP feature extraction
        'uncond_p_t5': 0.44,                            # the probability of dropping training text used for mT5 feature extraction
        'results_dir': './log_EXP',                     # save root for results
        'resume': './ckpts/t2i/model/',                 # resume experiment from a checkpoint
        'strict': True,                                 # strict loading of checkpoint
        'resume_deepspeed': False,                      # resume model and ema states from a checkpoint saved by Deepspeed version of DIT
        'resume_split': True,                           # resume model and ema states from two checkpoint separated from DeepSpeed ckpt
        'ema_to_module': True,                          # if true, initialize the module with EMA weights
        'module_to_ema': False,                         # if true, initialize the ema with Module weights
        'use_ema': True,                                # use EMA model
        'ema_dtype': 'fp32',                            # choices = ['fp16', 'fp32', 'none']. if none, use the same data type as the model
        'ema_decay': None,                              # EMA decay rate. If None, use the default value of the model
        'ema_warmup': False,                            # EMA warmup. If True, perform ema_decay warmup from 0 to ema_decay
        'ema_warmup_power': None,                       # EMA power. If None, use the default value of the model
        'ema_reset_decay': False,                       # reset EMA decay to 0 and restart increasing the EMA decay
        'use_flash_attn': True,                         # use flash attention to accelerate training
        'use_zero_stage': 2,                            # use AngelPTM zero stage. choices = [1, 2, 3]
        'grad_accu_steps': 2,                           # gradient accumulation steps
        'use_fp16': True,                               # use FP16 precision
        'extra_fp16': False                             # use extra fp16 for vae and text_encoder
    },
    'lora_config': {
        'training_parts': 'lora',                       # training parts, choices=['all', 'lora']
        'rank': 64,                                     # rank of LoRA
        'lora_ckpt': None,                              # LoRA checkpoint
        'target_modules':
            ['Wqkv', 'q_proj', 'kv_proj', 'out_proj'],  # target modules for LoRA fine tune
        'output_merge_path': None                       # output path for merged model
    },
    'deepspeed_config': {
        'local_rank': None,                             # local rank passed from distributed launcher.
        'deepspeed_optimizer': True,                    # switching to the optimizers in DeepSpeed.
        'remote_device': 'none',                        # remote device for ZeRO-3 initialized parameters. choices = ['none', 'cpu', 'nvme'].
        'zero_stage': 1,                                # ZeRO optimization stage.
        'async_ema': False                              # whether to use multi-stream to execute EMA.
    }
}
