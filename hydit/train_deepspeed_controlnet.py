import gc
import json
import os
import random
import sys
import time
from functools import partial
from glob import glob
from pathlib import Path
import numpy as np

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.distributed.optim import ZeroRedundancyOptimizer
from torchvision.transforms import functional as TF
from diffusers.models import AutoencoderKL
from transformers import BertModel, BertTokenizer, logging as tf_logging

from hydit.config import get_args
from hydit.constants import VAE_EMA_PATH, TEXT_ENCODER, TOKENIZER, T5_ENCODER
from hydit.lr_scheduler import WarmupLR
from hydit.data_loader.arrow_load_stream import TextImageArrowStream
from hydit.diffusion import create_diffusion
from hydit.ds_config import deepspeed_config_from_args
from hydit.modules.ema import EMA
from hydit.modules.fp16_layers import Float16Module
from hydit.modules.models import HUNYUAN_DIT_MODELS
from hydit.modules.controlnet import HunYuanControlNet
from hydit.modules.posemb_layers import init_image_posemb
from hydit.utils.tools import create_logger, set_seeds, create_exp_folder, model_resume, get_trainable_params
from IndexKits.index_kits import ResolutionGroup
from IndexKits.index_kits.sampler import DistributedSamplerWithStartIndex, BlockDistributedSampler
from peft import LoraConfig, get_peft_model

from hydit.annotator.dwpose import DWposeDetector
torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler._LRScheduler
from transformers import pipeline
import cv2
from PIL import Image

depth_estimator = pipeline('depth-estimation', device='cuda:{}'.format(int(os.getenv('LOCAL_RANK', '0'))))
pose_detector = DWposeDetector()

def deepspeed_initialize(args, logger, model, opt, deepspeed_config):
    logger.info(f"Initialize deepspeed...")
    logger.info(f"    Using deepspeed optimizer")

    def get_learning_rate_scheduler(warmup_min_lr, lr, warmup_num_steps, opt):
        return WarmupLR(opt, warmup_min_lr, lr, warmup_num_steps)

    logger.info(f"    Building scheduler with warmup_min_lr={args.warmup_min_lr}, warmup_num_steps={args.warmup_num_steps}")
    model, opt, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=get_trainable_params(model),
        config_params=deepspeed_config,
        args=args,
        lr_scheduler=partial(get_learning_rate_scheduler, args.warmup_min_lr, args.lr, args.warmup_num_steps) if args.warmup_num_steps > 0 else None,
    )
    return model, opt, scheduler

def save_checkpoint(args, rank, logger, model, ema, epoch, train_steps, checkpoint_dir):
    def save_lora_weight(checkpoint_dir, client_state, tag=f"{train_steps:07d}.pt"):
        cur_ckpt_save_dir = f"{checkpoint_dir}/{tag}"
        if rank == 0:
            if args.use_fp16:
                model.module.module.save_pretrained(cur_ckpt_save_dir)
            else:
                model.module.save_pretrained(cur_ckpt_save_dir)

    checkpoint_path = "[Not rank 0. Disabled output.]"

    client_state = {
        "steps": train_steps,
        "epoch": epoch,
        "args": args
    }
    if ema is not None:
        client_state['ema'] = ema.state_dict()

    dst_paths = []
    if train_steps % args.ckpt_every == 0:
        checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state, tag=f"{train_steps:07d}.pt")
            else:
                model.save_checkpoint(checkpoint_dir, client_state=client_state, tag=f"{train_steps:07d}.pt")
            dst_paths.append(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except:
            logger.error(f"Saved failed to {checkpoint_path}")

    if train_steps % args.ckpt_latest_every == 0 or train_steps == args.max_training_steps:
        save_name = "latest.pt"
        checkpoint_path = f"{checkpoint_dir}/{save_name}"
        try:
            if args.training_parts == "lora":
                save_lora_weight(checkpoint_dir, client_state, tag=f"{save_name}")
            else:
                model.save_checkpoint(checkpoint_dir, client_state=client_state, tag=f"{save_name}")
            dst_paths.append(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        except:
            logger.error(f"Saved failed to {checkpoint_path}")

    dist.barrier()
    if rank == 0 and len(dst_paths) > 0:
        # Delete optimizer states to avoid occupying too much disk space.
        for dst_path in dst_paths:
            for opt_state_path in glob(f"{dst_path}/*_00_optim_states.pt"):
                os.remove(opt_state_path)

    return checkpoint_path

def get_canny(np_img, low_threshold = 100, high_threshold = 200):
    # tensor = deNormalize()
    # image = tensor_to_img(tensor)
    image = cv2.Canny(np_img, low_threshold,high_threshold)
    image = image[:,:,None]
    image = np.concatenate([image,image,image], axis=2)
    # canny_tensor = img_to_norm_tensor(canny_img)
    return image

def get_depth(np_img):
    pil_img = Image.fromarray(np_img)
    depth = depth_estimator(pil_img)['depth']
    depth = np.array(depth)
    depth = depth[:, :, None]
    depth = np.concatenate([depth, depth, depth], axis=2)
    return depth

def get_pose(np_img):
    return pose_detector(np_img)[0]

@torch.no_grad()
def prepare_model_inputs(args, batch, device, vae, text_encoder, text_encoder_t5, freqs_cis_img):
    image, text_embedding, text_embedding_mask, text_embedding_t5, text_embedding_mask_t5, kwargs = batch

    # clip & mT5 text embedding
    text_embedding = text_embedding.to(device)
    text_embedding_mask = text_embedding_mask.to(device)
    encoder_hidden_states = text_encoder(
        text_embedding.to(device),
        attention_mask=text_embedding_mask.to(device),
    )[0]
    text_embedding_t5 = text_embedding_t5.to(device).squeeze(1)
    text_embedding_mask_t5 = text_embedding_mask_t5.to(device).squeeze(1)
    with torch.no_grad():
        output_t5 = text_encoder_t5(
            input_ids=text_embedding_t5,
            attention_mask=text_embedding_mask_t5 if T5_ENCODER['attention_mask'] else None,
            output_hidden_states=True
        )
        encoder_hidden_states_t5 = output_t5['hidden_states'][T5_ENCODER['layer_index']].detach()

    # additional condition
    if args.size_cond:
        image_meta_size = kwargs['image_meta_size'].to(device)
    else:
        image_meta_size = None
    if args.use_style_cond:
        style = kwargs['style'].to(device)
    else:
        style = None

    np_img = image.squeeze(0).add(1).mul(255 / 2).permute(1, 2, 0).cpu().numpy().astype('uint8')
    if args.control_type == 'canny':
        condition = get_canny(np_img)
    elif args.control_type == 'depth':
        condition = get_depth(np_img)
    elif args.control_type == 'pose':
        condition = get_pose(np_img)
    else:
        raise NotImplementedError
    condtion = Image.fromarray(condition)
    condition = TF.to_tensor(condition)
    condition = TF.normalize(condition, [0.5], [0.5])
    condition = condition.unsqueeze(0).to(device)

    if args.extra_fp16:
        image = image.half()

    # Map input images to latent space + normalize latents:
    image = image.to(device)
    vae_scaling_factor = vae.config.scaling_factor
    latents = vae.encode(image).latent_dist.sample().mul_(vae_scaling_factor)
    condition = vae.encode(condition).latent_dist.sample().mul_(vae_scaling_factor)

    # positional embedding
    _, _, height, width = image.shape
    reso = f"{height}x{width}"
    cos_cis_img, sin_cis_img = freqs_cis_img[reso]

    # Model conditions
    model_kwargs = dict(
        encoder_hidden_states=encoder_hidden_states,
        text_embedding_mask=text_embedding_mask,
        encoder_hidden_states_t5=encoder_hidden_states_t5,
        text_embedding_mask_t5=text_embedding_mask_t5,
        image_meta_size=image_meta_size,
        style=style,
        cos_cis_img=cos_cis_img,
        sin_cis_img=sin_cis_img,
        condition=condition,
    )

    return latents, model_kwargs

def main(args):

    args.use_ema = False # EMA usage is discouraged during ControlNet training.
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    world_size = dist.get_world_size()
    batch_size = args.batch_size
    grad_accu_steps = args.grad_accu_steps
    global_batch_size = world_size * batch_size * grad_accu_steps

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * world_size + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")
    deepspeed_config = deepspeed_config_from_args(args, global_batch_size)

    # Setup an experiment folder
    experiment_dir, checkpoint_dir, logger = create_exp_folder(args, rank)

    # Log all the arguments
    logger.info(sys.argv)
    logger.info(str(args))
    # Save to a json file
    args_dict = vars(args)
    args_dict['world_size'] = world_size
    with open(f"{experiment_dir}/args.json", 'w') as f:
        json.dump(args_dict, f, indent=4)

    # Disable the message "Some weights of the model checkpoint at ... were not used when initializing BertModel."
    # If needed, just comment the following line.
    tf_logging.set_verbosity_error()

    # ===========================================================================
    # Building HYDIT
    # ===========================================================================

    logger.info("Building HYDIT Model.")

    # ---------------------------------------------------------------------------
    #   Training sample base size, such as 256/512/1024. Notice that this size is
    #   just a base size, not necessary the actual size of training samples. Actual
    #   size of the training samples are correlated with `resolutions` when enabling
    #   multi-resolution training.
    # ---------------------------------------------------------------------------
    image_size = args.image_size
    if len(image_size) == 1:
        image_size = [image_size[0], image_size[0]]
    if len(image_size) != 2:
        raise ValueError(f"Invalid image size: {args.image_size}")
    assert image_size[0] % 8 == 0 and image_size[1] % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder). " \
                                                              f"got {image_size}"
    latent_size = [image_size[0] // 8, image_size[1] // 8]

    # initialize model by deepspeed
    assert args.deepspeed, f"Must enable deepspeed in this script: train_deepspeed.py"
    with deepspeed.zero.Init(data_parallel_group=torch.distributed.group.WORLD,
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=deepspeed_config,
                             mpu=None,
                             enabled=args.zero_stage == 3):
        model = HUNYUAN_DIT_MODELS[args.model](args,
                                       input_size=latent_size,
                                       log_fn=logger.info,
                                        )
        controlnet = HunYuanControlNet(args,
                                       input_size=latent_size,
                                       depth=40, hidden_size=1408, patch_size=2, num_heads=16, mlp_ratio=4.3637,
                                       log_fn=logger.info,
                                        )
    # Multi-resolution / Single-resolution training.
    if args.multireso:
        resolutions = ResolutionGroup(image_size[0],
                                      align=16,
                                      step=args.reso_step,
                                      target_ratios=args.target_ratios).data
    else:
        resolutions = ResolutionGroup(image_size[0],
                                      align=16,
                                      target_ratios=['1:1']).data

    freqs_cis_img = init_image_posemb(args.rope_img,
                                      resolutions=resolutions,
                                      patch_size=model.patch_size,
                                      hidden_size=model.hidden_size,
                                      num_heads=model.num_heads,
                                      log_fn=logger.info,
                                      rope_real=args.rope_real,
                                      )

    # Create EMA model and convert to fp16 if needed.
    ema = None
    if args.use_ema:
        ema = EMA(args, model, device, logger)

    # Setup gradient checkpointing
    if args.gradient_checkpointing:
        model.enable_gradient_checkpointing()
        controlnet.enable_gradient_checkpointing()

    # Setup FP16 main model:
    if args.use_fp16:
        model = Float16Module(model, args)
        controlnet = Float16Module(controlnet, args)
    logger.info(f"    Using main model with data type {'fp16' if args.use_fp16 else 'fp32'}")

    diffusion = create_diffusion(
        noise_schedule=args.noise_schedule,
        predict_type=args.predict_type,
        learn_sigma=args.learn_sigma,
        mse_loss_weight_type=args.mse_loss_weight_type,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_offset=args.noise_offset,
    )

    # Setup VAE
    logger.info(f"    Loading vae from {VAE_EMA_PATH}")
    vae = AutoencoderKL.from_pretrained(VAE_EMA_PATH)
    # Setup BERT text encoder
    logger.info(f"    Loading Bert text encoder from {TEXT_ENCODER}")
    text_encoder = BertModel.from_pretrained(TEXT_ENCODER, False, revision=None)
    # Setup BERT tokenizer:
    logger.info(f"    Loading Bert tokenizer from {TOKENIZER}")
    tokenizer = BertTokenizer.from_pretrained(TOKENIZER)
    # Setup T5 text encoder
    from hydit.modules.text_encoder import MT5Embedder
    mt5_path = T5_ENCODER['MT5']
    embedder_t5 = MT5Embedder(mt5_path, torch_dtype=T5_ENCODER['torch_dtype'], max_length=args.text_len_t5)
    tokenizer_t5 = embedder_t5.tokenizer
    text_encoder_t5 = embedder_t5.model

    if args.extra_fp16:
        logger.info(f"    Using fp16 for extra modules: vae, text_encoder")
        vae = vae.half().to(device)
        text_encoder = text_encoder.half().to(device)
        text_encoder_t5 = text_encoder_t5.half().to(device)
    else:
        vae = vae.to(device)
        text_encoder = text_encoder.to(device)
        text_encoder_t5 = text_encoder_t5.to(device)

    logger.info(f"    Optimizer parameters: lr={args.lr}, weight_decay={args.weight_decay}")
    logger.info("    Using deepspeed optimizer")
    opt = None

    # ===========================================================================
    # Building Dataset
    # ===========================================================================

    logger.info(f"Building Streaming Dataset.")
    logger.info(f"    Loading index file {args.index_file} (v2)")

    dataset = TextImageArrowStream(args=args,
                                   resolution=image_size[0],
                                   random_flip=args.random_flip,
                                   log_fn=logger.info,
                                   index_file=args.index_file,
                                   multireso=args.multireso,
                                   batch_size=batch_size,
                                   world_size=world_size,
                                   random_shrink_size_cond=args.random_shrink_size_cond,
                                   merge_src_cond=args.merge_src_cond,
                                   uncond_p=args.uncond_p,
                                   text_ctx_len=args.text_len,
                                   tokenizer=tokenizer,
                                   uncond_p_t5=args.uncond_p_t5,
                                   text_ctx_len_t5=args.text_len_t5,
                                   tokenizer_t5=tokenizer_t5,
                                   )
    if args.multireso:
        sampler = BlockDistributedSampler(dataset, num_replicas=world_size, rank=rank, seed=args.global_seed,
                                          shuffle=False, drop_last=True, batch_size=batch_size)
    else:
        sampler = DistributedSamplerWithStartIndex(dataset, num_replicas=world_size, rank=rank, seed=args.global_seed,
                                                   shuffle=False, drop_last=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=sampler,
                        num_workers=args.num_workers, pin_memory=True, drop_last=True)
    logger.info(f"    Dataset contains {len(dataset):,} images.")
    logger.info(f"    Index file: {args.index_file}.")
    if args.multireso:
        logger.info(f'    Using MultiResolutionBucketIndexV2 with step {dataset.index_manager.step} '
                    f'and base size {dataset.index_manager.base_size}')
        logger.info(f'\n  {dataset.index_manager.resolutions}')

    # ===========================================================================
    # Loading parameter
    # ===========================================================================

    logger.info(f"Loading parameter")
    start_epoch = 0
    start_epoch_step = 0
    train_steps = 0
    # Resume checkpoint if needed
    if args.resume:
        model, ema, start_epoch, start_epoch_step, train_steps = model_resume(args, model, ema, logger, len(loader))

    if args.training_parts == "lora":
        loraconfig = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            target_modules=args.target_modules
        )
        if args.use_fp16:
            model.module = get_peft_model(model.module, loraconfig)
        else:
            model = get_peft_model(model, loraconfig)
        
    logger.info(f"    Training parts: {args.training_parts}")

    if args.use_fp16:
        controlnet.module.from_dit(model.module)
        controlnet.module.set_trainable()
    else:
        controlnet.from_dit(model)
        controlnet.set_trainable()
    logger.info(f"    ControlNet loaded from DIT")

    
    
    controlnet, opt, scheduler = deepspeed_initialize(args, logger, controlnet, opt, deepspeed_config)

    # ===========================================================================
    # Training
    # ===========================================================================

    model.eval()
    model.requires_grad_(False)
    model = model.to(device)
    
    if args.use_ema:
        ema.eval()

    print(f"    Worker {rank} ready.")
    dist.barrier()

    iters_per_epoch = len(loader)
    logger.info(" ****************************** Running training ******************************")
    logger.info(f"      Number GPUs:               {world_size}")
    logger.info(f"      Number training samples:   {len(dataset):,}")
    logger.info(f"      Number parameters:         {sum(p.numel() for p in controlnet.parameters()):,}")
    logger.info(f"      Number trainable params:   {sum(p.numel() for p in get_trainable_params(controlnet)):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Iters per epoch:           {iters_per_epoch:,}")
    logger.info(f"      Batch size per device:     {batch_size}")
    logger.info(f"      Batch size all device:     {batch_size * world_size * grad_accu_steps:,} (world_size * batch_size * grad_accu_steps)")
    logger.info(f"      Gradient Accu steps:       {args.grad_accu_steps}")
    logger.info(f"      Total optimization steps:  {args.epochs * iters_per_epoch // grad_accu_steps:,}")

    logger.info(f"      Training epochs:           {start_epoch}/{args.epochs}")
    logger.info(f"      Training epoch steps:      {start_epoch_step:,}/{iters_per_epoch:,}")
    logger.info(f"      Training total steps:      {train_steps:,}/{min(args.max_training_steps, args.epochs * iters_per_epoch):,}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Noise schedule:            {args.noise_schedule}")
    logger.info(f"      Beta limits:               ({args.beta_start}, {args.beta_end})")
    logger.info(f"      Learn sigma:               {args.learn_sigma}")
    logger.info(f"      Prediction type:           {args.predict_type}")
    logger.info(f"      Noise offset:              {args.noise_offset}")

    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Using EMA model:           {args.use_ema} ({args.ema_dtype})")
    if args.use_ema:
        logger.info(f"      Using EMA decay:           {ema.max_value if args.use_ema else None}")
        logger.info(f"      Using EMA warmup power:    {ema.power if args.use_ema else None}")
    logger.info(f"      Using main model fp16:     {args.use_fp16}")
    logger.info(f"      Using extra modules fp16:  {args.extra_fp16}")
    logger.info("    ------------------------------------------------------------------------------")
    logger.info(f"      Experiment directory:      {experiment_dir}")
    logger.info("    *******************************************************************************")

    if args.gc_interval > 0:
        gc.disable()
        gc.collect()

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    start_time = time.time()

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"    Start random shuffle with seed={seed}")
        # Makesure all processors use the same seed to shuffle dataset.
        dataset.shuffle(seed=args.global_seed + epoch, fast=True)
        logger.info(f"    End of random shuffle")

        # Move sampler to start_index
        if not args.multireso:
            start_index = start_epoch_step * world_size * batch_size
            if start_index != sampler.start_index:
                sampler.start_index = start_index
                # Reset start_epoch_step to zero, to ensure next epoch will start from the beginning.
                start_epoch_step = 0
                logger.info(f"      Iters left this epoch: {len(loader):,}")

        logger.info(f"    Beginning epoch {epoch}...")
        step = 0
        for batch in loader:
            step += 1

            latents, model_kwargs = prepare_model_inputs(args, batch, device, vae, text_encoder, text_encoder_t5, freqs_cis_img)

            loss_dict = diffusion.training_losses(model=model, x_start=latents, model_kwargs=model_kwargs, controlnet=controlnet)
            loss = loss_dict["loss"].mean()
            controlnet.backward(loss)
            last_batch_iteration = (train_steps + 1) // (global_batch_size // (batch_size * world_size))
            controlnet.step(lr_kwargs={'last_batch_iteration': last_batch_iteration})

            if args.use_ema:
                if args.use_fp16:
                    ema.update(model.module.module, step=step)
                else:
                    ema.update(model.module, step=step)

            # ===========================================================================
            # Log loss values:
            # ===========================================================================
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                end_time = time.time()
                steps_per_sec = log_steps / (end_time - start_time)
                # Reduce loss history over all processes:
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / world_size
                # get lr from deepspeed fused optimizer
                logger.info(f"(step={train_steps:07d}) " +
                            (f"(update_step={train_steps // args.grad_accu_steps:07d}) " if args.grad_accu_steps > 1 else "") +
                            f"Train Loss: {avg_loss:.4f}, "
                            f"Lr: {opt.param_groups[0]['lr']:.6g}, "
                            f"Steps/Sec: {steps_per_sec:.2f}, "
                            f"Samples/Sec: {int(steps_per_sec * batch_size * world_size):d}")
                # Reset monitoring variables:
                running_loss = 0
                log_steps = 0
                start_time = time.time()

            # collect gc:
            if args.gc_interval > 0 and (step % args.gc_interval == 0):
                gc.collect()

            if (train_steps % args.ckpt_every == 0 or train_steps % args.ckpt_latest_every == 0  # or train_steps == args.max_training_steps
                ) and train_steps > 0:
                save_checkpoint(args, rank, logger, controlnet, ema, epoch, train_steps, checkpoint_dir)

            if train_steps >= args.max_training_steps:
                logger.info(f"Breaking step loop at train_steps={train_steps}.")
                break

        if train_steps >= args.max_training_steps:
            logger.info(f"Breaking epoch loop at epoch={epoch}.")
            break

    dist.destroy_process_group()


if __name__ == "__main__":
    # Start
    main(get_args())
