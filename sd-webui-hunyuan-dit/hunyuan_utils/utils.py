from modules import devices, rng, shared
import numpy as np
import gc
import inspect
import torch
from typing import Any, Dict, List, Optional, Union, Tuple
from diffusers.schedulers import (
    DDIMScheduler,
    DDPMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
)

hunyuan_transformer_config_v12 = {
  "_class_name": "HunyuanDiT2DModel",
  "_diffusers_version": "0.30.0.dev0",
  "activation_fn": "gelu-approximate",
  "attention_head_dim": 88,
  "cross_attention_dim": 1024,
  "cross_attention_dim_t5": 2048,
  "hidden_size": 1408,
  "in_channels": 4,
  "learn_sigma": True,
  "mlp_ratio": 4.3637,
  "norm_type": "layer_norm",
  "num_attention_heads": 16,
  "num_layers": 40,
  "patch_size": 2,
  "pooled_projection_dim": 1024,
  "sample_size": 128,
  "text_len": 77,
  "text_len_t5": 256,
  "use_style_cond_and_image_meta_size": False
}

dit_sampler_dict = {
    "Euler a":EulerAncestralDiscreteScheduler(),
    "Euler":EulerDiscreteScheduler(),
    "LMS":LMSDiscreteScheduler(),
 	"Heun":HeunDiscreteScheduler(),
    "DPM2":KDPM2DiscreteScheduler(),
    "DPM2 a":KDPM2AncestralDiscreteScheduler(),
    "DPM++ SDE":DPMSolverSinglestepScheduler(),
 	"DPM++ 2M":DPMSolverMultistepScheduler(),
    "DPM++ 2S a":DPMSolverSinglestepScheduler(),
    "LMS Karras":LMSDiscreteScheduler(use_karras_sigmas=True),
    "DPM2 Karras":KDPM2DiscreteScheduler(use_karras_sigmas=True),
    "DPM2 a Karras":KDPM2AncestralDiscreteScheduler(use_karras_sigmas=True),
    "DPM++ SDE Karras":DPMSolverSinglestepScheduler(use_karras_sigmas=True),
    "DPM++ 2M Karras":DPMSolverMultistepScheduler(use_karras_sigmas=True),
    "DPM++ 2S a Karras":DPMSolverSinglestepScheduler(use_karras_sigmas=True),
    "DDIM":DDIMScheduler(),
    "UniPC":UniPCMultistepScheduler(),
    "DPM++ 2M SDE Karras":DPMSolverMultistepScheduler(use_karras_sigmas=True,algorithm_type="sde-dpmsolver++"),
    "DPM++ 2M SDE":DPMSolverMultistepScheduler(algorithm_type="sde-dpmsolver++"),
    "LCM":LCMScheduler()
}

def get_resize_crop_region_for_grid(src, tgt_size):
    th = tw = tgt_size
    h, w = src

    r = h / w

    # resize
    if r > 1:
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.rescale_noise_cfg
def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

def unload_model(current_model):
    if current_model is not None:
        current_model.to(devices.cpu)
        current_model = None
        gc.collect()
        devices.torch_gc()
    return current_model

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
def prepare_extra_step_kwargs(scheduler, generator, eta):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs

def randn_tensor(
    shape: Union[Tuple, List],
    generator: Optional[Union[List["torch.Generator"], "torch.Generator"]] = None,
    device: Optional["torch.device"] = None,
    dtype: Optional["torch.dtype"] = None,
    layout: Optional["torch.layout"] = None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
def prepare_latents_txt2img(vae_scale_factor, scheduler, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    if latents is None:
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
    else:
        latents = latents.to(device)

    # scale the initial noise by the standard deviation required by the scheduler
    if hasattr(scheduler, 'init_noise_sigma'):
        latents = latents * scheduler.init_noise_sigma
    return latents

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps

# Copied from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl_img2img.StableDiffusionXLImg2ImgPipeline.get_timesteps
def get_timesteps(scheduler, num_inference_steps, strength, device, denoising_start=None):
    # get the original timestep using init_timestep
    if denoising_start is None:
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
        t_start = max(num_inference_steps - init_timestep, 0)
    else:
        t_start = 0

    timesteps = scheduler.timesteps[t_start * scheduler.order :]

    # Strength is irrelevant if we directly request a timestep to start at;
    # that is, strength is determined by the denoising_start instead.
    if denoising_start is not None:
        discrete_timestep_cutoff = int(
            round(
                scheduler.config.num_train_timesteps
                - (denoising_start * scheduler.config.num_train_timesteps)
            )
        )

        num_inference_steps = (timesteps < discrete_timestep_cutoff).sum().item()
        if scheduler.order == 2 and num_inference_steps % 2 == 0:
            # if the scheduler is a 2nd order scheduler we might have to do +1
            # because `num_inference_steps` might be even given that every timestep
            # (except the highest one) is duplicated. If `num_inference_steps` is even it would
            # mean that we cut the timesteps in the middle of the denoising step
            # (between 1st and 2nd derivative) which leads to incorrect results. By adding 1
            # we ensure that the denoising process always ends after the 2nd derivate step of the scheduler
            num_inference_steps = num_inference_steps + 1

        # because t_n+1 >= t_n, we slice the timesteps starting from the end
        timesteps = timesteps[-num_inference_steps:]
        return timesteps, num_inference_steps

    return timesteps, num_inference_steps - t_start

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

def _encode_vae_image(image: torch.Tensor, generator: torch.Generator):
    #dtype = image.dtype
    #image = image.float()
    #self.vae_model.to(dtype=torch.float32)

    if isinstance(generator, list):
        image_latents = [
            retrieve_latents(shared.vae_model.encode(image[i : i + 1]), generator=generator[i])
            for i in range(image.shape[0])
        ]
        image_latents = torch.cat(image_latents, dim=0)
    else:
        image_latents = retrieve_latents(shared.vae_model.encode(image), generator=generator)

    #self.vae_model.to(dtype)

    #image_latents = image_latents.to(dtype)
    image_latents = shared.vae_model.config.scaling_factor * image_latents

    return image_latents

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
def prepare_latents_img2img(vae_scale_factor, scheduler, image, batch_size, num_channels_latents, height, width, dtype, device, generator, seeds, timestep):
    shape = (
        batch_size,
        num_channels_latents,
        int(height) // vae_scale_factor,
        int(width) // vae_scale_factor,
    )
    generators = [rng.create_generator(seed) for seed in seeds]
    if isinstance(generator, list) and len(generator) != batch_size:
        raise ValueError(
            f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
            f" size of {batch_size}. Make sure the batch size matches the length of the generators."
        )

    image_latents = _encode_vae_image(image, generator=generators)
    image_latents = image_latents.repeat(batch_size // image_latents.shape[0], 1, 1, 1)
    
    noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype).to(devices.device)
    init_latents = scheduler.add_noise(image_latents, noise, timestep)
    latents = init_latents.to(device=devices.device, dtype=dtype)

    return latents, noise, image_latents

def guess_dit_model(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if "mlp_t5.0.weight" in state_dict:
        return "hunyuan-original"
    elif "text_embedder.linear_1.weight" in state_dict:
        return "hunyuan"
    else:
        return "non supported dit"

def convert_hunyuan_to_diffusers(state_dict):
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # input_size -> sample_size, text_dim -> cross_attention_dim
    num_layers = 40
    for i in range(num_layers):
        # attn1
        # Wkqv -> to_q, to_k, to_v
        q, k, v = torch.chunk(state_dict[f"blocks.{i}.attn1.Wqkv.weight"], 3, dim=0)
        q_bias, k_bias, v_bias = torch.chunk(state_dict[f"blocks.{i}.attn1.Wqkv.bias"], 3, dim=0)
        state_dict[f"blocks.{i}.attn1.to_q.weight"] = q
        state_dict[f"blocks.{i}.attn1.to_q.bias"] = q_bias
        state_dict[f"blocks.{i}.attn1.to_k.weight"] = k
        state_dict[f"blocks.{i}.attn1.to_k.bias"] = k_bias
        state_dict[f"blocks.{i}.attn1.to_v.weight"] = v
        state_dict[f"blocks.{i}.attn1.to_v.bias"] = v_bias
        state_dict.pop(f"blocks.{i}.attn1.Wqkv.weight")
        state_dict.pop(f"blocks.{i}.attn1.Wqkv.bias")

        # q_norm, k_norm -> norm_q, norm_k
        state_dict[f"blocks.{i}.attn1.norm_q.weight"] = state_dict[f"blocks.{i}.attn1.q_norm.weight"]
        state_dict[f"blocks.{i}.attn1.norm_q.bias"] = state_dict[f"blocks.{i}.attn1.q_norm.bias"]
        state_dict[f"blocks.{i}.attn1.norm_k.weight"] = state_dict[f"blocks.{i}.attn1.k_norm.weight"]
        state_dict[f"blocks.{i}.attn1.norm_k.bias"] = state_dict[f"blocks.{i}.attn1.k_norm.bias"]

        state_dict.pop(f"blocks.{i}.attn1.q_norm.weight")
        state_dict.pop(f"blocks.{i}.attn1.q_norm.bias")
        state_dict.pop(f"blocks.{i}.attn1.k_norm.weight")
        state_dict.pop(f"blocks.{i}.attn1.k_norm.bias")

        # out_proj -> to_out
        state_dict[f"blocks.{i}.attn1.to_out.0.weight"] = state_dict[f"blocks.{i}.attn1.out_proj.weight"]
        state_dict[f"blocks.{i}.attn1.to_out.0.bias"] = state_dict[f"blocks.{i}.attn1.out_proj.bias"]
        state_dict.pop(f"blocks.{i}.attn1.out_proj.weight")
        state_dict.pop(f"blocks.{i}.attn1.out_proj.bias")

        # attn2
        # kq_proj -> to_k, to_v
        k, v = torch.chunk(state_dict[f"blocks.{i}.attn2.kv_proj.weight"], 2, dim=0)
        k_bias, v_bias = torch.chunk(state_dict[f"blocks.{i}.attn2.kv_proj.bias"], 2, dim=0)
        state_dict[f"blocks.{i}.attn2.to_k.weight"] = k
        state_dict[f"blocks.{i}.attn2.to_k.bias"] = k_bias
        state_dict[f"blocks.{i}.attn2.to_v.weight"] = v
        state_dict[f"blocks.{i}.attn2.to_v.bias"] = v_bias
        state_dict.pop(f"blocks.{i}.attn2.kv_proj.weight")
        state_dict.pop(f"blocks.{i}.attn2.kv_proj.bias")

        # q_proj -> to_q
        state_dict[f"blocks.{i}.attn2.to_q.weight"] = state_dict[f"blocks.{i}.attn2.q_proj.weight"]
        state_dict[f"blocks.{i}.attn2.to_q.bias"] = state_dict[f"blocks.{i}.attn2.q_proj.bias"]
        state_dict.pop(f"blocks.{i}.attn2.q_proj.weight")
        state_dict.pop(f"blocks.{i}.attn2.q_proj.bias")

        # q_norm, k_norm -> norm_q, norm_k
        state_dict[f"blocks.{i}.attn2.norm_q.weight"] = state_dict[f"blocks.{i}.attn2.q_norm.weight"]
        state_dict[f"blocks.{i}.attn2.norm_q.bias"] = state_dict[f"blocks.{i}.attn2.q_norm.bias"]
        state_dict[f"blocks.{i}.attn2.norm_k.weight"] = state_dict[f"blocks.{i}.attn2.k_norm.weight"]
        state_dict[f"blocks.{i}.attn2.norm_k.bias"] = state_dict[f"blocks.{i}.attn2.k_norm.bias"]

        state_dict.pop(f"blocks.{i}.attn2.q_norm.weight")
        state_dict.pop(f"blocks.{i}.attn2.q_norm.bias")
        state_dict.pop(f"blocks.{i}.attn2.k_norm.weight")
        state_dict.pop(f"blocks.{i}.attn2.k_norm.bias")

        # out_proj -> to_out
        state_dict[f"blocks.{i}.attn2.to_out.0.weight"] = state_dict[f"blocks.{i}.attn2.out_proj.weight"]
        state_dict[f"blocks.{i}.attn2.to_out.0.bias"] = state_dict[f"blocks.{i}.attn2.out_proj.bias"]
        state_dict.pop(f"blocks.{i}.attn2.out_proj.weight")
        state_dict.pop(f"blocks.{i}.attn2.out_proj.bias")

        # switch norm 2 and norm 3
        norm2_weight = state_dict[f"blocks.{i}.norm2.weight"]
        norm2_bias = state_dict[f"blocks.{i}.norm2.bias"]
        state_dict[f"blocks.{i}.norm2.weight"] = state_dict[f"blocks.{i}.norm3.weight"]
        state_dict[f"blocks.{i}.norm2.bias"] = state_dict[f"blocks.{i}.norm3.bias"]
        state_dict[f"blocks.{i}.norm3.weight"] = norm2_weight
        state_dict[f"blocks.{i}.norm3.bias"] = norm2_bias

        # norm1 -> norm1.norm
        # default_modulation.1 -> norm1.linear
        state_dict[f"blocks.{i}.norm1.norm.weight"] = state_dict[f"blocks.{i}.norm1.weight"]
        state_dict[f"blocks.{i}.norm1.norm.bias"] = state_dict[f"blocks.{i}.norm1.bias"]
        state_dict[f"blocks.{i}.norm1.linear.weight"] = state_dict[f"blocks.{i}.default_modulation.1.weight"]
        state_dict[f"blocks.{i}.norm1.linear.bias"] = state_dict[f"blocks.{i}.default_modulation.1.bias"]
        state_dict.pop(f"blocks.{i}.norm1.weight")
        state_dict.pop(f"blocks.{i}.norm1.bias")
        state_dict.pop(f"blocks.{i}.default_modulation.1.weight")
        state_dict.pop(f"blocks.{i}.default_modulation.1.bias")

        # mlp.fc1 -> ff.net.0, mlp.fc2 -> ff.net.2
        state_dict[f"blocks.{i}.ff.net.0.proj.weight"] = state_dict[f"blocks.{i}.mlp.fc1.weight"]
        state_dict[f"blocks.{i}.ff.net.0.proj.bias"] = state_dict[f"blocks.{i}.mlp.fc1.bias"]
        state_dict[f"blocks.{i}.ff.net.2.weight"] = state_dict[f"blocks.{i}.mlp.fc2.weight"]
        state_dict[f"blocks.{i}.ff.net.2.bias"] = state_dict[f"blocks.{i}.mlp.fc2.bias"]
        state_dict.pop(f"blocks.{i}.mlp.fc1.weight")
        state_dict.pop(f"blocks.{i}.mlp.fc1.bias")
        state_dict.pop(f"blocks.{i}.mlp.fc2.weight")
        state_dict.pop(f"blocks.{i}.mlp.fc2.bias")

    # pooler -> time_extra_emb
    state_dict["time_extra_emb.pooler.positional_embedding"] = state_dict["pooler.positional_embedding"]
    state_dict["time_extra_emb.pooler.k_proj.weight"] = state_dict["pooler.k_proj.weight"]
    state_dict["time_extra_emb.pooler.k_proj.bias"] = state_dict["pooler.k_proj.bias"]
    state_dict["time_extra_emb.pooler.q_proj.weight"] = state_dict["pooler.q_proj.weight"]
    state_dict["time_extra_emb.pooler.q_proj.bias"] = state_dict["pooler.q_proj.bias"]
    state_dict["time_extra_emb.pooler.v_proj.weight"] = state_dict["pooler.v_proj.weight"]
    state_dict["time_extra_emb.pooler.v_proj.bias"] = state_dict["pooler.v_proj.bias"]
    state_dict["time_extra_emb.pooler.c_proj.weight"] = state_dict["pooler.c_proj.weight"]
    state_dict["time_extra_emb.pooler.c_proj.bias"] = state_dict["pooler.c_proj.bias"]
    state_dict.pop("pooler.k_proj.weight")
    state_dict.pop("pooler.k_proj.bias")
    state_dict.pop("pooler.q_proj.weight")
    state_dict.pop("pooler.q_proj.bias")
    state_dict.pop("pooler.v_proj.weight")
    state_dict.pop("pooler.v_proj.bias")
    state_dict.pop("pooler.c_proj.weight")
    state_dict.pop("pooler.c_proj.bias")
    state_dict.pop("pooler.positional_embedding")

    # t_embedder -> time_embedding (`TimestepEmbedding`)
    state_dict["time_extra_emb.timestep_embedder.linear_1.bias"] = state_dict["t_embedder.mlp.0.bias"]
    state_dict["time_extra_emb.timestep_embedder.linear_1.weight"] = state_dict["t_embedder.mlp.0.weight"]
    state_dict["time_extra_emb.timestep_embedder.linear_2.bias"] = state_dict["t_embedder.mlp.2.bias"]
    state_dict["time_extra_emb.timestep_embedder.linear_2.weight"] = state_dict["t_embedder.mlp.2.weight"]

    state_dict.pop("t_embedder.mlp.0.bias")
    state_dict.pop("t_embedder.mlp.0.weight")
    state_dict.pop("t_embedder.mlp.2.bias")
    state_dict.pop("t_embedder.mlp.2.weight")

    # x_embedder -> pos_embd (`PatchEmbed`)
    state_dict["pos_embed.proj.weight"] = state_dict["x_embedder.proj.weight"]
    state_dict["pos_embed.proj.bias"] = state_dict["x_embedder.proj.bias"]
    state_dict.pop("x_embedder.proj.weight")
    state_dict.pop("x_embedder.proj.bias")

    # mlp_t5 -> text_embedder
    state_dict["text_embedder.linear_1.bias"] = state_dict["mlp_t5.0.bias"]
    state_dict["text_embedder.linear_1.weight"] = state_dict["mlp_t5.0.weight"]
    state_dict["text_embedder.linear_2.bias"] = state_dict["mlp_t5.2.bias"]
    state_dict["text_embedder.linear_2.weight"] = state_dict["mlp_t5.2.weight"]
    state_dict.pop("mlp_t5.0.bias")
    state_dict.pop("mlp_t5.0.weight")
    state_dict.pop("mlp_t5.2.bias")
    state_dict.pop("mlp_t5.2.weight")

    # extra_embedder -> extra_embedder
    state_dict["time_extra_emb.extra_embedder.linear_1.bias"] = state_dict["extra_embedder.0.bias"]
    state_dict["time_extra_emb.extra_embedder.linear_1.weight"] = state_dict["extra_embedder.0.weight"]
    state_dict["time_extra_emb.extra_embedder.linear_2.bias"] = state_dict["extra_embedder.2.bias"]
    state_dict["time_extra_emb.extra_embedder.linear_2.weight"] = state_dict["extra_embedder.2.weight"]
    state_dict.pop("extra_embedder.0.bias")
    state_dict.pop("extra_embedder.0.weight")
    state_dict.pop("extra_embedder.2.bias")
    state_dict.pop("extra_embedder.2.weight")

    # model.final_adaLN_modulation.1 -> norm_out.linear
    def swap_scale_shift(weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    state_dict["norm_out.linear.weight"] = swap_scale_shift(state_dict["final_layer.adaLN_modulation.1.weight"])
    state_dict["norm_out.linear.bias"] = swap_scale_shift(state_dict["final_layer.adaLN_modulation.1.bias"])
    state_dict.pop("final_layer.adaLN_modulation.1.weight")
    state_dict.pop("final_layer.adaLN_modulation.1.bias")

    # final_linear -> proj_out
    state_dict["proj_out.weight"] = state_dict["final_layer.linear.weight"]
    state_dict["proj_out.bias"] = state_dict["final_layer.linear.bias"]
    state_dict.pop("final_layer.linear.weight")
    state_dict.pop("final_layer.linear.bias")
    return state_dict