import numpy as np
import torch

from k_diffusion.external import DiscreteVDDPMDenoiser
from k_diffusion.sampling import sample_euler_ancestral, get_sigmas_exponential

from PIL import Image
from pytorch_lightning import seed_everything

from library.hunyuan_models import *
from library.hunyuan_utils import *

# from lycoris.kohya import create_network_from_weights
from networks.lora import create_network_from_weights

from copy import deepcopy

PROMPT = "青花瓷风格, 在白色背景上，一只小狗在追逐蝴蝶"
NEG_PROMPT = ""
CLIP_TOKENS = 75 * 2 + 2
ATTN_MODE = "xformers"
H = 1024
W = 1024
STEPS = 100
CFG_SCALE = 6
DEVICE = "cuda"
DTYPE = torch.float16

VERSION = "1.2"

if VERSION == "1.1":
    BETA_END = 0.03
    USE_EXTRA_COND = True
    MODEL_PATH = "/root/albertxyu/HunYuanDiT-V1.1-fp16-pruned"
elif VERSION == "1.2":
    BETA_END = 0.018
    USE_EXTRA_COND = False
    MODEL_PATH = "/root/albertxyu/HunYuanDiT-V1.2-fp16-pruned"
else:
    raise ValueError(f"Invalid version: {VERSION}")

LORA_WEIGHT = "./test_lora_unet_clip_v1.2_0630/last-step00000400.ckpt"

# Global variables to store model components
_loaded_model = None
_loaded_model_path = None
_loaded_ckpt_path = None


def load_scheduler_sigmas(beta_start=0.00085, beta_end=0.018, num_train_timesteps=1000):
    betas = (
        torch.linspace(
            beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=torch.float32
        )
        ** 2
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    sigmas = np.array(((1 - alphas_cumprod) / alphas_cumprod) ** 0.5)
    sigmas = np.concatenate([sigmas[::-1], [0.0]]).astype(np.float32)
    sigmas = torch.from_numpy(sigmas)
    return alphas_cumprod, sigmas


def load_model_if_needed(model_path, ckpt_path):
    global _loaded_model, _loaded_model_path, _loaded_ckpt_path
    if (
        _loaded_model is None
        or _loaded_model_path != model_path
        or _loaded_ckpt_path != ckpt_path
    ):
        (
            denoiser,
            patch_size,
            head_dim,
            clip_tokenizer,
            clip_encoder,
            mt5_embedder,
            vae,
        ) = load_model(
            model_path, dtype=DTYPE, device=DEVICE, use_extra_cond=USE_EXTRA_COND
        )
        denoiser.eval()
        denoiser.disable_fp32_silu()
        denoiser.disable_fp32_layer_norm()
        denoiser.set_attn_mode(ATTN_MODE)
        vae.requires_grad_(False)
        mt5_embedder.to(torch.float16)

        lora_net, state_dict = create_network_from_weights(
            multiplier=1.0,
            file=ckpt_path,
            vae=vae,
            text_encoder=[clip_encoder, mt5_embedder],
            unet=denoiser,
        )
        lora_net.apply_to(
            text_encoder=[clip_encoder, mt5_embedder],
            unet=denoiser,
        )
        lora_net.load_state_dict(state_dict)
        lora_net = lora_net.to(DEVICE, dtype=DTYPE)

        _loaded_model = (
            denoiser,
            patch_size,
            head_dim,
            clip_tokenizer,
            clip_encoder,
            mt5_embedder,
            vae,
        )
        _loaded_model_path = model_path
        _loaded_ckpt_path = ckpt_path

    return _loaded_model


def generate_image(
    prompt: str,
    neg_prompt: str,
    seed: int,
    height: int,
    width: int,
    steps: int,
    cfg_scale: int,
    model_path: str,
    ckpt_path: str,
    model_version: str,
):
    seed_everything(seed)
    if model_version == "1.2":
        BETA_END = 0.018
        USE_EXTRA_COND = False
    elif model_version == "1.1":
        BETA_END = 0.03
        USE_EXTRA_COND = True
    else:
        raise ValueError(f"Invalid version: {model_version}")
    PROMPT = prompt
    NEG_PROMPT = neg_prompt
    H = height
    W = width
    STEPS = steps
    CFG_SCALE = cfg_scale
    MODEL_PATH = model_path
    LORA_WEIGHT = ckpt_path

    with torch.inference_mode(True), torch.no_grad():
        alphas, sigmas = load_scheduler_sigmas(beta_end=BETA_END)
        (
            denoiser,
            patch_size,
            head_dim,
            clip_tokenizer,
            clip_encoder,
            mt5_embedder,
            vae,
        ) = load_model_if_needed(MODEL_PATH, LORA_WEIGHT)

        with torch.autocast("cuda"):
            clip_h, clip_m, mt5_h, mt5_m = get_cond(
                PROMPT,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                # Should be same as original implementation with max_length_clip=77
                # Support 75*n + 2
                max_length_clip=CLIP_TOKENS,
            )
            neg_clip_h, neg_clip_m, neg_mt5_h, neg_mt5_m = get_cond(
                NEG_PROMPT,
                mt5_embedder,
                clip_tokenizer,
                clip_encoder,
                max_length_clip=CLIP_TOKENS,
            )
            clip_h = torch.concat([clip_h, neg_clip_h], dim=0)
            clip_m = torch.concat([clip_m, neg_clip_m], dim=0)
            mt5_h = torch.concat([mt5_h, neg_mt5_h], dim=0)
            mt5_m = torch.concat([mt5_m, neg_mt5_m], dim=0)
            torch.cuda.empty_cache()

        style = torch.as_tensor([0] * 2, device=DEVICE)
        # src hw, dst hw, 0, 0
        size_cond = [H, W, H, W, 0, 0]
        image_meta_size = torch.as_tensor([size_cond] * 2, device=DEVICE)
        freqs_cis_img = calc_rope(H, W, patch_size, head_dim)

        denoiser_wrapper = DiscreteVDDPMDenoiser(
            # A quick patch for learn_sigma
            lambda *args, **kwargs: denoiser(*args, **kwargs).chunk(2, dim=1)[0],
            alphas,
            False,
        ).to(DEVICE)

        def cfg_denoise_func(x, sigma):
            cond, uncond = denoiser_wrapper(
                x.repeat(2, 1, 1, 1),
                sigma.repeat(2),
                encoder_hidden_states=clip_h,
                text_embedding_mask=clip_m,
                encoder_hidden_states_t5=mt5_h,
                text_embedding_mask_t5=mt5_m,
                image_meta_size=image_meta_size,
                style=style,
                cos_cis_img=freqs_cis_img[0],
                sin_cis_img=freqs_cis_img[1],
            ).chunk(2, dim=0)
            return uncond + (cond - uncond) * CFG_SCALE

        sigmas = denoiser_wrapper.get_sigmas(STEPS).to(DEVICE)
        sigmas = get_sigmas_exponential(
            STEPS, denoiser_wrapper.sigma_min, denoiser_wrapper.sigma_max, DEVICE
        )
        x1 = torch.randn(1, 4, H // 8, W // 8, dtype=torch.float16, device=DEVICE)

        with torch.autocast("cuda"):
            sample = sample_euler_ancestral(
                cfg_denoise_func,
                x1 * sigmas[0],
                sigmas,
            )
            torch.cuda.empty_cache()
            with torch.no_grad():
                latent = sample / 0.13025
                image = vae.decode(latent).sample
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.permute(0, 2, 3, 1).cpu().numpy()
                image = (image * 255).round().astype(np.uint8)
                image = [Image.fromarray(im) for im in image]
                for im in image:
                    # im.save(f"test_1600_{VERSION}_lora.png")
                    im.save(f"output.png")


if __name__ == "__main__":
    seed_everything(0)
    generate_image(
        PROMPT, NEG_PROMPT, 0, H, W, STEPS, CFG_SCALE, MODEL_PATH, LORA_WEIGHT, VERSION
    )
