import random
import time
from pathlib import Path

import numpy as np
import torch

# For reproducibility
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True

from diffusers import schedulers
from diffusers.models import AutoencoderKL
from loguru import logger
from transformers import BertModel, BertTokenizer
from transformers.modeling_utils import logger as tf_logger

from .constants import SAMPLER_FACTORY, NEGATIVE_PROMPT, TRT_MAX_WIDTH, TRT_MAX_HEIGHT, TRT_MAX_BATCH_SIZE
from .diffusion.pipeline import StableDiffusionPipeline
from .modules.models import HunYuanDiT, HUNYUAN_DIT_CONFIG
from .modules.posemb_layers import get_2d_rotary_pos_embed, get_fill_resize_and_crop
from .modules.text_encoder import MT5Embedder
from .utils.tools import set_seeds


class Resolution:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __str__(self):
        return f'{self.height}x{self.width}'


class ResolutionGroup:
    def __init__(self):
        self.data = [
            Resolution(1024, 1024), # 1:1
            Resolution(1280, 1280), # 1:1
            Resolution(1024, 768),  # 4:3
            Resolution(1152, 864),  # 4:3
            Resolution(1280, 960),  # 4:3
            Resolution(768, 1024),  # 3:4
            Resolution(864, 1152),  # 3:4
            Resolution(960, 1280),  # 3:4
            Resolution(1280, 768),  # 16:9
            Resolution(768, 1280),  # 9:16
        ]
        self.supported_sizes = set([(r.width, r.height) for r in self.data])

    def is_valid(self, width, height):
        return (width, height) in self.supported_sizes


STANDARD_RATIO = np.array([
    1.0,        # 1:1
    4.0 / 3.0,  # 4:3
    3.0 / 4.0,  # 3:4
    16.0 / 9.0, # 16:9
    9.0 / 16.0, # 9:16
])
STANDARD_SHAPE = [
    [(1024, 1024), (1280, 1280)],   # 1:1
    [(1024, 768), (1152, 864), (1280, 960)],    # 4:3
    [(768, 1024), (864, 1152), (960, 1280)],    # 3:4
    [(1280, 768)],                              # 16:9
    [(768, 1280)],                              # 9:16
]
STANDARD_AREA = [
    np.array([w * h for w, h in shapes])
    for shapes in STANDARD_SHAPE
]


def get_standard_shape(target_width, target_height):
    """
    Map image size to standard size.
    """
    target_ratio = target_width / target_height
    closest_ratio_idx = np.argmin(np.abs(STANDARD_RATIO - target_ratio))
    closest_area_idx = np.argmin(np.abs(STANDARD_AREA[closest_ratio_idx] - target_width * target_height))
    width, height = STANDARD_SHAPE[closest_ratio_idx][closest_area_idx]
    return width, height


def _to_tuple(val):
    if isinstance(val, (list, tuple)):
        if len(val) == 1:
            val = [val[0], val[0]]
        elif len(val) == 2:
            val = tuple(val)
        else:
            raise ValueError(f"Invalid value: {val}")
    elif isinstance(val, (int, float)):
        val = (val, val)
    else:
        raise ValueError(f"Invalid value: {val}")
    return val


def get_pipeline(args, vae, text_encoder, tokenizer, model, device, rank,
                 embedder_t5, infer_mode, sampler=None):
    """
    Get scheduler and pipeline for sampling. The sampler and pipeline are both
    based on diffusers and make some modifications.

    Returns
    -------
    pipeline: StableDiffusionPipeline
    sampler_name: str
    """
    sampler = sampler or args.sampler

    # Load sampler from factory
    kwargs = SAMPLER_FACTORY[sampler]['kwargs']
    scheduler = SAMPLER_FACTORY[sampler]['scheduler']

    # Update sampler according to the arguments
    kwargs['beta_schedule'] = args.noise_schedule
    kwargs['beta_start'] = args.beta_start
    kwargs['beta_end'] = args.beta_end
    kwargs['prediction_type'] = args.predict_type

    # Build scheduler according to the sampler.
    scheduler_class = getattr(schedulers, scheduler)
    scheduler = scheduler_class(**kwargs)

    # Set timesteps for inference steps.
    scheduler.set_timesteps(args.infer_steps, device)

    # Only enable progress bar for rank 0
    progress_bar_config = {} if rank == 0 else {'disable': True}

    pipeline = StableDiffusionPipeline(vae=vae,
                                       text_encoder=text_encoder,
                                       tokenizer=tokenizer,
                                       unet=model,
                                       scheduler=scheduler,
                                       feature_extractor=None,
                                       safety_checker=None,
                                       requires_safety_checker=False,
                                       progress_bar_config=progress_bar_config,
                                       embedder_t5=embedder_t5,
                                       infer_mode=infer_mode,
                                       )

    pipeline = pipeline.to(device)

    return pipeline, sampler


class End2End(object):
    def __init__(self, args, models_root_path):
        self.args = args

        # Check arguments
        t2i_root_path = Path(models_root_path) / "t2i"
        self.root = t2i_root_path
        logger.info(f"Got text-to-image model root path: {t2i_root_path}")

        # Set device and disable gradient
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)
        # Disable BertModel logging checkpoint info
        tf_logger.setLevel('ERROR')

        # ========================================================================
        logger.info(f"Loading CLIP Text Encoder...")
        text_encoder_path = self.root / "clip_text_encoder"
        self.clip_text_encoder = BertModel.from_pretrained(str(text_encoder_path), False, revision=None).to(self.device)
        logger.info(f"Loading CLIP Text Encoder finished")

        # ========================================================================
        logger.info(f"Loading CLIP Tokenizer...")
        tokenizer_path = self.root / "tokenizer"
        self.tokenizer = BertTokenizer.from_pretrained(str(tokenizer_path))
        logger.info(f"Loading CLIP Tokenizer finished")

        # ========================================================================
        logger.info(f"Loading T5 Text Encoder and T5 Tokenizer...")
        t5_text_encoder_path = self.root / 'mt5'
        embedder_t5 = MT5Embedder(t5_text_encoder_path, torch_dtype=torch.float16, max_length=256)
        self.embedder_t5 = embedder_t5
        logger.info(f"Loading t5_text_encoder and t5_tokenizer finished")

        # ========================================================================
        logger.info(f"Loading VAE...")
        vae_path = self.root / "sdxl-vae-fp16-fix"
        self.vae = AutoencoderKL.from_pretrained(str(vae_path)).to(self.device)
        logger.info(f"Loading VAE finished")

        # ========================================================================
        # Create model structure and load the checkpoint
        logger.info(f"Building HunYuan-DiT model...")
        model_config = HUNYUAN_DIT_CONFIG[self.args.model]
        self.patch_size = model_config['patch_size']
        self.head_size = model_config['hidden_size'] // model_config['num_heads']
        self.resolutions, self.freqs_cis_img = self.standard_shapes()   # Used for TensorRT models
        self.image_size = _to_tuple(self.args.image_size)
        latent_size = (self.image_size[0] // 8, self.image_size[1] // 8)

        self.infer_mode = self.args.infer_mode
        if self.infer_mode in ['fa', 'torch']:
            model_dir = self.root / "model"
            model_path = model_dir / f"pytorch_model_{self.args.load_key}.pt"
            if not model_path.exists():
                raise ValueError(f"model_path not exists: {model_path}")
            # Build model structure
            self.model = HunYuanDiT(self.args,
                                    input_size=latent_size,
                                    **model_config,
                                    log_fn=logger.info,
                                    ).half().to(self.device)    # Force to use fp16
            # Load model checkpoint
            logger.info(f"Loading torch model {model_path}...")
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            logger.info(f"Loading torch model finished")
        elif self.infer_mode == 'trt':
            from .modules.trt.hcf_model import TRTModel

            trt_dir = self.root / "model_trt"
            engine_dir = trt_dir / "engine"
            plugin_path = trt_dir / "fmha_plugins/9.2_plugin_cuda11/fMHAPlugin.so"
            model_name = "model_onnx"

            logger.info(f"Loading TensorRT model {engine_dir}/{model_name}...")
            self.model = TRTModel(model_name=model_name,
                                  engine_dir=str(engine_dir),
                                  image_height=TRT_MAX_HEIGHT,
                                  image_width=TRT_MAX_WIDTH,
                                  text_maxlen=args.text_len,
                                  embedding_dim=args.text_states_dim,
                                  plugin_path=str(plugin_path),
                                  max_batch_size=TRT_MAX_BATCH_SIZE,
                                  )
            logger.info(f"Loading TensorRT model finished")
        else:
            raise ValueError(f"Unknown infer_mode: {self.infer_mode}")

        # ========================================================================
        # Build inference pipeline. We use a customized StableDiffusionPipeline.
        logger.info(f"Loading inference pipeline...")
        self.pipeline, self.sampler = self.load_sampler()
        logger.info(f'Loading pipeline finished')

        # ========================================================================
        self.default_negative_prompt = NEGATIVE_PROMPT
        logger.info("==================================================")
        logger.info(f"                Model is ready.                  ")
        logger.info("==================================================")

    def load_sampler(self, sampler=None):
        pipeline, sampler = get_pipeline(self.args,
                                         self.vae,
                                         self.clip_text_encoder,
                                         self.tokenizer,
                                         self.model,
                                         device=self.device,
                                         rank=0,
                                         embedder_t5=self.embedder_t5,
                                         infer_mode=self.infer_mode,
                                         sampler=sampler,
                                         )
        return pipeline, sampler

    def calc_rope(self, height, width):
        th = height // 8 // self.patch_size
        tw = width // 8 // self.patch_size
        base_size = 512 // 8 // self.patch_size
        start, stop = get_fill_resize_and_crop((th, tw), base_size)
        sub_args = [start, stop, (th, tw)]
        rope = get_2d_rotary_pos_embed(self.head_size, *sub_args)
        return rope

    def standard_shapes(self):
        resolutions = ResolutionGroup()
        freqs_cis_img = {}
        for reso in resolutions.data:
            freqs_cis_img[str(reso)] = self.calc_rope(reso.height, reso.width)
        return resolutions, freqs_cis_img

    def predict(self,
                user_prompt,
                height=1024,
                width=1024,
                seed=None,
                enhanced_prompt=None,
                negative_prompt=None,
                infer_steps=100,
                guidance_scale=6,
                batch_size=1,
                src_size_cond=(1024, 1024),
                sampler=None,
                ):
        # ========================================================================
        # Arguments: seed
        # ========================================================================
        if seed is None:
            seed = random.randint(0, 1_000_000)
        if not isinstance(seed, int):
            raise TypeError(f"`seed` must be an integer, but got {type(seed)}")
        generator = set_seeds(seed)

        # ========================================================================
        # Arguments: target_width, target_height
        # ========================================================================
        if width <= 0 or height <= 0:
            raise ValueError(f"`height` and `width` must be positive integers, got height={height}, width={width}")
        logger.info(f"Input (height, width) = ({height}, {width})")
        if self.infer_mode in ['fa', 'torch']:
            # We must force height and width to align to 16 and to be an integer.
            target_height = int((height // 16) * 16)
            target_width = int((width // 16) * 16)
            logger.info(f"Align to 16: (height, width) = ({target_height}, {target_width})")
        elif self.infer_mode == 'trt':
            target_width, target_height = get_standard_shape(width, height)
            logger.info(f"Align to standard shape: (height, width) = ({target_height}, {target_width})")
        else:
            raise ValueError(f"Unknown infer_mode: {self.infer_mode}")

        # ========================================================================
        # Arguments: prompt, new_prompt, negative_prompt
        # ========================================================================
        if not isinstance(user_prompt, str):
            raise TypeError(f"`user_prompt` must be a string, but got {type(user_prompt)}")
        user_prompt = user_prompt.strip()
        prompt = user_prompt

        if enhanced_prompt is not None:
            if not isinstance(enhanced_prompt, str):
                raise TypeError(f"`enhanced_prompt` must be a string, but got {type(enhanced_prompt)}")
            enhanced_prompt = enhanced_prompt.strip()
            prompt = enhanced_prompt

        # negative prompt
        if negative_prompt is None or negative_prompt == '':
            negative_prompt = self.default_negative_prompt
        if not isinstance(negative_prompt, str):
            raise TypeError(f"`negative_prompt` must be a string, but got {type(negative_prompt)}")

        # ========================================================================
        # Arguments: style. (A fixed argument. Don't Change it.)
        # ========================================================================
        style = torch.as_tensor([0, 0] * batch_size, device=self.device)

        # ========================================================================
        # Inner arguments: image_meta_size (Please refer to SDXL.)
        # ========================================================================
        if isinstance(src_size_cond, int):
            src_size_cond = [src_size_cond, src_size_cond]
        if not isinstance(src_size_cond, (list, tuple)):
            raise TypeError(f"`src_size_cond` must be a list or tuple, but got {type(src_size_cond)}")
        if len(src_size_cond) != 2:
            raise ValueError(f"`src_size_cond` must be a tuple of 2 integers, but got {len(src_size_cond)}")
        size_cond = list(src_size_cond) + [target_width, target_height, 0, 0]
        image_meta_size = torch.as_tensor([size_cond] * 2 * batch_size, device=self.device)

        # ========================================================================
        start_time = time.time()
        logger.debug(f"""
                       prompt: {user_prompt}
              enhanced prompt: {enhanced_prompt}
                         seed: {seed}
              (height, width): {(target_height, target_width)}
              negative_prompt: {negative_prompt}
                   batch_size: {batch_size}
               guidance_scale: {guidance_scale}
                  infer_steps: {infer_steps}
              image_meta_size: {size_cond}
        """)
        reso = f'{target_height}x{target_width}'
        if reso in self.freqs_cis_img:
            freqs_cis_img = self.freqs_cis_img[reso]
        else:
            freqs_cis_img = self.calc_rope(target_height, target_width)

        if sampler is not None and sampler != self.sampler:
            self.pipeline, self.sampler = self.load_sampler(sampler)

        samples = self.pipeline(
            height=target_height,
            width=target_width,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_images_per_prompt=batch_size,
            guidance_scale=guidance_scale,
            num_inference_steps=infer_steps,
            image_meta_size=image_meta_size,
            style=style,
            return_dict=False,
            generator=generator,
            freqs_cis_img=freqs_cis_img,
            use_fp16=self.args.use_fp16,
            learn_sigma=self.args.learn_sigma,
        )[0]
        gen_time = time.time() - start_time
        logger.debug(f"Success, time: {gen_time}")

        return {
            'images': samples,
            'seed': seed,
        }
