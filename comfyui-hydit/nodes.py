import copy
import os
import torch
from .utils import convert_images_to_tensors
from comfy.model_management import get_torch_device
import folder_paths
from .hydit_v1_1.diffusion.pipeline import StableDiffusionPipeline
from .hydit_v1_1.diffusion.pipeline_controlnet import StableDiffusionControlNetPipeline
from .hydit_v1_1.config_comfyui import get_args
from .hydit_v1_1.inference_comfyui import End2End
from pathlib import Path
from .hydit_v1_1.constants import SAMPLER_FACTORY
from diffusers import schedulers
from .constant import HUNYUAN_PATH, SCHEDULERS_hunyuan, T5_PATH, LORA_PATH
from .dit import  load_checkpoint, load_vae
from .clip import CLIP
from .hydit_v1_1.modules.controlnet import HunYuanControlNet
from .hydit_v1_1.modules.models import HUNYUAN_DIT_CONFIG
from loguru import logger
import numpy as np
from torchvision import transforms as T

norm_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)

from PIL import Image


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




class DiffusersPipelineLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipeline_folder_name": (os.listdir(HUNYUAN_PATH),), },
                "optional": {"lora": ("lora_path",), }}

    RETURN_TYPES = ("PIPELINE",)

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, pipeline_folder_name, lora=None):
        if lora != None:
            LORA_PATH = lora
        else:
            LORA_PATH = None

        args_hunyuan = get_args()
        gen = End2End(args_hunyuan[0], Path(os.path.join(HUNYUAN_PATH, pipeline_folder_name)), LOAR_PATH=LORA_PATH)
        return (gen,)


class DiffusersCheckpointLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("checkpoints"),), 
            "version": (list(["v1.1", "v1.2"]),), }}

    RETURN_TYPES = ("MODEL",)

    FUNCTION = "load_checkpoint"

    CATEGORY = "Diffusers"

    def load_checkpoint(self, model_name, version):
        MODEL_PATH = folder_paths.get_full_path("checkpoints", model_name)
        out = load_checkpoint(MODEL_PATH, version)
        return out


class DiffusersVAELoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "model_name": (folder_paths.get_filename_list("vae"),), }}

    RETURN_TYPES = ("VAE",)

    FUNCTION = "load_vae"

    CATEGORY = "Diffusers"

    def load_vae(self, model_name):
        MODEL_PATH = folder_paths.get_full_path("vae", model_name)
        out = load_vae(MODEL_PATH)
        return out


class DiffusersLoraLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"lora_name": (os.listdir(LORA_PATH),), }}

    RETURN_TYPES = ("lora_path",)

    FUNCTION = "load_lora"

    CATEGORY = "Diffusers"

    def load_lora(self, lora_name):
        MODEL_PATH = os.path.join(LORA_PATH, lora_name)
        return (MODEL_PATH,)


class DiffusersCLIPLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text_encoder_path": (folder_paths.get_filename_list("clip"),),
            "t5_text_encoder_path": (os.listdir(T5_PATH),), }}

    RETURN_TYPES = ("CLIP",)

    FUNCTION = "load_clip"

    CATEGORY = "Diffusers"

    def load_clip(self, text_encoder_path, t5_text_encoder_path):
        CLIP_PATH = folder_paths.get_full_path("clip", text_encoder_path)
        t5_file = os.path.join(T5_PATH, t5_text_encoder_path)
        root = None
        out = CLIP(False, root, CLIP_PATH, t5_file)
        return (out,)


class DiffusersControlNetLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "controlnet_path": (folder_paths.get_filename_list("controlnet"),), }}

    RETURN_TYPES = ("CTRL",)

    FUNCTION = "load_controlnet"

    CATEGORY = "Diffusers"

    def load_controlnet(self, controlnet_path):
        DiffusersControlNetLoader_PATH = folder_paths.get_full_path("controlnet", controlnet_path)
        args_hunyuan = get_args()
        args = args_hunyuan[0]
        image_size = _to_tuple(args.image_size)
        latent_size = (image_size[0] // 8, image_size[1] // 8)
        model_config = HUNYUAN_DIT_CONFIG[args.model]
        controlnet = HunYuanControlNet(args,
                                       input_size=latent_size,
                                       **model_config,
                                       log_fn=logger.info,
                                       ).half().to(self.torch_device)
        controlnet_state_dict = torch.load(DiffusersControlNetLoader_PATH)
        controlnet.load_state_dict(controlnet_state_dict)
        controlnet.eval()
        return (controlnet,)


class DiffusersSchedulerLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler_name": (list(SCHEDULERS_hunyuan),),
            }
        }

    RETURN_TYPES = ("SCHEDULER",)

    FUNCTION = "load_scheduler"

    CATEGORY = "Diffusers"

    def load_scheduler(self, scheduler_name):
        # Load sampler from factory
        kwargs = SAMPLER_FACTORY[scheduler_name]['kwargs']
        scheduler = SAMPLER_FACTORY[scheduler_name]['scheduler']
        args_hunyuan = get_args()
        args_hunyuan = args_hunyuan[0]

        # Update sampler according to the arguments
        kwargs['beta_schedule'] = args_hunyuan.noise_schedule
        kwargs['beta_start'] = args_hunyuan.beta_start
        kwargs['beta_end'] = args_hunyuan.beta_end
        kwargs['prediction_type'] = args_hunyuan.predict_type

        # Build scheduler according to the sampler.
        scheduler_class = getattr(schedulers, scheduler)
        scheduler = scheduler_class(**kwargs)

        return (scheduler,)


class DiffusersModelMakeup:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("PIPELINE",),
                "scheduler": ("SCHEDULER",),
            },
            "optional":
                {"controlnet": ("CTRL",), }
        }

    RETURN_TYPES = ("MAKED_PIPELINE",)

    FUNCTION = "makeup_pipeline"

    CATEGORY = "Diffusers"

    def makeup_pipeline(self, pipeline, scheduler, controlnet=None):
        progress_bar_config = {}
        if not controlnet:
            pipe = StableDiffusionPipeline(vae=pipeline.vae,
                                           text_encoder=pipeline.clip_text_encoder,
                                           tokenizer=pipeline.tokenizer,
                                           unet=pipeline.model,
                                           scheduler=scheduler,
                                           feature_extractor=None,
                                           safety_checker=None,
                                           requires_safety_checker=False,
                                           progress_bar_config=progress_bar_config,
                                           embedder_t5=pipeline.embedder_t5,
                                           infer_mode=pipeline.infer_mode,
                                           )
        else:
            pipe = StableDiffusionControlNetPipeline(vae=pipeline.vae,
                                                     text_encoder=pipeline.clip_text_encoder,
                                                     tokenizer=pipeline.tokenizer,
                                                     unet=pipeline.model,
                                                     scheduler=scheduler,
                                                     feature_extractor=None,
                                                     safety_checker=None,
                                                     requires_safety_checker=False,
                                                     progress_bar_config=progress_bar_config,
                                                     embedder_t5=pipeline.embedder_t5,
                                                     infer_mode=pipeline.infer_mode,
                                                     controlnet=controlnet
                                                     )
        pipe = pipe.to(pipeline.device)
        pipeline.pipeline = pipe
        return (pipeline,)


class DiffusersClipTextEncode:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "positive": ("STRING", {"multiline": True}),
            "negative": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("STRINGC", "STRINGC",)
    RETURN_NAMES = ("positive", "negative",)

    FUNCTION = "concat_embeds"

    CATEGORY = "Diffusers"

    def concat_embeds(self, positive, negative):
        return (positive, negative,)


class DiffusersSampler:
    def __init__(self):
        self.torch_device = get_torch_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "maked_pipeline": ("MAKED_PIPELINE",),
            "positive": ("STRINGC",),
            "negative": ("STRINGC",),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
            "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "control_weight": ("FLOAT", {"default": 0.7, "min": 0, "max": 1.0}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2 ** 32 - 1}),
        },
            "optional": {"image": ("IMAGE",), }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers"

    def sample(self, maked_pipeline, positive, negative, batch_size, width, height, steps, cfg, seed, control_weight,
               image=None):
        if not isinstance(image, torch.Tensor):
            results = maked_pipeline.predict(positive,
                                             height=height,
                                             width=width,
                                             seed=int(seed),
                                             enhanced_prompt=None,
                                             negative_prompt=negative,
                                             infer_steps=steps,
                                             guidance_scale=cfg,
                                             batch_size=batch_size,
                                             src_size_cond=[height, width],
                                             image=image
                                             )
        else:
            # import pdb
            # pdb.set_trace()
            img = 255. * image.cpu().numpy()
            # print(i.size())
            img = Image.fromarray(np.clip(img[0], 0, 255).astype(np.uint8))
            img = img.convert('RGB').resize((height, width))
            image = norm_transform(img)
            image = image.unsqueeze(0).cuda()
            # print(image.size)
            results = maked_pipeline.predict(positive,
                                             height=height,
                                             width=width,
                                             seed=int(seed),
                                             enhanced_prompt=None,
                                             negative_prompt=negative,
                                             infer_steps=steps,
                                             guidance_scale=cfg,
                                             batch_size=batch_size,
                                             src_size_cond=[height, width],
                                             image=image,
                                             control_weight=control_weight,
                                             )

        images = results['images']
        return (convert_images_to_tensors(images),)


NODE_CLASS_MAPPINGS = {
    "DiffusersPipelineLoader": DiffusersPipelineLoader,
    "DiffusersSchedulerLoader": DiffusersSchedulerLoader,
    "DiffusersModelMakeup": DiffusersModelMakeup,
    "DiffusersClipTextEncode": DiffusersClipTextEncode,
    "DiffusersSampler": DiffusersSampler,
    "DiffusersCheckpointLoader": DiffusersCheckpointLoader,
    "DiffusersVAELoader": DiffusersVAELoader,
    "DiffusersCLIPLoader": DiffusersCLIPLoader,
    "DiffusersControlNetLoader": DiffusersControlNetLoader,
    "DiffusersLoraLoader": DiffusersLoraLoader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersPipelineLoader": "HunYuan Pipeline Loader",
    "DiffusersSchedulerLoader": "HunYuan Scheduler Loader",
    "DiffusersModelMakeup": "HunYuan Model Makeup",
    "DiffusersClipTextEncode": "HunYuan Clip Text Encode",
    "DiffusersSampler": "HunYuan Sampler",
    "DiffusersCheckpointLoader": "HunYuan Checkpoint Loader",
    "DiffusersVAELoader": "HunYuan VAE Loader",
    "DiffusersCLIPLoader": "HunYuan CLIP Loader",
    "DiffusersControlNetLoader": "HunYuan ControlNet Loader",
    "DiffusersLoraLoader": "HunYuan Lora Loader",
}
