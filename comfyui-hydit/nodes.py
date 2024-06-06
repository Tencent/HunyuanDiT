import copy
import os
import torch
from .utils import convert_images_to_tensors
from comfy.model_management import get_torch_device
import folder_paths
from .hydit.diffusion.pipeline import StableDiffusionPipeline
from .hydit.config import get_args
from .hydit.inference import End2End
from pathlib import Path
from .hydit.constants import SAMPLER_FACTORY
from diffusers import schedulers
from .constant import HUNYUAN_PATH, SCHEDULERS_hunyuan
from .dit import load_dit

class DiffusersPipelineLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
        
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "pipeline_folder_name": (os.listdir(HUNYUAN_PATH), ),
                             "model_name": (["disable"] + folder_paths.get_filename_list("checkpoints"), ),
                             "vae_name": (["disable"] + folder_paths.get_filename_list("vae"), ),                             
                              "backend": (["ksampler", "diffusers"], ),  }}

    RETURN_TYPES = ("PIPELINE", "MODEL", "CLIP", "VAE")

    FUNCTION = "create_pipeline"

    CATEGORY = "Diffusers"

    def create_pipeline(self, pipeline_folder_name, model_name, vae_name, backend):
        if model_name != "disable":
            MODEL_PATH = folder_paths.get_full_path("checkpoints", model_name)
        else:
            MODEL_PATH = None
        if vae_name != "disable":
            VAE_PATH = folder_paths.get_full_path("vae", vae_name)
        else:
            VAE_PATH = None
  
        if backend == "diffusers":
            args_hunyuan = get_args()
            gen = End2End(args_hunyuan[0], Path(os.path.join(HUNYUAN_PATH, pipeline_folder_name)), MODEL_PATH, VAE_PATH)
            return (gen, None, None, None)
        elif backend == "ksampler":
            out = load_dit(model_path = os.path.join(HUNYUAN_PATH, pipeline_folder_name), MODEL_PATH = MODEL_PATH, VAE_PATH = VAE_PATH)
            return (None,) + out[:3]

class DiffusersSchedulerLoader:
    def __init__(self):
        self.tmp_dir = folder_paths.get_temp_directory()
        self.dtype = torch.float32
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "scheduler_name": (list(SCHEDULERS_hunyuan), ), 
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
                "pipeline": ("PIPELINE", ), 
                "scheduler": ("SCHEDULER", ),
            }, 
        }

    RETURN_TYPES = ("MAKED_PIPELINE",)

    FUNCTION = "makeup_pipeline"

    CATEGORY = "Diffusers"

    def makeup_pipeline(self, pipeline, scheduler):
        progress_bar_config = {}

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

    RETURN_TYPES = ("STRINGC", "STRINGC", )
    RETURN_NAMES = ("positive", "negative", )

    FUNCTION = "concat_embeds"

    CATEGORY = "Diffusers"

    def concat_embeds(self, positive, negative):

        return (positive, negative, )


class DiffusersSampler:
    def __init__(self):
        self.torch_device = get_torch_device()
        
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "maked_pipeline": ("MAKED_PIPELINE", ),
            "positive": ("STRINGC",),
            "negative": ("STRINGC",),
            "batch_size": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
            "width": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            "height": ("INT", {"default": 1024, "min": 1, "max": 8192, "step": 1}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
            "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**32-1}),
        }}

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "sample"

    CATEGORY = "Diffusers"

    def sample(self, maked_pipeline, positive, negative, batch_size, width, height, steps, cfg, seed):
  
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
                            )
        images = results['images']
        return (convert_images_to_tensors(images),)




NODE_CLASS_MAPPINGS = {
    "DiffusersPipelineLoader": DiffusersPipelineLoader,
    "DiffusersSchedulerLoader": DiffusersSchedulerLoader,
    "DiffusersModelMakeup": DiffusersModelMakeup,
    "DiffusersClipTextEncode": DiffusersClipTextEncode,
    "DiffusersSampler": DiffusersSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffusersPipelineLoader": "HunYuan Pipeline Loader",
    "DiffusersSchedulerLoader": "HunYuan Scheduler Loader",
    "DiffusersModelMakeup": "HunYuan Model Makeup",
    "DiffusersClipTextEncode": "HunYuan Clip Text Encode",
    "DiffusersSampler": "HunYuan Sampler",
}
