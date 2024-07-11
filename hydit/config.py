import os
import argparse
import importlib.util

from .constants import *
from .modules.models import HUNYUAN_DIT_CONFIG, HUNYUAN_DIT_MODELS
from .diffusion.gaussian_diffusion import ModelVarType

import deepspeed

def model_var_type(value):
    try:
        return ModelVarType[value]
    except KeyError:
        raise ValueError(f"Invalid choice '{value}', valid choices are {[v.name for v in ModelVarType]}")

def load_config(training_type):
    config_path = "./hydit/config/train_config.py" if training_type == 'full' \
        else f"./hydit/config/train_{training_type}_config.py"
    config_path = os.path.normpath(config_path)
    spec = importlib.util.spec_from_file_location("train_config", config_path)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return config.train_config


def merge_args_with_config(args, config, parser):
    def is_default_or_none_value(parser, key, value):
        for action in parser._actions:
            if action.dest == key:
                return value is None or action.default == value
        return False

    for key, value in config.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if is_default_or_none_value(parser, sub_key, getattr(args, key, None)):
                    setattr(args, sub_key, sub_value)
        else:
            if is_default_or_none_value(parser, key, getattr(args, key, None)):
                setattr(args, key, value)
    return args


def get_args(default_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-flag", type=str)

    # General Setting
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument('--seed', type=int, default=42, help="A seed for all the prompts.")
    parser.add_argument("--use-fp16", action="store_true", help="Use FP16 precision.")
    parser.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    parser.set_defaults(use_fp16=True)
    parser.add_argument("--extra-fp16", action="store_true", help="Use extra fp16 for vae and text_encoder.")

    # HunYuan-DiT
    parser.add_argument("--model", type=str, choices=list(HUNYUAN_DIT_CONFIG.keys()), default='DiT-g/2')
    parser.add_argument("--image-size", type=int, nargs='+', default=[1024, 1024],
                        help='Image size (h, w). If a single value is provided, the image will be treated to '
                             '(value, value).')
    parser.add_argument("--qk-norm", action="store_true", help="Query Key normalization. See http://arxiv.org/abs/2302.05442 for details.")
    parser.set_defaults(qk_norm=True)
    parser.add_argument("--norm", type=str, choices=["rms", "layer"], default="layer", help="Normalization layer type")
    parser.add_argument("--text-states-dim", type=int, default=1024, help="Hidden size of CLIP text encoder.")
    parser.add_argument("--text-len", type=int, default=77, help="Token length of CLIP text encoder output.")
    parser.add_argument("--text-states-dim-t5", type=int, default=2048, help="Hidden size of CLIP text encoder.")
    parser.add_argument("--text-len-t5", type=int, default=256, help="Token length of T5 text encoder output.")

    # LoRA config
    parser.add_argument("--training-parts", type=str, default='all', choices=['all', 'lora'], help="Training parts")
    parser.add_argument("--rank", type=int, default=64, help="Rank of LoRA")
    parser.add_argument("--lora-ckpt", type=str, default=None, help="LoRA checkpoint")
    parser.add_argument('--target-modules', type=str, nargs='+', default=['Wqkv', 'q_proj', 'kv_proj', 'out_proj'],
                        help="Target modules for LoRA fine tune")
    parser.add_argument("--output-merge-path", type=str, default=None, help="Output path for merged model")

    # controlnet config
    parser.add_argument("--control-type", type=str, default='canny', choices=['canny', 'depth', 'pose'], help="Controlnet condition type")
    parser.add_argument("--control-weight", type=str, default=1.0, help="Controlnet weight, You can use a float to specify the weight for all layers, or use a list to separately specify the weight for each layer, for example, '[1.0 * (0.825 ** float(19 - i)) for i in range(19)]'")
    parser.add_argument("--condition-image-path", type=str, default=None, help="Inference condition image path")

    # Diffusion
    parser.add_argument("--learn-sigma", action="store_true", help="Learn extra channels for sigma.")
    parser.add_argument("--no-learn-sigma", dest="learn_sigma", action="store_false")
    parser.set_defaults(learn_sigma=True)
    parser.add_argument("--predict-type", type=str, choices=list(PREDICT_TYPE), default="v_prediction",
                        help="Diffusion predict type")
    parser.add_argument("--noise-schedule", type=str, choices=list(NOISE_SCHEDULES), default="scaled_linear",
                        help="Noise schedule")
    parser.add_argument("--beta-start", type=float, default=0.00085, help="Beta start value")
    parser.add_argument("--beta-end", type=float, default=0.03, help="Beta end value")
    parser.add_argument("--sigma-small", action="store_true")
    parser.add_argument("--mse-loss-weight-type", type=str, default="constant",
                        help="Min-SNR-gamma. Can be constant or min_snr_<gamma> where gamma is a integer. 5 is recommended in the paper.")
    parser.add_argument("--model-var-type", type=model_var_type, default=None, help="Specify the model variable type.")
    parser.add_argument("--noise-offset", type=float, default=0.0, help="Add extra noise to the input image.")

    # ========================================================================================================
    # Inference
    # ========================================================================================================

    # Basic Setting
    parser.add_argument("--prompt", type=str, default="一只小猫", help="The prompt for generating images.")
    parser.add_argument("--model-root", type=str, default="ckpts", help="Model root path.")

    # Model setting
    parser.add_argument("--load-key", type=str, choices=["ema", "module", "distill", 'merge'], default="ema", help="Load model key for HunYuanDiT checkpoint.")
    parser.add_argument('--size-cond', type=int, nargs='+', default=[1024, 1024],
                        help="Size condition used in sampling. 2 values are required for height and width. "
                             "If a single value is provided, the image will be treated to (value, value).")
    parser.add_argument('--target-ratios', type=str, nargs='+', default=None,
                        help="Target ratios for multi-resolution training.")
    parser.add_argument("--cfg-scale", type=float, default=6.0, help="Guidance scale for classifier-free.")
    parser.add_argument("--negative", type=str, default=None, help="Negative prompt.")

    # Acceleration
    parser.add_argument("--infer-mode", type=str, choices=["fa", "torch", "trt"], default="torch", help="Inference mode")
    parser.add_argument("--onnx-workdir", type=str, default="onnx_model", help="Path to save ONNX model")

    # Sampling
    parser.add_argument("--sampler", type=str, choices=SAMPLER_FACTORY, default="ddpm", help="Diffusion sampler")
    parser.add_argument("--infer-steps", type=int, default=100, help="Inference steps")

    # Prompt enhancement
    parser.add_argument("--enhance", action="store_true", help="Enhance prompt with mllm.")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false")
    parser.add_argument("--load-4bit", help="load DialogGen model with 4bit quantization.", action="store_true")
    parser.set_defaults(enhance=True)

    # App
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Language")


    # ========================================================================================================
    # Training
    # ========================================================================================================

    # Basic Setting
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--max-training-steps", type=int, default=10_000_000)
    parser.add_argument("--gc-interval", type=int, default=40,
                        help='To address the memory bottleneck encountered during the preprocessing of the dataset,'
                             ' memory fragments are reclaimed here by invoking the gc.collect() function.')
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=100_000, help="Create a ckpt every a few steps.")
    parser.add_argument("--ckpt-latest-every", type=int, default=10_000, help="Create a ckpt named `latest.pt` every a few steps.")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--global-seed", type=int, default=1234)
    parser.add_argument("--warmup-min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-num-steps", type=float, default=0)
    parser.add_argument("--weight-decay", type=float, default=0, help="weight-decay in optimizer")
    parser.add_argument("--rope-img", type=str, default=None, choices=['extend', 'base512', 'base1024'],
                        help="Extend or interpolate the positional embedding of the image.")
    parser.add_argument("--rope-real", action="store_true",
                        help="Use real part and imaginary part separately for RoPE.")

    # Classifier-free
    parser.add_argument("--uncond-p", type=float, default=0.2,
                        help="The probability of dropping training text used for CLIP feature extraction")
    parser.add_argument("--uncond-p-t5", type=float, default=0.2,
                        help="The probability of dropping training text used for mT5 feature extraction")

    # Directory
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--resume", type=str, default=None, help="Resume experiment from a checkpoint")
    parser.add_argument("--no-strict", dest="strict", action="store_false", help="Strict loading of checkpoint")
    parser.set_defaults(strict=True)
    parser.add_argument("--resume-deepspeed", action="store_true",
                        help="Resume model and ema states from a checkpoint saved by Deepspeed version of DIT.")
    parser.add_argument("--resume-split", action="store_true",
                        help="Resume model and ema states from two checkpoint separated from DeepSpeed ckpt.")
    parser.add_argument("--ema-to-module", action="store_true",
                        help="If true, initialize the module with EMA weights.")
    parser.add_argument("--module-to-ema", action="store_true",
                        help="if true, initialize the ema with Module weights.")

    # Dataset
    parser.add_argument("--index-file", type=str, nargs='+', help="During training, provide a JSON file with data indices.")
    parser.add_argument("--random-flip", action="store_true", help="Random flip image")
    parser.add_argument("--reset-loader", action="store_true",
                        help="Reset the data loader. It is useful when resuming from a checkpoint but switch to a new dataset.")
    parser.add_argument("--multireso", action="store_true", help="Use multi-resolution training.")
    parser.add_argument("--reso-step", type=int, default=None, help="Step size for multi-resolution training.")

    # Additional condition
    parser.add_argument("--random-shrink-size-cond", action="store_true",
                        help="Randomly shrink the original size condition.")
    parser.add_argument("--merge-src-cond", action="store_true", help="Merge the source condition into a single value.")

    # EMA Model
    parser.add_argument("--use-ema", action="store_true", help="Use EMA model")
    parser.add_argument("--ema-dtype", type=str, choices=['fp16', 'fp32', 'none'], default="none",
                        help="EMA data type. If none, use the same data type as the model.")
    parser.add_argument("--ema-decay", type=float, default=None,
                        help="EMA decay rate. If None, use the default value of the model.")
    parser.add_argument("--ema-warmup", action="store_true",
                        help="EMA warmup. If True, perform ema_decay warmup from 0 to ema_decay.")
    parser.add_argument("--ema-warmup-power", type=float, default=None,
                        help="EMA power. If None, use the default value of the model.")
    parser.add_argument("--ema-reset-decay", action="store_true",
                        help="Reset EMA decay to 0 and restart increasing the EMA decay."
                             "Only works when --ema-warmup is enabled.")
    # Acceleration
    parser.add_argument("--use-flash-attn", action="store_true", help="During training, "
                                                                      "flash attention is used to accelerate training.")
    parser.add_argument("--no-flash-attn", dest="use_flash_attn",
                        action="store_false", help="During training, flash attention is not used to accelerate training.")
    parser.add_argument("--use-zero-stage", type=int, default=1, help="Use AngelPTM zero stage. Support 2 and 3")
    parser.add_argument("--grad-accu-steps", type=int, default=1, help="Gradient accumulation steps.")

    # ========================================================================================================
    # Deepspeed config
    # ========================================================================================================
    parser = deepspeed.add_config_arguments(parser)
    parser.add_argument('--local_rank', type=int, default=None,
                        help='local rank passed from distributed launcher.')
    parser.add_argument('--deepspeed-optimizer', action='store_true',
                        help='Switching to the optimizers in DeepSpeed')
    parser.add_argument('--remote-device', type=str, default='none', choices=['none', 'cpu', 'nvme'],
                        help='Remote device for ZeRO-3 initialized parameters.')
    parser.add_argument('--zero-stage', type=int, default=1)
    parser.add_argument("--async-ema", action="store_true", help="Whether to use multi stream to excut EMA.")

    parser.add_argument("--training-type", type=str, required=True, default='full', choices=['full', 'controlnet', 'lora'],
                        help="Specify the type of training")
    args = parser.parse_args(default_args)

    config = load_config(args.training_type)
    args = merge_args_with_config(args, config, parser)

    return args
