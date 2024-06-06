import argparse

from .constants import *
from .modules.models import HUNYUAN_DIT_CONFIG


def get_args(default_args=None):
    parser = argparse.ArgumentParser()

    # Basic
    parser.add_argument("--prompt", type=str, default="一只小猫", help="The prompt for generating images.")
    parser.add_argument("--model-root", type=str, default="ckpts", help="Model root path.")
    parser.add_argument("--image-size", type=int, nargs='+', default=[1024, 1024],
                        help='Image size (h, w). If a single value is provided, the image will be treated to '
                             '(value, value).')
    parser.add_argument("--infer-mode", type=str, choices=["fa", "torch", "trt"], default="torch",
                        help="Inference mode")

    # HunYuan-DiT
    parser.add_argument("--model", type=str, choices=list(HUNYUAN_DIT_CONFIG.keys()), default='DiT-g/2')
    parser.add_argument("--norm", type=str, default="layer", help="Normalization layer type")
    parser.add_argument("--load-key", type=str, choices=["ema", "module", "distill"], default="ema", help="Load model key for HunYuanDiT checkpoint.")
    parser.add_argument('--size-cond', type=int, nargs='+', default=[1024, 1024],
                        help="Size condition used in sampling. 2 values are required for height and width. "
                             "If a single value is provided, the image will be treated to (value, value).")
    parser.add_argument("--cfg-scale", type=float, default=6.0, help="Guidance scale for classifier-free.")

    # Prompt enhancement
    parser.add_argument("--enhance", action="store_true", help="Enhance prompt with dialoggen.")
    parser.add_argument("--no-enhance", dest="enhance", action="store_false")
    parser.add_argument("--load-4bit", help="load DialogGen model with 4bit quantization.", action="store_true")
    parser.set_defaults(enhance=True)

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

    # Text condition
    parser.add_argument("--text-states-dim", type=int, default=1024, help="Hidden size of CLIP text encoder.")
    parser.add_argument("--text-len", type=int, default=77, help="Token length of CLIP text encoder output.")
    parser.add_argument("--text-states-dim-t5", type=int, default=2048, help="Hidden size of CLIP text encoder.")
    parser.add_argument("--text-len-t5", type=int, default=256, help="Token length of T5 text encoder output.")
    parser.add_argument("--negative", type=str, default=None, help="Negative prompt.")

    # Acceleration
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision.")
    parser.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    parser.set_defaults(use_fp16=True)
    parser.add_argument("--onnx-workdir", type=str, default="onnx_model", help="Path to save ONNX model")

    # Sampling
    parser.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size")
    parser.add_argument("--sampler", type=str, choices=SAMPLER_FACTORY, default="ddpm", help="Diffusion sampler")
    parser.add_argument("--infer-steps", type=int, default=100, help="Inference steps")
    parser.add_argument('--seed', type=int, default=42, help="A seed for all the prompts.")

    # App
    parser.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Language")

    args = parser.parse_args(default_args)

    return args
