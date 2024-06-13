import argparse

from .constants import *
from .modules.models import HUNYUAN_DIT_CONFIG


def get_args(default_args=None):
    parser_hunyuan = argparse.ArgumentParser()
    #print(parser_hunyuan)

    # Basic
    parser_hunyuan.add_argument("--prompt", type=str, default="现实主义风格，画面主要描述一个巴洛克风格的花瓶，带有金色的装饰边框，花瓶上盛开着各种色彩鲜艳的花，白色背景", help="The prompt for generating images.")
    parser_hunyuan.add_argument("--model-root", type=str, default="ckpts", help="Model root path.")
    parser_hunyuan.add_argument("--image-size", type=int, nargs='+', default=[1024, 1024],
                        help='Image size (h, w). If a single value is provided, the image will be treated to '
                             '(value, value).')
    parser_hunyuan.add_argument("--infer-mode", type=str, choices=["fa", "torch", "trt"], default="torch",
                        help="Inference mode")

    # HunYuan-DiT
    parser_hunyuan.add_argument("--model", type=str, choices=list(HUNYUAN_DIT_CONFIG.keys()), default='DiT-g/2')
    parser_hunyuan.add_argument("--norm", type=str, default="layer", help="Normalization layer type")
    parser_hunyuan.add_argument("--load-key", type=str, choices=["ema", "module", "distill"], default="ema", help="Load model key for HunYuanDiT checkpoint.")
    parser_hunyuan.add_argument('--size-cond', type=int, nargs='+', default=[1024, 1024],
                        help="Size condition used in sampling. 2 values are required for height and width. "
                             "If a single value is provided, the image will be treated to (value, value).")
    parser_hunyuan.add_argument("--cfg-scale", type=float, default=6.0, help="Guidance scale for classifier-free.")

    # Prompt enhancement
    parser_hunyuan.add_argument("--enhance", action="store_true", help="Enhance prompt with dialoggen.")
    parser_hunyuan.add_argument("--no-enhance", dest="enhance", action="store_false")
    parser_hunyuan.add_argument("--load-4bit", help="load DialogGen model with 4bit quantization.", action="store_true")
    parser_hunyuan.set_defaults(enhance=True)

    # Diffusion
    parser_hunyuan.add_argument("--learn-sigma", action="store_true", help="Learn extra channels for sigma.")
    parser_hunyuan.add_argument("--no-learn-sigma", dest="learn_sigma", action="store_false")
    parser_hunyuan.set_defaults(learn_sigma=True)
    parser_hunyuan.add_argument("--predict-type", type=str, choices=list(PREDICT_TYPE), default="v_prediction",
                        help="Diffusion predict type")
    parser_hunyuan.add_argument("--noise-schedule", type=str, choices=list(NOISE_SCHEDULES), default="scaled_linear",
                        help="Noise schedule")
    parser_hunyuan.add_argument("--beta-start", type=float, default=0.00085, help="Beta start value")
    parser_hunyuan.add_argument("--beta-end", type=float, default=0.03, help="Beta end value")

    # Text condition
    parser_hunyuan.add_argument("--text-states-dim", type=int, default=1024, help="Hidden size of CLIP text encoder.")
    parser_hunyuan.add_argument("--text-len", type=int, default=77, help="Token length of CLIP text encoder output.")
    parser_hunyuan.add_argument("--text-states-dim-t5", type=int, default=2048, help="Hidden size of CLIP text encoder.")
    parser_hunyuan.add_argument("--text-len-t5", type=int, default=256, help="Token length of T5 text encoder output.")
    parser_hunyuan.add_argument("--negative", type=str, default="错误的眼睛，糟糕的人脸，毁容，糟糕的艺术，变形，多余的肢体，模糊的颜色，模糊，重复，病态，残缺，", help="Negative prompt.")

    # Acceleration
    parser_hunyuan.add_argument("--use-fp16", action="store_true", help="Use FP16 precision.")
    parser_hunyuan.add_argument("--no-fp16", dest="use_fp16", action="store_false")
    parser_hunyuan.set_defaults(use_fp16=True)

    # Sampling
    parser_hunyuan.add_argument("--batch-size", type=int, default=1, help="Per-GPU batch size")
    parser_hunyuan.add_argument("--sampler", type=str, choices=SAMPLER_FACTORY, default="ddim", help="Diffusion sampler")
    parser_hunyuan.add_argument("--infer-steps", type=int, default=30, help="Inference steps")
    parser_hunyuan.add_argument('--seed', type=int, default=666, help="A seed for all the prompts.")

    # App
    parser_hunyuan.add_argument("--lang", type=str, default="zh", choices=["zh", "en"], help="Language")

    args = parser_hunyuan.parse_known_args()

    return args
