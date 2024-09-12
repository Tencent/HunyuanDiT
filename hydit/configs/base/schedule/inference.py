# Basic Setting
prompt = '一只小猫'           # The prompt for generating images.
model_root = 'ckpts'        # Root path of all the models, including t2i model and dialoggen model.
dit_weight = None           # Path to the HunYuan_DiT model. If None, search the model in the args.model_root. 1. If it is a file, load the model directly. In this case, the __load_key is ignored. 2. If it is a directory, search the model in the directory. Support two types of models: 1) named `pytorch_model_*.pt`, where * is specified by the __load_key. 2) named `*_model_states.pt`, where * can be `mp_rank_00`. *_model_states.pt contains both 'module' and 'ema' weights. Therefore, you still use __load_key to specify the weights to load. By default, load 'ema' weights.
controlnet_weight = None    # Path to the HunYuan_DiT controlnet model. If None, search the model in the args.model_root. 1. If it is a directory, search the model in the directory. 2. If it is a file, load the model directly. In this case, the __load_key is ignored. 

# Model setting
load_key = 'ema'        # Load model key for HunYuanDiT checkpoint, choices=["ema", "module", "distill", 'merge']
use_style_cond = False  # Use style condition in hydit. Only for hydit version <= 1.1"
size_cond = None        # Size condition used in sampling. 2 values are required for height and width. If a single value is provided, the image will be treated to (value, value). Recommended values are [1024, 1024]. Only for hydit version <= 1.1
target_ratios = None    # Target ratios for multi_resolution training.
cfg_scale = 6.0         # Guidance scale for classifier_free.
negative = None         # Negative prompt.

# Acceleration
infer_mode = 'fa'       # Inference mode, choices=["fa", "torch", "trt"], default="fa"
onnx_workdir = 'onnx_model' # Path to save ONNX model

# Sampling
sampler = 'ddpm'        # Diffusion sampler, choices=SAMPLER_FACTORY
infer_steps = 100       # Inference steps

# Prompt enhancement
enhance = True          # Enhance prompt with mllm.
load_4bit = False       # Load DialogGen model with 4bit quantization.

# App
lang='zh'               # Language, choices=["zh", "en"]