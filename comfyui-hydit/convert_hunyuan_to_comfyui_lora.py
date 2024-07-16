import argparse
from safetensors import safe_open
from safetensors.torch import save_file

parser = argparse.ArgumentParser(description='Process lora paths.')

parser.add_argument('--lora_path', type=str, default="../models/loras/adapter_model.safetensors",
                    help='The input path of the LoRa weights trained using the official code.')
parser.add_argument('--save_lora_path', type=str, default="../../models/loras/adapter_model_convert.safetensors",
                    help='The path of the converted LoRa weights, used for inference with ComfyUI.')

args = parser.parse_args()

lora_state_dict = {}
with safe_open(args.lora_path, framework="pt", device=0) as f:
    for k in f.keys():
        new_key = "lora_unet_" + "_".join(k[17:].split("."))
        lora_state_dict[new_key] = f.get_tensor(k)  # remove 'basemodel.model'

save_file(lora_state_dict, args.save_lora_path)