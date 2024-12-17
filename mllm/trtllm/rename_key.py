from safetensors.torch import load_file, safe_open
from safetensors.torch import save_file
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--huggingface_repo_dir",
        type=str,
    )
    parser.add_argument(
        "--thirdparty_repo_dir",
        type=str,
    )
    parser.add_argument(
        "--merged_repo_dir",
        type=str,
    )
    return parser.parse_args()


args = parse_arguments()

import shutil

shutil.copytree(args.huggingface_repo_dir, args.merged_repo_dir)

import torch

hf_weights_dict = dict()
hf_wgt_names = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]
for wgt in hf_wgt_names:
    ori_weights = load_file(args.huggingface_repo_dir + wgt)
    for key, value in ori_weights.items():
        if key == "language_model.lm_head.weight":
            hf_weights_dict[key] = value
        elif key == "language_model.model.embed_tokens.weight":
            hf_weights_dict[key] = value

weights = [
    "model-00001-of-00004.safetensors",
    "model-00002-of-00004.safetensors",
    "model-00003-of-00004.safetensors",
    "model-00004-of-00004.safetensors",
]
for wgt in weights:
    ori_weights = load_file(args.thirdparty_repo_dir + wgt)
    # import pdb;pdb.set_trace()
    new_weights = dict()
    for key, value in ori_weights.items():
        if key == "lm_head.weight":
            new_key = "language_model.lm_head.weight"
        elif key == "model.embed_tokens.weight":
            new_key = "language_model.model.embed_tokens.weight"
        elif key == "model.image_newline":
            new_key = "image_newline"
        elif "model.layers." in key:
            new_key = key.replace("model", "language_model.model")
        elif key == "model.norm.weight":
            new_key = "language_model.model.norm.weight"
        elif key == "model.mm_projector.0.bias":
            new_key = "multi_modal_projector.linear_1.bias"
        elif key == "model.mm_projector.0.weight":
            new_key = "multi_modal_projector.linear_1.weight"
        elif key == "model.mm_projector.2.bias":
            new_key = "multi_modal_projector.linear_2.bias"
        elif key == "model.mm_projector.2.weight":
            new_key = "multi_modal_projector.linear_2.weight"
        elif "model.vision_tower.vision_tower" in key:
            new_key = key.replace("model.vision_tower.vision_tower", "vision_tower")

        if new_key == "language_model.lm_head.weight":
            value = torch.cat(
                (value, hf_weights_dict["language_model.lm_head.weight"][32000:]), dim=0
            )

        elif new_key == "language_model.model.embed_tokens.weight":
            value = torch.cat(
                (
                    value,
                    hf_weights_dict["language_model.model.embed_tokens.weight"][32000:],
                ),
                dim=0,
            )

        new_weights[new_key] = value
    save_file(new_weights, args.merged_repo_dir + wgt, metadata={"format": "pt"})
