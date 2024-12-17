import argparse
import torch
import sys
import os

# 添加当前命令行运行的目录到 sys.path
sys.path.append(os.getcwd() + "/mllm")


from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IMAGE_PLACEHOLDER,
)
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)

import requests
from PIL import Image
from io import BytesIO
import re


def image_parser(image_file, sep=","):
    out = image_file.split(sep)
    return out


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_file).convert("RGB")
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def init_dialoggen_model(model_path, model_base=None, load_4bit=False):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, llava_type_model=True, load_4bit=load_4bit
    )
    return {"tokenizer": tokenizer, "model": model, "image_processor": image_processor}


def eval_model(
    models,
    query="详细描述一下这张图片",
    image_file=None,
    sep=",",
    temperature=0.2,
    top_p=None,
    num_beams=1,
    max_new_tokens=512,
    return_history=False,
    history=None,
    skip_special=False,
):
    # Model
    disable_torch_init()

    qs = query
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if models["model"].config.mm_use_im_start_end:
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if models["model"].config.mm_use_im_start_end:
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    if not history:
        conv = conv_templates["llava_v1"].copy()
    else:
        conv = history

    if skip_special:
        conv.append_message(conv.roles[0], query)
    else:
        conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    if image_file is not None:
        image_files = image_parser(image_file, sep=sep)
        images = load_images(image_files)
        image_sizes = [x.size for x in images]
        images_tensor = process_images(
            images, models["image_processor"], models["model"].config
        ).to(models["model"].device, dtype=torch.float16)
    else:
        # fomatted input as training data
        image_sizes = [(1024, 1024)]
        images_tensor = torch.zeros(
            1,
            5,
            3,
            models["image_processor"].crop_size["height"],
            models["image_processor"].crop_size["width"],
        )
        images_tensor = images_tensor.to(models["model"].device, dtype=torch.float16)

    input_ids = (
        tokenizer_image_token(
            prompt, models["tokenizer"], IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )
    with torch.inference_mode():
        output_ids = models["model"].generate(
            input_ids,
            images=images_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = (
        models["tokenizer"]
        .batch_decode(output_ids, skip_special_tokens=True)[0]
        .strip()
    )
    if return_history:
        return outputs, conv
    return outputs


def remove_prefix(text):
    if text.startswith("<画图>"):
        return text[len("<画图>") :], True
    elif text.startswith("对不起"):
        # 拒绝画图
        return "", False
    else:
        return text, True


class DialogGen(object):
    def __init__(self, model_path, load_4bit=False):
        self.models = init_dialoggen_model(model_path, load_4bit=load_4bit)
        self.query_template = "请先判断用户的意图，若为画图则在输出前加入<画图>:{}"

    def __call__(self, prompt, return_history=False, history=None, skip_special=False):
        enhanced_prompt = eval_model(
            models=self.models,
            query=self.query_template.format(prompt),
            image_file=None,
            return_history=return_history,
            history=history,
            skip_special=skip_special,
        )
        if return_history:
            return enhanced_prompt

        enhanced_prompt, compliance = remove_prefix(enhanced_prompt)
        if not compliance:
            return False, ""
        return True, enhanced_prompt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./ckpts/dialoggen")
    parser.add_argument("--prompt", type=str, default="画一只小猫")
    parser.add_argument("--image_file", type=str, default=None)  # 'images/demo1.jpeg'
    args = parser.parse_args()

    query = f"请先判断用户的意图，若为画图则在输出前加入<画图>:{args.prompt}"

    models = init_dialoggen_model(args.model_path)

    res = eval_model(
        models,
        query=query,
        image_file=args.image_file,
    )
    print(res)
