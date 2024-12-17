import gradio as gr
import pandas as pd
from pathlib import Path
from PIL import Image, PngImagePlugin
import sys
import numpy as np
import torch
from torchvision import transforms as T

sys.path.insert(0, str(Path(__file__).parent.parent))
import datetime

from hydit.constants import SAMPLER_FACTORY
from sample_t2i import inferencer
import os

ROOT = Path(__file__).parent.parent
SAMPLERS = list(SAMPLER_FACTORY.keys())

norm_transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize([0.5], [0.5]),
    ]
)


def get_strings(lang):
    lang_file = Path(f"app/lang/{lang}.csv")
    strings = pd.read_csv(lang_file, header=0)
    strings = strings.set_index("key")["value"].to_dict()
    return strings


def get_files_with_extension(path, extension):
    return {
        os.path.splitext(file)[0]: os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
        and any(file.endswith(ext) for ext in extension)
    }


args, gen, enhancer = inferencer()
output_dir = ROOT / f"{args.output_img_path}"
os.makedirs(output_dir, exist_ok=True)
strings = get_strings(args.lang)
controlnet_list = get_files_with_extension(
    args.model_root + "/t2i/controlnet",
    [".pt", ".safetensors"],
)
module_list = get_files_with_extension(
    args.model_root + "/t2i/model",
    [".pt", ".safetensors"],
)
lora_list = get_files_with_extension(
    args.model_root + "/t2i/lora",
    [".pt", ".safetensors"],
)


def upgrade_dit_model_load(model):
    model_path = module_list[model]
    gen.args.dit_weight = model_path
    gen.load_torch_weights()


def generate_metadata(
    prompt,
    negative_prompt,
    seed,
    cfg_scale,
    infer_steps,
    sampler,
    imgW,
    imgH,
    controlnet_module,
    control_weight,
    lora_ctrls,
):
    """生成图像元数据。"""
    return {
        "parameters": "Power by HunYun",
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "cfg_scale": cfg_scale,
        "infer_steps": infer_steps,
        "sampler": sampler,
        "imgW": imgW,
        "imgH": imgH,
        "controlnet_module": controlnet_module,
        "control_weight": control_weight,
        "lora_ctrls": [
            {
                "lora_enabled": lora_ctrl[0],
                "lora_model": lora_ctrl[1],
                "lora_weight": lora_ctrl[2],
            }
            for lora_ctrl in zip(*[iter(lora_ctrls)] * 3)
        ],
        "model_name": gen.model_name,
    }


def infer(
    prompt,
    negative_prompt,
    seed,
    cfg_scale,
    infer_steps,
    sampler,
    imgW,
    imgH,
    input_image,
    controlnet_module,
    control_weight,
    enhance,
    img_crop_type,
    *lora_ctrls,
):
    if enhance and enhancer is not None:
        success, enhanced_prompt = enhancer(prompt)
        if not success:
            fail_image = Image.open(ROOT / "app/fail.png")
            return fail_image
    else:
        enhanced_prompt = None
    active_loras = [
        {"model": lora_ctrls[i + 1], "weight": lora_ctrls[i + 2]}
        for i in range(0, len(lora_ctrls), 3)
        if lora_ctrls[i]
    ]
    if input_image is not None:
        # # Convert image to PyTorch tensor if it is a NumPy array
        if isinstance(input_image, np.ndarray):
            input_image = Image.fromarray(input_image).convert("RGB")

        input_image = gen.pixel_perfect_resolution(
            input_image, imgH, imgW, img_crop_type
        )
        # Apply the normalization transform
        input_image = norm_transform(input_image)

        # Add batch dimension and move to GPU (if available)
        input_image = (
            input_image.unsqueeze(0).cuda()
            if torch.cuda.is_available()
            else input_image.unsqueeze(0)
        )

    results = gen.predict(
        prompt,
        image=input_image,
        height=imgH,
        width=imgW,
        seed=seed,
        enhanced_prompt=enhanced_prompt,
        negative_prompt=negative_prompt,
        infer_steps=infer_steps,
        guidance_scale=cfg_scale,
        batch_size=1,
        src_size_cond=None,
        sampler=sampler,
        control_weight=control_weight,
        controlnet=controlnet_module,
        lora_ctrls=active_loras,
    )
    image = results["images"][0]
    seed = results["seed"]
    metadata = generate_metadata(
        prompt,
        negative_prompt,
        seed,
        cfg_scale,
        infer_steps,
        sampler,
        imgW,
        imgH,
        controlnet_module,
        control_weight,
        active_loras,
    )
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = output_dir.joinpath(f"generated_image_{timestamp}_{seed}.png")
    png_info = PngImagePlugin.PngInfo()
    for k, v in metadata.items():
        png_info.add_text(k, str(v))
    image.save(
        output_path,
        pnginfo=png_info,
    )
    return image


def ui():
    block = gr.Blocks()
    description = f"""
    # {strings['title']}
    
    ## {strings['desc']}
    
    """

    with block:
        with gr.Row():
            gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                prompt = gr.Textbox(
                    label=strings["prompt"], value=strings["default prompt"], lines=3
                )
                with gr.Row():
                    imgW = gr.Slider(
                        label=strings["width"],
                        minimum=64,
                        maximum=4096,
                        value=1024,
                        step=64,
                    )
                    imgH = gr.Slider(
                        label=strings["height"],
                        minimum=64,
                        maximum=4096,
                        value=1024,
                        step=64,
                    )
                with gr.Row():
                    infer_steps = gr.Slider(
                        label=strings["infer steps"],
                        minimum=1,
                        maximum=200,
                        value=100,
                        step=1,
                    )
                    seed = gr.Number(
                        label=strings["seed"],
                        minimum=-1,
                        maximum=1_000_000_000,
                        value=0,
                        step=1,
                        precision=0,
                    )
                    enhance = gr.Checkbox(
                        label=strings["enhance"],
                        value=enhancer is not None,
                        interactive=True,
                    )

                with gr.Accordion(strings["accordion"], open=False):
                    with gr.Row():
                        negative_prompt = gr.Textbox(
                            label=strings["negative_prompt"],
                            value=gen.default_negative_prompt,
                            lines=2,
                        )
                    with gr.Row():
                        sampler = gr.Dropdown(
                            SAMPLERS, label=strings["sampler"], value="ddpm"
                        )
                        cfg_scale = gr.Slider(
                            label=strings["cfg"],
                            minimum=1.0,
                            maximum=16.0,
                            value=6.0,
                            step=1,
                        )

                    with gr.Accordion(strings["model_list"], open=False):
                        with gr.Row():
                            dit_model = gr.Dropdown(
                                label=strings["dit_model"],
                                choices=[
                                    name
                                    for name, path in get_files_with_extension(
                                        args.model_root + "/t2i/model",
                                        [".pt", ".safetensors"],
                                    ).items()
                                ],
                                value=f"pytorch_model_{args.load_key}",
                            )
                            dit_model.change(
                                fn=upgrade_dit_model_load,
                                inputs=dit_model,
                                outputs=None,
                            )
                    with gr.Accordion(strings["lora_list"], open=False):
                        lora_ctrls = []
                        for i in range(5):
                            with gr.Row():
                                lora_enabled = gr.Checkbox(
                                    label="Enable",
                                    value=False,
                                )
                                lora_model = gr.Dropdown(
                                    label=f"Lora{i+1}",
                                    choices=["none"]
                                    + [name for name, path in lora_list.items()],
                                    value="none",
                                )
                                lora_weight = gr.Slider(
                                    label="weight",
                                    minimum=-1,
                                    maximum=2,
                                    step=0.01,
                                    value=0,
                                    scale=5,
                                )
                                lora_ctrls += [lora_enabled, lora_model, lora_weight]

                with gr.Accordion(strings["controlnet"], open=False):
                    with gr.Row():
                        controlnet_module = gr.Dropdown(
                            label=strings["controlnet_model"],
                            choices=["None"]
                            + [name for name, path in controlnet_list.items()],
                            value="None",
                        )
                        control_weight = gr.Slider(
                            label=strings["Control_Weight"],
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            step=0.1,
                        )
                    input_image = gr.Image(label=strings["input image"])
                    with gr.Row():
                        img_crop_type = gr.Radio(
                            label=strings["Crop_mode"],
                            choices=[
                                (strings["Resize"], "Resize"),
                                (strings["Crop_and_Resize"], "Crop_and_Resize"),
                                (strings["Resize_and_Fill"], "Resize_and_Fill"),
                            ],
                            value="Crop_and_Resize",
                        )
                with gr.Row():
                    advanced_button = gr.Button(strings["run"])
            with gr.Column():
                default_img = Image.open(ROOT / "app/default.png")
                output_img = gr.Image(
                    label=strings["generated image"],
                    interactive=False,
                    format="png",
                    value=default_img,
                )
            advanced_button.click(
                fn=infer,
                inputs=[
                    prompt,
                    negative_prompt,
                    seed,
                    cfg_scale,
                    infer_steps,
                    sampler,
                    imgW,
                    imgH,
                    input_image,
                    controlnet_module,
                    control_weight,
                    enhance,
                    img_crop_type,
                    *lora_ctrls,
                ],
                outputs=output_img,
            )

        with gr.Row():
            gr.Examples(
                [
                    ["一只小猫"],
                    [
                        "现实主义风格，画面主要描述一个巴洛克风格的花瓶，带有金色的装饰边框，花瓶上盛开着各种色彩鲜艳的花，白色背景"
                    ],
                    ["一只聪明的狐狸走在阔叶树林里, 旁边是一条小溪, 细节真实, 摄影"],
                    ["飞流直下三千尺，疑是银河落九天"],
                    [
                        "一只长靴猫手持亮银色的宝剑，身着铠甲，眼神坚毅，站在一堆金币上，背景是暗色调的洞穴，图像上有金币的光影点缀。"
                    ],
                    ["麻婆豆腐"],
                    ["苏州园林"],
                    [
                        "一颗新鲜的草莓特写，红色的外表，表面布满许多种子，背景是淡绿色的叶子"
                    ],
                    ["请将“杞人忧天”的样子画出来"],
                    ["枯藤老树昏鸦，小桥流水人家"],
                    [
                        "湖水清澈，天空湛蓝，阳光灿烂。一只优雅的白天鹅在湖边游泳。它周围有几只小鸭子，看起来非常可爱，整个画面给人一种宁静祥和的感觉。"
                    ],
                    ["一朵鲜艳的红色玫瑰花，花瓣撒有一些水珠，晶莹剔透，特写镜头"],
                    ["臭豆腐"],
                    ["九寨沟"],
                    ["俗语“鲤鱼跃龙门”"],
                    [
                        "风格是写实，画面主要描述一个亚洲戏曲艺术家正在表演，她穿着华丽的戏服，脸上戴着精致的面具，身姿优雅，背景是古色古香的舞台，镜头是近景"
                    ],
                ],
                [prompt],
                label=strings["examples"],
            )
    return block


if __name__ == "__main__":
    interface = ui()
    interface.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.gradio_share,
    )
