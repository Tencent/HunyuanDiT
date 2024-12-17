# -- coding: utf-8 --
#!/usr/bin/env python
import gradio as gr
from PIL import Image
import sys
import os

sys.path.append(os.getcwd())
import json
import numpy as np
from pathlib import Path
import io
import hashlib
import requests
import base64
import pandas as pd
from sample_t2i import inferencer
from mllm.dialoggen_demo import init_dialoggen_model, eval_model

SIZES = {
    "正方形(square, 1024x1024)": (1024, 1024),
    "风景(landscape, 1280x768)": (768, 1280),
    "人像(portrait, 768x1280)": (1280, 768),
}

global_seed = np.random.randint(0, 10000)


# Helper Functions
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    return encoded_image


def get_strings(lang):
    lang_file = Path(f"app/lang/{lang}.csv")
    strings = pd.read_csv(lang_file, header=0)
    strings = strings.set_index("key")["value"].to_dict()
    return strings


def get_image_md5(image):
    image_data = io.BytesIO()
    image.save(image_data, format="PNG")
    image_data = image_data.getvalue()
    md5_hash = hashlib.md5(image_data).hexdigest()
    return md5_hash


# mllm调用
def request_dialogGen(
    server_url="http://0.0.0.0:8080",
    history_messages=[],
    question="画一个木制的鸟",
    image="",
):
    if image != "":
        image = base64.b64encode(open(image, "rb").read()).decode()
    print("history_messages before request", history_messages)
    headers = {"accept": "application/json", "Content-Type": "application/json"}
    data = {
        "text": question,
        "image": image,  # "image为空字符串，则进行文本对话"
        "history": history_messages,
    }
    response = requests.post(server_url, headers=headers, json=data)
    print("response", response)
    response = response.json()
    print(response)
    response_text = response["result"]
    history_messages = response["history"]
    print("history_messages before request", history_messages)
    return history_messages, response_text


# 画图
def image_generation(prompt, infer_steps, seed, image_size):
    print(
        f"prompt sent to T2I model: {prompt}, infer_steps: {infer_steps}, seed: {seed}, size: {image_size}"
    )
    height, width = SIZES[image_size]
    results = gen.predict(
        prompt,
        height=height,
        width=width,
        seed=seed,
        infer_steps=infer_steps,
        batch_size=1,
    )
    image = results["images"][0]
    file_name = get_image_md5(image)
    # Save images
    save_dir = Path("results")
    save_dir.mkdir(exist_ok=True)
    save_path = f"results/multiRound_{file_name}.png"
    image.save(save_path)
    encoded_image = image_to_base64(save_path)

    return encoded_image


# 图文对话
def chat(history_messages, input_text):

    history_messages, response_text = request_dialogGen(
        history_messages=history_messages, question=input_text
    )
    return history_messages, response_text


#
def pipeline(input_text, state, infer_steps, seed, image_size):

    # 忽略空输入
    if len(input_text) == 0:
        return state, state[0]

    conversation = state[0]
    history_messages = state[1]

    system_prompt = "请先判断用户的意图，若为画图则在输出前加入<画图>:"
    print(f"input history:{history_messages}")
    if not isinstance(history_messages, list) and len(history_messages.messages) >= 2:
        response, history_messages = enhancer(
            input_text, return_history=True, history=history_messages, skip_special=True
        )
    else:
        response, history_messages = enhancer(
            input_text,
            return_history=True,
            history=history_messages,
            skip_special=False,
        )

    history_messages.messages[-1][-1] = response

    if "<画图>" in response:
        intention_draw = True
    else:
        intention_draw = False

    print(f"response:{response}")
    print("-" * 80)
    print(f"history_messages:{history_messages}")
    print(f"intention_draw:{intention_draw}")
    if intention_draw:
        prompt = response.split("<画图>")[-1]
        # 画图
        image_url = image_generation(prompt, infer_steps, seed, image_size)
        response = f'<img src="data:image/png;base64,{image_url}" style="display: inline-block;"><p style="font-size: 14px; color: #555; margin-top: 0;">{prompt}</p>'
    conversation += [((input_text, response))]
    return [conversation, history_messages], conversation


# 页面设计
def upload_image(state, image_input):
    conversation = state[0]
    history_messages = state[1]
    input_image = Image.open(image_input.name).resize((224, 224)).convert("RGB")
    input_image.save(image_input.name)  # Overwrite with smaller image.
    system_prompt = "请先判断用户的意图，若为画图则在输出前加入<画图>:"
    history_messages, response = request_dialogGen(
        question="这张图描述了什么？",
        history_messages=history_messages,
        image=image_input.name,
    )
    conversation += [
        (
            f'<img src="./file={image_input.name}"  style="display: inline-block;">',
            response,
        )
    ]
    print("conversation", conversation)
    print("history_messages after uploading image", history_messages)
    return [conversation, history_messages], conversation


def reset():
    global global_seed
    global_seed = np.random.randint(0, 10000)
    return [[], []], []


def reset_last(state):
    conversation, history = state[0], state[1]
    conversation = conversation[:-1]
    history.messages = history.messages[:-2]
    return [conversation, history], conversation


if __name__ == "__main__":

    # Initialize dialoggen and HunyuanDiT model
    args, gen, enhancer = inferencer()
    strings = get_strings(args.lang)

    css = """
        #chatbot { min-height: 800px; }
        #save-btn {
            background-image: linear-gradient(to right bottom, rgba(130,217,244, 0.9), rgba(158,231,214, 1.0));
        }
        #save-btn:hover {
            background-image: linear-gradient(to right bottom, rgba(110,197,224, 0.9), rgba(138,211,194, 1.0));
        }
        #share-btn {
            background-image: linear-gradient(to right bottom, rgba(130,217,244, 0.9), rgba(158,231,214, 1.0));
        }
        #share-btn:hover {
            background-image: linear-gradient(to right bottom, rgba(110,197,224, 0.9), rgba(138,211,194, 1.0));
        }
        #gallery { z-index: 999999; }
        #gallery img:hover {transform: scale(2.3); z-index: 999999; position: relative; padding-right: 30%; padding-bottom: 30%;}
        #gallery button img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; padding-bottom: 0;}
        @media (hover: none) {
            #gallery img:hover {transform: none; z-index: 999999; position: relative; padding-right: 0; 0;}
        }
        .html2canvas-container { width: 3000px !important; height: 3000px !important; }
    """

    with gr.Blocks(css=css) as demo:
        DESCRIPTION = """# <a style="color: black; text-decoration: none;">多轮对话绘图 Multi-turn Text2Image Generation</a>
            你可以参照[DialogGen](https://arxiv.org/abs/2403.08857)，通过简单的交互式语句来进行历史图片的修改，例如：主体编辑、增加主体、删除主体、背景更换、风格转换、镜头转换、图像合并。

            (You can modify historical images through simple interactive statements referred to [DialogGen](https://arxiv.org/abs/2403.08857), such as: enity edit, add object, remove object, change background, change style, change lens, and combine images. )
            
            例如，主体编辑 (For example, enity edit) :
            ```none
            Round1: 画一个木制的鸟
            (Round1: draw a wooden bird)
            
            Round2: 变成玻璃的
            (Round2: turn into glass)
            ```
        """

        gr.Markdown(DESCRIPTION)
        gr_state = gr.State([[], []])  # conversation, chat_history

        with gr.Row():
            with gr.Column(scale=1, min_width=1000):
                with gr.Row():
                    chatbot = gr.Chatbot(
                        elem_id="chatbot", label="DialogGen&HunyuanDiT"
                    )
                with gr.Row():
                    infer_steps = gr.Slider(
                        label="采样步数(sampling steps)",
                        minimum=1,
                        maximum=200,
                        value=100,
                        step=1,
                    )
                    seed = gr.Number(
                        label="种子(seed)",
                        minimum=-1,
                        maximum=1_000_000_000,
                        value=666,
                        step=1,
                        precision=0,
                    )
                    size_dropdown = gr.Dropdown(
                        choices=[
                            "正方形(square, 1024x1024)",
                            "风景(landscape, 1280x768)",
                            "人像(portrait, 768x1280)",
                        ],
                        value="正方形(square, 1024x1024)",
                        label="图片尺寸(Image Size)",
                    )

                with gr.Row():
                    # image_btn = gr.UploadButton("🖼️ Upload Image", file_types=["image"])
                    text_input = gr.Textbox(
                        label="提示词(prompt)", placeholder="输入提示词(Type a prompt)"
                    )

                    with gr.Column():
                        submit_btn = gr.Button(
                            "提交(Submit)", interactive=True, variant="primary"
                        )
                        clear_last_btn = gr.Button("回退(Undo)")
                        clear_btn = gr.Button("全部重置(Reset All)")
                with gr.Row():
                    gr.Examples(
                        [
                            ["画一个木制的鸟"],
                            ["一只小猫"],
                            [
                                "现实主义风格，画面主要描述一个巴洛克风格的花瓶，带有金色的装饰边框，花瓶上盛开着各种色彩鲜艳的花，白色背景"
                            ],
                            [
                                "一只聪明的狐狸走在阔叶树林里, 旁边是一条小溪, 细节真实, 摄影"
                            ],
                            ["飞流直下三千尺，疑是银河落九天"],
                            [
                                "一只长靴猫手持亮银色的宝剑，身着铠甲，眼神坚毅，站在一堆金币上，背景是暗色调的洞穴，图像上有金币的光影点缀。"
                            ],
                            ["麻婆豆腐"],
                            ["苏州园林"],
                            [
                                "一颗新鲜的草莓特写，红色的外表，表面布满许多种子，背景是淡绿色的叶子"
                            ],
                            ["枯藤老树昏鸦，小桥流水人家"],
                            [
                                "湖水清澈，天空湛蓝，阳光灿烂。一只优雅的白天鹅在湖边游泳。它周围有几只小鸭子，看起来非常可爱，整个画面给人一种宁静祥和的感觉。"
                            ],
                            [
                                "一朵鲜艳的红色玫瑰花，花瓣撒有一些水珠，晶莹剔透，特写镜头"
                            ],
                            ["臭豆腐"],
                            ["九寨沟"],
                            ["俗语“鲤鱼跃龙门”"],
                            [
                                "风格是写实，画面主要描述一个亚洲戏曲艺术家正在表演，她穿着华丽的戏服，脸上戴着精致的面具，身姿优雅，背景是古色古香的舞台，镜头是近景"
                            ],
                        ],
                        [text_input],
                        label=strings["examples"],
                    )
                gr.Markdown(
                    """<p style="font-size: 20px; color: #888;">powered by <a href="https://github.com/Centaurusalpha/DialogGen" target="_blank">DialogGen</a> and <a href="https://github.com/Tencent/HunyuanDiT" target="_blank">HunyuanDiT</a></p>"""
                )

        text_input.submit(
            pipeline,
            [text_input, gr_state, infer_steps, seed, size_dropdown],
            [gr_state, chatbot],
        )
        text_input.submit(lambda: "", None, text_input)  # Reset chatbox.
        submit_btn.click(
            pipeline,
            [text_input, gr_state, infer_steps, seed, size_dropdown],
            [gr_state, chatbot],
        )
        submit_btn.click(lambda: "", None, text_input)  # Reset chatbox.

        # image_btn.upload(upload_image, [gr_state, image_btn], [gr_state, chatbot])
        clear_last_btn.click(reset_last, [gr_state], [gr_state, chatbot])
        clear_btn.click(reset, [], [gr_state, chatbot])

    interface = demo
    interface.launch(server_name="0.0.0.0", server_port=443, share=False)
