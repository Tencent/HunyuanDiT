import gradio as gr
import pandas as pd
from pathlib import Path
from PIL import Image
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from hydit.constants import SAMPLER_FACTORY
from sample_t2i import inferencer

ROOT = Path(__file__).parent.parent
SAMPLERS = list(SAMPLER_FACTORY.keys())
SIZES = {
    "square": (1024, 1024),
    "landscape": (768, 1280),
    "portrait": (1280, 768),
}

def get_strings(lang):
    lang_file = Path(f"app/lang/{lang}.csv")
    strings = pd.read_csv(lang_file, header=0)
    strings = strings.set_index("key")['value'].to_dict()
    return strings


args, gen, enhancer = inferencer()
strings = get_strings(args.lang)


def infer(
    prompt,
    negative_prompt,
    seed,
    cfg_scale,
    infer_steps,
    oriW, oriH,
    sampler,
    size,
    enhance
):
    if enhance and enhancer is not None:
        success, enhanced_prompt = enhancer(prompt)
        if not success:
            fail_image = Image.open(ROOT / 'app/fail.png')
            return fail_image
    else:
        enhanced_prompt = None

    height, width = SIZES[size]
    results = gen.predict(prompt,
                          height=height,
                          width=width,
                          seed=seed,
                          enhanced_prompt=enhanced_prompt,
                          negative_prompt=negative_prompt,
                          infer_steps=infer_steps,
                          guidance_scale=cfg_scale,
                          batch_size=1,
                          src_size_cond=(oriW, oriH),
                          sampler=sampler,
                          )
    image = results['images'][0]
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
                with gr.Row():
                    size = gr.Radio(
                        label=strings['size'], choices=[
                            (strings['square'], 'square'),
                            (strings['landscape'], 'landscape'),
                            (strings['portrait'], 'portrait'),
                        ],
                        value="square"
                    )
                prompt = gr.Textbox(label=strings['prompt'], value=strings['default prompt'], lines=3)
                with gr.Row():
                    infer_steps = gr.Slider(
                        label=strings['infer steps'], minimum=1, maximum=200, value=100, step=1,
                    )
                    seed = gr.Number(
                        label=strings['seed'], minimum=-1, maximum=1_000_000_000, value=42, step=1, precision=0,
                    )
                    enhance = gr.Checkbox(
                        label=strings['enhance'], value=enhancer is not None, interactive=True,
                    )

                with gr.Accordion(
                    strings['accordion'], open=False
                ):
                    with gr.Row():
                        negative_prompt = gr.Textbox(label=strings['negative_prompt'],
                                                     value=gen.default_negative_prompt,
                                                     lines=2,
                                                     )
                    with gr.Row():
                        sampler = gr.Dropdown(SAMPLERS, label=strings['sampler'], value="ddpm")
                        cfg_scale = gr.Slider(
                            label=strings['cfg'], minimum=1.0, maximum=16.0, value=6.0, step=1
                        )
                        oriW = gr.Number(
                            label=strings['width cond'], minimum=1024, maximum=4096, value=1024, step=64, precision=0,
                            min_width=80,
                        )
                        oriH = gr.Number(
                            label=strings['height cond'], minimum=1024, maximum=4096, value=1024, step=64, precision=0,
                            min_width=80,
                        )
                with gr.Row():
                    advanced_button = gr.Button(strings['run'])
            with gr.Column():
                default_img = Image.open(ROOT / 'app/default.png')
                output_img = gr.Image(
                    label=strings['generated image'],
                    interactive=False,
                    format='png',
                    value=default_img,
                )
            advanced_button.click(
                fn=infer,
                inputs=[
                    prompt, negative_prompt, seed, cfg_scale, infer_steps,
                    oriW, oriH, sampler, size, enhance,
                ],
                outputs=output_img,
            )

        with gr.Row():
            gr.Examples([
                ['一只小猫'],
                ['现实主义风格，画面主要描述一个巴洛克风格的花瓶，带有金色的装饰边框，花瓶上盛开着各种色彩鲜艳的花，白色背景'],
                ['一只聪明的狐狸走在阔叶树林里, 旁边是一条小溪, 细节真实, 摄影'],
                ['飞流直下三千尺，疑是银河落九天'],
                ['一只长靴猫手持亮银色的宝剑，身着铠甲，眼神坚毅，站在一堆金币上，背景是暗色调的洞穴，图像上有金币的光影点缀。'],
                ['麻婆豆腐'],
                ['苏州园林'],
                ['一颗新鲜的草莓特写，红色的外表，表面布满许多种子，背景是淡绿色的叶子'],
                ['请将“杞人忧天”的样子画出来'],
                ['枯藤老树昏鸦，小桥流水人家'],
                ['湖水清澈，天空湛蓝，阳光灿烂。一只优雅的白天鹅在湖边游泳。它周围有几只小鸭子，看起来非常可爱，整个画面给人一种宁静祥和的感觉。'],
                ['一朵鲜艳的红色玫瑰花，花瓣撒有一些水珠，晶莹剔透，特写镜头'],
                ['臭豆腐'],
                ['九寨沟'],
                ['俗语“鲤鱼跃龙门”'],
                ['风格是写实，画面主要描述一个亚洲戏曲艺术家正在表演，她穿着华丽的戏服，脸上戴着精致的面具，身姿优雅，背景是古色古香的舞台，镜头是近景'],
            ],
            [prompt],
            label=strings['examples']
            )
    return block


if __name__ == "__main__":
    interface = ui()
    interface.launch(server_name="0.0.0.0", server_port=443, share=True)
