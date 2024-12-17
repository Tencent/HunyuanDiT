import gradio as gr
import hunyuan_test_lora
import hunyuan_test_lycoris
import hunyuan_test_dreambooth


def process_selection(
    selection,
    prompt,
    neg_prompt,
    height,
    weight,
    steps,
    cfg_scale,
    seed,
    model_path,
    ckpt_path,
    model_version,
):
    result = None

    if "Dreambooth" == selection:
        result = inference_by_dreambooth(
            prompt,
            neg_prompt,
            height,
            weight,
            steps,
            cfg_scale,
            seed,
            model_path,
            ckpt_path,
            model_version,
        )

    if "LoRA" == selection:
        result = inference_by_lora(
            prompt,
            neg_prompt,
            height,
            weight,
            steps,
            cfg_scale,
            seed,
            model_path,
            ckpt_path,
            model_version,
        )

    if "LyCORIS" == selection:
        result = inference_by_lycoris(
            prompt,
            neg_prompt,
            height,
            weight,
            steps,
            cfg_scale,
            seed,
            model_path,
            ckpt_path,
            model_version,
        )

    return "output.png"


def inference_by_dreambooth(
    prompt,
    neg_prompt,
    height,
    weight,
    steps,
    cfg_scale,
    seed,
    model_path,
    ckpt_path,
    model_version,
):
    hunyuan_test_dreambooth.generate_image(
        prompt,
        neg_prompt,
        seed,
        height,
        weight,
        steps,
        cfg_scale,
        model_path,
        ckpt_path,
        model_version,
    )


def inference_by_lora(
    prompt,
    neg_prompt,
    height,
    weight,
    steps,
    cfg_scale,
    seed,
    model_path,
    ckpt_path,
    model_version,
):
    hunyuan_test_lora.generate_image(
        prompt,
        neg_prompt,
        seed,
        height,
        weight,
        steps,
        cfg_scale,
        model_path,
        ckpt_path,
        model_version,
    )


def inference_by_lycoris(
    prompt,
    neg_prompt,
    height,
    weight,
    steps,
    cfg_scale,
    seed,
    model_path,
    ckpt_path,
    model_version,
):
    hunyuan_test_lycoris.generate_image(
        prompt,
        neg_prompt,
        seed,
        height,
        weight,
        steps,
        cfg_scale,
        model_path,
        ckpt_path,
        model_version,
    )


fintune_options = ["Dreambooth", "LoRA", "LyCORIS"]
version_options = ["1.1", "1.2"]

iface = gr.Interface(
    process_selection,
    inputs=[
        gr.Radio(
            choices=fintune_options,
            label="Please select a training method.",
            value="Dreambooth",
        ),
        gr.Textbox(label="prompt", value="画一只小猫"),
        gr.Textbox(label="neg_prompt", value=""),
        gr.Number(label="height", value=1024),
        gr.Number(label="weight", value=1024),
        gr.Number(label="steps", value=30),
        gr.Number(label="cfg_scale", value=5),
        gr.Number(label="seed", value=287816226),
        gr.Textbox(label="model_path", value="/data/model_path/.."),
        gr.Textbox(label="ckpt_path", value="/data/ckpt_path/.."),
        gr.Radio(choices=version_options, label="HunYuanDiT version", value="1.2"),
    ],
    outputs=gr.Image(label="Outputs"),
    title="HunyuanDIT Inference Tool Adapted for Kohya",
)

iface.launch(server_name="0.0.0.0", server_port=7888)
