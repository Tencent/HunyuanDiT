## Using HunyuanDiT Inference with under 6GB GPU VRAM

### Instructions
Running HunyuanDiT in under 6GB GPU VRAM is available now based on [**diffusers**](https://huggingface.co/docs/diffusers/main/en/api/pipelines/hunyuandit). Here we provide instructions and demo for your quick start.

The 6Glite version supports Nvidia Ampere architecture series graphics cards such as RTX 3070/3080/4080/4090, A100, and so on.

The only thing you need do is to install the following library:

```bash
pip install -U bitsandbytes
pip install git+https://github.com/huggingface/diffusers
pip install torch==2.0.0
```

Then you can enjoy your HunyuanDiT text-to-image journey under 6GB GPU VRAM directly!

Here is a demo for you.

```bash
cd HunyuanDiT

# Quick start
model_id=Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled
prompt=一个宇航员在骑马
infer_steps=50
guidance_scale=6
python3 lite/inference.py ${model_id} ${prompt} ${infer_steps} ${guidance_scale}
```

Note: To use other features in hydit requires torch 1.13.1. In this case, you may need to downgrade your torch version.

```bash
pip install torch==1.13.1
``` 