# Hunyuan-DiT + 🤗 Diffusers

You can use Hunyuan-DiT in 🤗 Diffusers library. Before using the pipelines, please install the latest version of 🤗 Diffusers with
```bash
pip install git+https://github.com/huggingface/diffusers.git
```

## Inference with th Base Model

You can generate images with both Chinese and English prompts using the following Python script:
```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt).images[0]
```
You can use our distilled model to generate images even faster:

```py
import torch
from diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled", torch_dtype=torch.float16)
pipe.to("cuda")

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt, num_inference_steps=25).images[0]
```
More details can be found in [HunyuanDiT-v1.2-Diffusers-Distilled](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled)

## LoRA
LoRA can be integrated with Hunyuan-DiT inside the 🤗 Diffusers framework. 
The following example loads and uses the pre-trained LoRA. To try it, please start by downloading our pre-trained LoRA checkpoints,
```bash
huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora
```
Then run the following code snippet to use the jade LoRA:
```python
import torch
from diffusers import HunyuanDiTPipeline

### convert checkpoint to diffusers format
num_layers = 40
def load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale):
    for i in range(num_layers):
        Wqkv = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn1.Wqkv.lora_A.weight"]) 
        q, k, v = torch.chunk(Wqkv, 3, dim=0)
        transformer_state_dict[f"blocks.{i}.attn1.to_q.weight"] += lora_scale * q
        transformer_state_dict[f"blocks.{i}.attn1.to_k.weight"] += lora_scale * k
        transformer_state_dict[f"blocks.{i}.attn1.to_v.weight"] += lora_scale * v

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn1.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn1.to_out.0.weight"] += lora_scale * out_proj

        q_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.q_proj.lora_A.weight"])
        transformer_state_dict[f"blocks.{i}.attn2.to_q.weight"] += lora_scale * q_proj

        kv_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.kv_proj.lora_A.weight"])
        k, v = torch.chunk(kv_proj, 2, dim=0)
        transformer_state_dict[f"blocks.{i}.attn2.to_k.weight"] += lora_scale * k
        transformer_state_dict[f"blocks.{i}.attn2.to_v.weight"] += lora_scale * v

        out_proj = torch.matmul(lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_B.weight"], lora_state_dict[f"blocks.{i}.attn2.out_proj.lora_A.weight"]) 
        transformer_state_dict[f"blocks.{i}.attn2.to_out.0.weight"] += lora_scale * out_proj
    
    q_proj = torch.matmul(lora_state_dict["pooler.q_proj.lora_B.weight"], lora_state_dict["pooler.q_proj.lora_A.weight"])
    transformer_state_dict["time_extra_emb.pooler.q_proj.weight"] += lora_scale * q_proj
    
    return transformer_state_dict

### use the diffusers pipeline with lora
pipe = HunyuanDiTPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", torch_dtype=torch.float16)
pipe.to("cuda")

from safetensors import safe_open

lora_state_dict = {}
with safe_open("./ckpts/t2i/lora/jade/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        lora_state_dict[k[17:]] = f.get_tensor(k) # remove 'basemodel.model'

transformer_state_dict = pipe.transformer.state_dict()
transformer_state_dict = load_hunyuan_dit_lora(transformer_state_dict, lora_state_dict, lora_scale=1.0)
pipe.transformer.load_state_dict(transformer_state_dict)

prompt = "玉石绘画风格，一只猫在追蝴蝶"
image = pipe(
    prompt, 
    num_inference_steps=100,
    guidance_scale=6.0, 
).images[0]
image.save('img.png')
``` 

You can control the strength of LoRA by changing the `lora_scale` parameter.

## ControlNet
Hunyuan-DiT + ControlNet is supported in 🤗 Diffusers. The following example shows how to use Hunyuan-DiT + Canny ControlNet.
```py
from diffusers import HunyuanDiT2DControlNetModel, HunyuanDiTControlNetPipeline
import torch
controlnet = HunyuanDiT2DControlNetModel.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny", torch_dtype=torch.float16)

pipe = HunyuanDiTControlNetPipeline.from_pretrained("Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers", controlnet=controlnet, torch_dtype=torch.float16)
pipe.to("cuda")

from diffusers.utils import load_image
cond_image = load_image('https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2-ControlNet-Diffusers-Canny/resolve/main/canny.jpg?download=true')

## You may also use English prompt as HunyuanDiT supports both English and Chinese
prompt="在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围"
#prompt="At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere."
image = pipe(
    prompt,
    height=1024,
    width=1024,
    control_image=cond_image,
    num_inference_steps=50,
).images[0]
```

There are other pre-trained ControlNets available. Please have a look at [the official huggingface website of Tencent Hunyuan Team](https://huggingface.co/Tencent-Hunyuan)

