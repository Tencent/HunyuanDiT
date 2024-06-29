
## Using LoRA to fine-tune HunyuanDiT


### Instructions

 The dependencies and installation are basically the same as the [**base model**](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2).

 We provide two types of trained LoRA weights for you to test.
 
 Then download the model using the following commands:

```bash
cd HunyuanDiT
# Use the huggingface-cli tool to download the model.
huggingface-cli download Tencent-Hunyuan/HYDiT-LoRA --local-dir ./ckpts/t2i/lora

# Quick start
python sample_t2i.py --prompt "青花瓷风格，一只猫在追蝴蝶"  --no-enhance --load-key ema --lora-ckpt ./ckpts/t2i/lora/porcelain --infer-mode fa
```

Examples of training data and inference results are as follows:
<table>
  <tr>
    <td colspan="4" align="center">Examples of training data</td>
  </tr>
  
  <tr>
    <td align="center"><img src="asset/porcelain/train/0.png" alt="Image 0" width="200"/></td>
    <td align="center"><img src="asset/porcelain/train/1.png" alt="Image 1" width="200"/></td>
    <td align="center"><img src="asset/porcelain/train/2.png" alt="Image 2" width="200"/></td>
    <td align="center"><img src="asset/porcelain/train/3.png" alt="Image 3" width="200"/></td>
  </tr>
  <tr>
    <td align="center">青花瓷风格，一只蓝色的鸟儿站在蓝色的花瓶上，周围点缀着白色花朵，背景是白色 （Porcelain style, a blue bird stands on a blue vase, surrounded by white flowers, with a white background.
）</td>
    <td align="center">青花瓷风格，这是一幅蓝白相间的陶瓷盘子，上面描绘着一只狐狸和它的幼崽在森林中漫步，背景是白色 （Porcelain style, this is a blue and white ceramic plate depicting a fox and its cubs strolling in the forest, with a white background.）</td>
    <td align="center">青花瓷风格，在黑色背景上，一只蓝色的狼站在蓝白相间的盘子上，周围是树木和月亮 （Porcelain style, on a black background, a blue wolf stands on a blue and white plate, surrounded by trees and the moon.）</td>
    <td align="center">青花瓷风格，在蓝色背景上，一只蓝色蝴蝶和白色花朵被放置在中央 （Porcelain style, on a blue background, a blue butterfly and white flowers are placed in the center.）</td>
  </tr>
  <tr>
    <td colspan="4" align="center">Examples of inference results</td>
  </tr>
  <tr>
    <td align="center"><img src="asset/porcelain/inference/0.png" alt="Image 4" width="200"/></td>
    <td align="center"><img src="asset/porcelain/inference/1.png" alt="Image 5" width="200"/></td>
    <td align="center"><img src="asset/porcelain/inference/2.png" alt="Image 6" width="200"/></td>
    <td align="center"><img src="asset/porcelain/inference/3.png" alt="Image 7" width="200"/></td>
  </tr>
  <tr>
    <td align="center">青花瓷风格，苏州园林 （Porcelain style,  Suzhou Gardens.）</td>
    <td align="center">青花瓷风格，一朵荷花 （Porcelain style,  a lotus flower.）</td>
    <td align="center">青花瓷风格，一只羊（Porcelain style, a sheep.）</td>
    <td align="center">青花瓷风格，一个女孩在雨中跳舞（Porcelain style, a girl dancing in the rain.）</td>
  </tr>
  
</table>


### Training
    
We provide three types of weights for fine-tuning LoRA, `ema`, `module` and `distill`, and you can choose according to the actual effect. By default, we use `ema` weights. 

Here is an example for LoRA with HunYuanDiT v1.2, we load the `distill` weights into the main model and perform LoRA fine-tuning through the `resume_module_root=./ckpts/t2i/model/pytorch_model_distill.pt` setting. 

If multiple resolution are used, you need to add the `--multireso` and `--reso-step 64 ` parameter. 

If you want to train LoRA with HunYuanDiT v1.1, you could add `--use-style-cond`, `--size-cond 1024 1024` and `--beta-end 0.03`.

```bash
model='DiT-g/2'                                                   # model type
task_flag="lora_porcelain_ema_rank64"                             # task flag
resume_module_root=./ckpts/t2i/model/pytorch_model_distill.pt     # resume checkpoint
index_file=dataset/porcelain/jsons/porcelain.json                 # the selected data indices
results_dir=./log_EXP                                             # save root for results
batch_size=1                                                      # training batch size
image_size=1024                                                   # training image resolution
grad_accu_steps=2                                                 # gradient accumulation steps
warmup_num_steps=0                                                # warm-up steps
lr=0.0001                                                         # learning rate
ckpt_every=100                                                    # create a ckpt every a few steps.
ckpt_latest_every=2000                                            # create a ckpt named `latest.pt` every a few steps.
rank=64                                                           # rank of lora
max_training_steps=2000                                           # Maximum training iteration steps

PYTHONPATH=./ deepspeed hydit/train_deepspeed.py \
    --task-flag ${task_flag} \
    --model ${model} \
    --training-parts lora \
    --rank ${rank} \
    --resume \
    --resume-module-root ${resume_module_root} \
    --lr ${lr} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0 \
    --uncond-p-t5 0 \
    --index-file ${index_file} \
    --random-flip \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --ema-dtype fp32 \
    --results-dir ${results_dir} \
    --ckpt-every ${ckpt_every} \
    --max-training-steps ${max_training_steps}\
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 10 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --qk-norm \
    --rope-img base512 \
    --rope-real \
    "$@"
```

Recommended parameter settings

|     Parameter     |  Description  |          Recommended Parameter Value                               | Note|
|:---------------:|:---------:|:---------------------------------------------------:|:--:|
|   `--batch-size` |    Training batch size    |        1        | Depends on GPU memory|
|   `--grad-accu-steps` |    Size of gradient accumulation    |       2        | - |
|   `--rank` |    Rank of lora    |       64        | Choosing from 8-128 |
|   `--max-training-steps` |    Training steps  |       2000        | Depend on training data size, for reference apply 2000 steps on 100 images|
|   `--lr` |    Learning rate  |        0.0001        | - |


### Inference

After the training is complete, you can use the following command line for inference.
We provide the `--lora-ckpt` parameter for selecting the folder which contains lora weights and configurations.

a. Using LoRA during inference

```bash
python sample_t2i.py --infer-mode fa --prompt "青花瓷风格，一只小狗"  --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt/
```

b. Using LoRA in gradio
```bash
python app/hydit_app.py --infer-mode fa --no-enhance --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0001000.pt/
```

c. Merge LoRA weights into the main model

We provide the `--output-merge-path` parameter to set the path for saving the merged weights.

```bash
PYTHONPATH=./ python lora/merge.py --lora-ckpt log_EXP/001-lora_porcelain_ema_rank64/checkpoints/0000100.pt/ --output-merge-path ./ckpts/t2i/model/pytorch_model_merge.pt
```

d. Regarding how to use the LoRA weights we trained in diffusion, we provide the following script. To ensure compatibility with the diffuser, some modifications are made, which means that LoRA cannot be directly loaded. 


```python
import torch
from diffusers import HunyuanDiTPipeline

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


e. For more information, please refer to [HYDiT-LoRA](https://huggingface.co/Tencent-Hunyuan/HYDiT-LoRA).
