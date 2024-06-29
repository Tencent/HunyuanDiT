
## Using HunyuanDiT ControlNet


### Instructions

 The dependencies and installation are basically the same as the [**base model**](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT-v1.2).

 We provide three types of ControlNet weights for you to test: canny, depth and pose ControlNet.
 
 Download the model using the following commands:

```bash
cd HunyuanDiT
# Use the huggingface-cli tool to download the model.
# We recommend using distilled weights as the base model for ControlNet inference, as our provided pretrained weights are trained on them.
huggingface-cli download Tencent-Hunyuan/HYDiT-ControlNet-v1.2 --local-dir ./ckpts/t2i/controlnet
huggingface-cli download Tencent-Hunyuan/Distillation-v1.2 ./pytorch_model_distill.pt --local-dir ./ckpts/t2i/model

# Quick start
python sample_controlnet.py --infer-mode fa --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0
```

Examples of condition input and ControlNet results are as follows:
<table>
  <tr>
    <td colspan="3" align="center">Condition Input</td>
  </tr>
  
   <tr>
    <td align="center">Canny ControlNet </td>
    <td align="center">Depth ControlNet </td>
    <td align="center">Pose ControlNet </td>
  </tr>

  <tr>
    <td align="center">在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围<br>（At night, an ancient Chinese-style lion statue stands in front of the hotel, its eyes gleaming as if guarding the building. The background is the hotel entrance at night, with a close-up, eye-level, and centered composition. This photo presents a realistic photographic style, embodies Chinese sculpture culture, and reveals a mysterious atmosphere.） </td>
    <td align="center">在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足。照片采用特写、平视和居中构图的方式，呈现出写实的效果<br>（In the dense forest, a black and white panda sits quietly among the green trees and red flowers, surrounded by mountains and oceans. The background is a daytime forest with ample light. The photo uses a close-up, eye-level, and centered composition to create a realistic effect.） </td>
    <td align="center">在白天的森林中，一位穿着绿色上衣的亚洲女性站在大象旁边。照片采用了中景、平视和居中构图的方式，呈现出写实的效果。这张照片蕴含了人物摄影文化，并展现了宁静的氛围<br>（In the daytime forest, an Asian woman wearing a green shirt stands beside an elephant. The photo uses a medium shot, eye-level, and centered composition to create a realistic effect. This picture embodies the character photography culture and conveys a serene atmosphere.） </td>
  </tr>

  <tr>
    <td align="center"><img src="asset/input/canny.jpg" alt="Image 0" width="200"/></td>
    <td align="center"><img src="asset/input/depth.jpg" alt="Image 1" width="200"/></td>
    <td align="center"><img src="asset/input/pose.jpg" alt="Image 2" width="200"/></td>
    
  </tr>
  
  <tr>
    <td colspan="3" align="center">ControlNet Output</td>
  </tr>

  <tr>
    <td align="center"><img src="asset/output/canny.jpg" alt="Image 3" width="200"/></td>
    <td align="center"><img src="asset/output/depth.jpg" alt="Image 4" width="200"/></td>
    <td align="center"><img src="asset/output/pose.jpg" alt="Image 5" width="200"/></td>
  </tr>
 
  
</table>


### Training

We utilize [**DWPose**](https://github.com/IDEA-Research/DWPose) for pose extraction. Please follow their guidelines to download the checkpoints and save them to `hydit/annotator/ckpts` directory. We provide serveral commands to quick install:
```bash
mkdir ./hydit/annotator/ckpts
wget -O ./hydit/annotator/ckpts/dwpose.zip https://dit.hunyuan.tencent.com/download/HunyuanDiT/dwpose.zip
unzip ./hydit/annotator/ckpts/dwpose.zip -d ./hydit/annotator/ckpts/
```
Additionally, ensure that you install the related dependencies.
```bash
pip install matplotlib==3.7.5
pip install onnxruntime_gpu==1.16.3
pip install opencv-python==4.8.1.78
```


We provide three types of weights for ControlNet training, `ema`, `module` and `distill`, and you can choose according to the actual effects. By default, we use `distill` weights. 

Here is an example, we load the `distill` weights into the main model and conduct ControlNet training. 

If apply multiple resolution training, you need to add the `--multireso` and `--reso-step 64` parameter. 

```bash
task_flag="canny_controlnet"                                   # the task flag is used to identify folders.
control_type=canny
resume_module_root=./ckpts/t2i/model/pytorch_model_distill.pt  # checkpoint root for resume
index_file=/path/to/your/indexfile                             # index file for dataloader
results_dir=./log_EXP                                          # save root for results
batch_size=1                                                   # training batch size
image_size=1024                                                # training image resolution
grad_accu_steps=2                                              # gradient accumulation
warmup_num_steps=0                                             # warm-up steps
lr=0.0001                                                      # learning rate
ckpt_every=10000                                               # create a ckpt every a few steps.
ckpt_latest_every=5000                                         # create a ckpt named `latest.pt` every a few steps.
epochs=100                                                     # total training epochs


sh $(dirname "$0")/run_g_controlnet.sh \
    --task-flag ${task_flag} \
    --control-type ${control_type} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.018 \
    --predict-type v_prediction \
    --uncond-p 0.44 \
    --uncond-p-t5 0.44 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --results-dir ${results_dir} \
    --resume \
    --resume-module-root ${resume_module_root} \
    --epochs ${epochs} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 10 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --gradient-checkpointing \
    "$@"

```

Recommended parameter settings

|     Parameter     |  Description  |          Recommended Parameter Value                               | Note|
|:---------------:|:---------:|:---------------------------------------------------:|:--:|
|   `--batch-size` |    Training batch size    |        1        | Depends on GPU memory|
|   `--grad-accu-steps` |    Size of gradient accumulation    |       2        | - |
|   `--lr` |    Learning rate  |        0.0001        | - |
|   `--control-type` |   ControlNet condition type, support 3 types now (canny, depth and pose)  |        /        | - |


### Inference
You can use the following command line for inference.

a. You can use a float to specify the weight for all layers, **or use a list to separately specify the weight for each layer**, for example, '[1.0 * (0.825 ** float(19 - i)) for i in range(19)]'
```bash
python sample_controlnet.py --infer-mode fa --control-weight "[1.0 * (0.825 ** float(19 - i)) for i in range(19)]" --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg
```

b. Using canny ControlNet during inference

```bash
python sample_controlnet.py --infer-mode fa --control-weight 1.0 --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg
```

c. Using depth ControlNet during inference

```bash
python sample_controlnet.py --infer-mode fa --control-weight 1.0 --no-enhance --load-key distill --infer-steps 50 --control-type depth --prompt "在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足。照片采用特写、平视和居中构图的方式，呈现出写实的效果" --condition-image-path controlnet/asset/input/depth.jpg
```

d. Using pose ControlNet during inference


```bash
python3 sample_controlnet.py --infer-mode fa --control-weight 1.0 --no-enhance --load-key distill --infer-steps 50 --control-type pose --prompt "在白天的森林中，一位穿着绿色上衣的亚洲女性站在大象旁边。照片采用了中景、平视和居中构图的方式，呈现出写实的效果。这张照片蕴含了人物摄影文化，并展现了宁静的氛围" --condition-image-path controlnet/asset/input/pose.jpg
```

## HunyuanDiT Controlnet v1.1

### Instructions
Download the v1.1 base model and controlnet using the following commands:
```bash
cd HunyuanDiT
# Use the huggingface-cli tool to download the model.
# We recommend using distilled weights as the base model for ControlNet inference, as our provided pretrained weights are trained on them.
huggingface-cli download Tencent-Hunyuan/HYDiT-ControlNet-v1.1 --local-dir ./HunyuanDiT-v1.1/t2i/controlnet
huggingface-cli download Tencent-Hunyuan/Distillation-v1.1 ./pytorch_model_distill.pt --local-dir ./HunyuanDiT-v1.1/t2i/model
```

### Training

```bash
task_flag="canny_controlnet"                                # the task flag is used to identify folders.
control_type=canny
resume_module_root=./ckpts/t2i/model/pytorch_model_distill.pt  # checkpoint root for resume
index_file=/path/to/your/indexfile                           # index file for dataloader
results_dir=./log_EXP                                        # save root for results
batch_size=1                                                 # training batch size
image_size=1024                                              # training image resolution
grad_accu_steps=2                                            # gradient accumulation
warmup_num_steps=0                                           # warm-up steps
lr=0.0001                                                    # learning rate
ckpt_every=10000                                             # create a ckpt every a few steps.
ckpt_latest_every=5000                                       # create a ckpt named `latest.pt` every a few steps.
epochs=100                                                   # total training epochs


sh $(dirname "$0")/run_g_controlnet.sh \
    --task-flag ${task_flag} \
    --control-type ${control_type} \
    --noise-schedule scaled_linear --beta-start 0.00085 --beta-end 0.03 \
    --predict-type v_prediction \
    --multireso \
    --reso-step 64 \
    --uncond-p 0.44 \
    --uncond-p-t5 0.44 \
    --index-file ${index_file} \
    --random-flip \
    --lr ${lr} \
    --batch-size ${batch_size} \
    --image-size ${image_size} \
    --global-seed 999 \
    --grad-accu-steps ${grad_accu_steps} \
    --warmup-num-steps ${warmup_num_steps} \
    --use-flash-attn \
    --use-fp16 \
    --results-dir ${results_dir} \
    --resume \
    --resume-module-root ${resume_module_root} \
    --epochs ${epochs} \
    --ckpt-every ${ckpt_every} \
    --ckpt-latest-every ${ckpt_latest_every} \
    --log-every 10 \
    --deepspeed \
    --deepspeed-optimizer \
    --use-zero-stage 2 \
    --use-style-cond \
    --size-cond 1024 1024 \
    "$@"
```

### Inference
You can use the following command line for inference.

a. Using canny ControlNet during inference

```bash
python3 sample_controlnet.py  --no-enhance --load-key distill --infer-steps 50 --control-type canny --prompt "在夜晚的酒店门前，一座古老的中国风格的狮子雕像矗立着，它的眼睛闪烁着光芒，仿佛在守护着这座建筑。背景是夜晚的酒店前，构图方式是特写，平视，居中构图。这张照片呈现了真实摄影风格，蕴含了中国雕塑文化，同时展现了神秘氛围" --condition-image-path controlnet/asset/input/canny.jpg --control-weight 1.0 --use-style-cond --size-cond 1024 1024 --beta-end 0.03
```

b. Using depth ControlNet during inference

```bash
python3 sample_controlnet.py  --no-enhance --load-key distill --infer-steps 50 --control-type depth --prompt "在茂密的森林中，一只黑白相间的熊猫静静地坐在绿树红花中，周围是山川和海洋。背景是白天的森林，光线充足" --condition-image-path controlnet/asset/input/depth.jpg --control-weight 1.0 --use-style-cond --size-cond 1024 1024 --beta-end 0.03
```

c. Using pose ControlNet during inference

```bash
python3 sample_controlnet.py  --no-enhance --load-key distill --infer-steps 50 --control-type pose --prompt "一位亚洲女性，身穿绿色上衣，戴着紫色头巾和紫色围巾，站在黑板前。背景是黑板。照片采用近景、平视和居中构图的方式呈现真实摄影风格" --condition-image-path controlnet/asset/input/pose.jpg --control-weight 1.0 --use-style-cond --size-cond 1024 1024 --beta-end 0.03
```
